import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { authorizeRequest, mayMutate, RequestRefused } from "./auth";
import { resetAppEnv } from "./env";
import { assumeIdentity, createApiKey } from "./workspace";

const original = { ...process.env };

afterEach(() => {
  process.env = { ...original };
  resetAppEnv();
});

async function console_(flags: Record<string, string>) {
  process.env.FLAKEGRAPH_APP_STATE_ROOT = await mkdtemp(path.join(tmpdir(), "flakegraph-auth-"));
  delete process.env.FLAKEGRAPH_TRUST_IDENTITY_HEADERS;
  delete process.env.FLAKEGRAPH_APP_REQUIRE_SIGN_IN;
  delete process.env.FLAKEGRAPH_DEMO_IDENTITIES;
  delete process.env.FLAKEGRAPH_SINGLE_OPERATOR;
  delete process.env.FLAKEGRAPH_APP_ANONYMOUS_READ;
  Object.assign(process.env, flags);
  resetAppEnv();
}

describe("authorizeRequest", () => {
  it("refuses a caller nobody identified, by default", async () => {
    await console_({});
    await expect(authorizeRequest(new Headers())).rejects.toBeInstanceOf(RequestRefused);
    // Identity headers are not believed where no gate is trusted.
    await expect(authorizeRequest(new Headers({ "x-auth-request-email": "mallory@example.test" }))).rejects.toThrow(
      /Sign in/,
    );
  });

  it("pins an explicitly isolated console to the configured operator", async () => {
    await console_({ FLAKEGRAPH_SINGLE_OPERATOR: "operator@example.test" });
    await assumeIdentity({ userName: "someone-else", roles: [], role: "operator" });
    const authorized = await authorizeRequest(new Headers({ "x-auth-request-email": "mallory@example.test" }));
    expect(authorized.mode).toBe("operator");
    expect(authorized.viewer.userName).toBe("OPERATOR@EXAMPLE.TEST");
    expect(authorized.assumed).toBe(false);
    const { secret } = await createApiKey("initial-key", authorized.viewer.userName);
    expect((await authorizeRequest(new Headers({ authorization: `Bearer ${secret}` }))).mode).toBe("machine");
  });

  it("refuses isolated operator configuration mixed with trusted identity or required sign-in", async () => {
    for (const name of ["FLAKEGRAPH_TRUST_IDENTITY_HEADERS", "FLAKEGRAPH_APP_REQUIRE_SIGN_IN", "FLAKEGRAPH_DEMO_IDENTITIES"]) {
      await console_({ FLAKEGRAPH_SINGLE_OPERATOR: "operator@example.test", [name]: "1" });
      await expect(authorizeRequest(new Headers())).rejects.toThrow(/isolated server/);
    }
  });

  it("serves the gate's viewer, and refuses a caller the gate never saw", async () => {
    await console_({ FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1" });
    const gated = await authorizeRequest(new Headers({ "x-auth-request-email": "alice@example.test" }));
    expect(gated.viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    expect(gated.mode).toBe("gate");
    expect(gated.machine).toBeNull();
    expect(mayMutate(gated)).toBe(true);
    await expect(authorizeRequest(new Headers())).rejects.toBeInstanceOf(RequestRefused);
  });

  it("serves a caller the gate admitted on a bearer token as the person it names", async () => {
    await console_({ FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1" });
    const delegated = await authorizeRequest(
      new Headers({ authorization: "Bearer eyJhbGciOiJSUzI1NiJ9.e30.sig", "x-auth-request-email": "alice@example.test" }),
    );
    expect(delegated.viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    expect(delegated.mode).toBe("gate");
    expect(delegated.machine).toBeNull();
  });

  it("serves read-only queries to an anonymous caller only where the deployment opened them", async () => {
    await console_({ FLAKEGRAPH_APP_ANONYMOUS_READ: "1" });
    const open = await authorizeRequest(new Headers());
    expect(open.viewer.userName).toBe("");
    expect(open.mode).toBe("anonymous");
    expect(mayMutate(open)).toBe(false);
    // A required sign-in closes even that.
    await console_({ FLAKEGRAPH_APP_ANONYMOUS_READ: "1", FLAKEGRAPH_APP_REQUIRE_SIGN_IN: "1" });
    await expect(authorizeRequest(new Headers())).rejects.toBeInstanceOf(RequestRefused);
  });

  it("serves demo identities on a console that asked for them, assumed or not", async () => {
    await console_({ FLAKEGRAPH_DEMO_IDENTITIES: "1" });
    const nobody = await authorizeRequest(new Headers());
    expect(nobody.mode).toBe("demo");
    expect(nobody.assumed).toBe(false);
    expect(mayMutate(nobody)).toBe(true);
    await assumeIdentity({ userName: "alice", roles: ["app_operator"], role: "operator" });
    const alice = await authorizeRequest(new Headers());
    expect(alice.viewer).toEqual({ userName: "ALICE", email: "", roles: ["APP_OPERATOR"] });
    expect(alice.assumed).toBe(true);
    // Behind a trusted gate the workspace's assumed identity is not consulted.
    await console_({ FLAKEGRAPH_DEMO_IDENTITIES: "1", FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1" });
    await assumeIdentity({ userName: "alice", roles: [], role: "operator" });
    await expect(authorizeRequest(new Headers())).rejects.toBeInstanceOf(RequestRefused);
  });

  it("serves a machine as the principal that minted its key, and refuses a key it does not hold", async () => {
    await console_({ FLAKEGRAPH_TRUST_IDENTITY_HEADERS: "1" });
    const { secret } = await createApiKey("ci", "alice@example.test");
    const machine = await authorizeRequest(new Headers({ authorization: `Bearer ${secret}` }));
    expect(machine.mode).toBe("machine");
    expect(machine.machine?.name).toBe("ci");
    expect(machine.machine?.owner).toBe("ALICE@EXAMPLE.TEST");
    expect(machine.viewer.userName).toBe("ALICE@EXAMPLE.TEST");
    expect(mayMutate(machine)).toBe(true);
    await expect(authorizeRequest(new Headers({ "x-flakegraph-api-key": "fg_nope" }))).rejects.toThrow(
      /missing or revoked/,
    );
  });
});
