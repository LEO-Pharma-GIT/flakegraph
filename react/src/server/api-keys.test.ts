import { describe, expect, it } from "vitest";
import { hashApiSecret, matchApiKey, presentedApiSecret, publicApiKeys } from "./api-keys";

describe("api keys", () => {
  it("matches a hashed secret and ignores SSO-shaped keys", () => {
    const secret = "fg_abcdefghijklmnopqrstuvwxyz12";
    const keys = [{ id: "key_1", secretHash: hashApiSecret(secret) }];
    expect(matchApiKey(keys, secret)?.id).toBe("key_1");
    expect(matchApiKey(keys, "fg_not-this-key")).toBeNull();
    expect(matchApiKey(keys, "not-a-flakegraph-key")).toBeNull();
  });

  it("reads Bearer and x-flakegraph-api-key", () => {
    expect(presentedApiSecret(new Headers({ authorization: `Bearer fg_abc` }))).toBe("fg_abc");
    expect(presentedApiSecret(new Headers({ "x-flakegraph-api-key": "fg_from_header" }))).toBe("fg_from_header");
    expect(presentedApiSecret(new Headers())).toBeNull();
  });

  it("leaves a bearer token that is not a key to the gate", () => {
    expect(presentedApiSecret(new Headers({ authorization: "Bearer eyJhbGciOiJSUzI1NiJ9.e30.sig" }))).toBeNull();
  });

  it("strips secret hashes from public key records", () => {
    const publicKeys = publicApiKeys([{ id: "key_1", name: "ci", preview: "fg_abc…1234", createdAt: "", note: "", secretHash: "deadbeef" }]);
    expect(publicKeys[0]).not.toHaveProperty("secretHash");
    expect(publicKeys[0]?.id).toBe("key_1");
  });
});
