import { describe, expect, it } from "vitest";
import { principalRole, roleFor, rolePrincipal, validPrincipal } from "./graph-access";
import { mayReadStagePath, SNOWFLAKE_ACCESS_POLICY, viewerStagePrefix, visibleGraphPredicate } from "./snowflake-access";

const viewer = (userName: string, roles: string[] = []) => ({ userName, email: "", roles });

describe("stage paths", () => {
  it("gives each viewer a folder of their own, whatever their name holds", () => {
    expect(viewerStagePrefix(viewer("ALICE"), "run-1")).toBe("app/ALICE/run-1");
    expect(viewerStagePrefix(viewer("A/../BOB"), "run-1")).toBe("app/A_.._BOB/run-1");
    expect(viewerStagePrefix(viewer(""), "run-1")).toBe("app/shared/run-1");
  });

  it("opens a curated folder to everyone and an upload folder to its uploader", () => {
    expect(mayReadStagePath(viewer("BOB"), "corpus/2026")).toBe(true);
    expect(mayReadStagePath(viewer("ALICE"), "app/ALICE/run-1")).toBe(true);
    expect(mayReadStagePath(viewer("BOB"), "app/ALICE/run-1")).toBe(false);
    expect(mayReadStagePath(viewer("BOB"), "app")).toBe(false);
    expect(mayReadStagePath(viewer("BOB"), "")).toBe(false);
  });

  it("reads a path as the stage does, so no spelling reaches another's folder", () => {
    expect(mayReadStagePath(viewer("BOB"), "./app/ALICE")).toBe(false);
    expect(mayReadStagePath(viewer("BOB"), "//app//ALICE/")).toBe(false);
  });
});

describe("the visibility predicate", () => {
  it("binds who the viewer is and names only their roles", () => {
    const { sql, parameters } = visibleGraphPredicate(viewer("ALICE", ["B_ROLE", "A_ROLE"]));
    expect(sql).not.toContain("ALICE");
    expect(parameters).toEqual(["ALICE", "ALICE", "A_ROLE", "B_ROLE"]);
  });

  it("shows nobody only what nobody owns", () => {
    expect(visibleGraphPredicate(viewer(""))).toEqual({ sql: "G.OWNER IS NULL", parameters: [] });
  });
});

describe("role grants", () => {
  const record = { owner: "ALICE", grants: [
    { principal: rolePrincipal("analysts"), level: "read" as const, grantedBy: "ALICE", grantedAt: "" },
    { principal: "BOB", level: "write" as const, grantedBy: "ALICE", grantedAt: "" },
  ] };

  it("reaches whoever holds the role, and the better of two grants wins", () => {
    expect(roleFor(record, viewer("CAROL", ["ANALYSTS"]), SNOWFLAKE_ACCESS_POLICY)).toBe("read");
    expect(roleFor(record, viewer("BOB", ["ANALYSTS"]), SNOWFLAKE_ACCESS_POLICY)).toBe("write");
    expect(roleFor(record, viewer("DAVE", ["OTHERS"]), SNOWFLAKE_ACCESS_POLICY)).toBeNull();
  });

  it("does not admit a user who happens to share the role's name", () => {
    expect(roleFor(record, viewer("ANALYSTS"), SNOWFLAKE_ACCESS_POLICY)).toBeNull();
    expect(visibleGraphPredicate(viewer("ANALYSTS")).sql).not.toContain("'ROLE'");
  });

  it("cannot be spelled as a person's name", () => {
    expect(principalRole("role:analysts")).toBe("ANALYSTS");
    expect(principalRole("ALICE")).toBeNull();
    expect(validPrincipal("ROLE:ANALYSTS", SNOWFLAKE_ACCESS_POLICY)).toBeNull();
    // A person whose name is the word ROLE stays a person.
    expect(roleFor(record, viewer("ROLE:ANALYSTS"), SNOWFLAKE_ACCESS_POLICY)).toBe("read");
  });
});
