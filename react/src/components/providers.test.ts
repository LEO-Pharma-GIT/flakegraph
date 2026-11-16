import { describe, expect, it } from "vitest";
import { shouldRetry } from "./providers";

describe("query retries", () => {
  it("tries a failure once more, but never a refusal", () => {
    expect(shouldRetry(0, new Error("network"))).toBe(true);
    expect(shouldRetry(1, new Error("network"))).toBe(false);
    for (const code of ["FORBIDDEN", "NOT_FOUND", "UNAUTHORIZED", "BAD_REQUEST"]) {
      expect(shouldRetry(0, { data: { code } })).toBe(false);
    }
    expect(shouldRetry(0, { data: { code: "INTERNAL_SERVER_ERROR" } })).toBe(true);
  });
});
