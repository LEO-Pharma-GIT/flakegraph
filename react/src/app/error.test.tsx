import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import ErrorPage from "./error";

describe("the error fallback", () => {
  it.each([null, undefined, "failure", 42, { message: { secret: "object" } }, new Error("Useful details")])("renders controls for thrown value %j", (error) => {
    const html = renderToStaticMarkup(<ErrorPage error={error} reset={() => undefined} />);
    expect(html).toContain("Something went wrong");
    expect(html).toContain("Try again");
    expect(html).toContain("Back to graphs");
    if (error instanceof Error) expect(html).toContain("Useful details");
  });
});
