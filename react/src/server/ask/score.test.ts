import { describe, expect, it } from "vitest";
import { scoreHaystack, terms, tokenize } from "./score";

describe("search terms", () => {
  it("keeps two-letter codes and any token with a digit", () => {
    expect(terms("Sales in the EU and UK rose 5% in Q3 2024")).toEqual(["sale", "eu", "uk", "rose", "5", "q3", "2024"]);
  });

  it("drops stopwords and lone letters", () => {
    expect(terms("Who is the owner of a plan?")).toEqual(["owner", "plan"]);
    expect(tokenize("who is the")).toEqual([]);
  });

  it("folds case and diacritics, in any script", () => {
    expect(terms("Café CRÈME naïve Über")).toEqual(["cafe", "creme", "naive", "uber"]);
    expect(terms("Smørrebrød på Nørrebro")).toEqual(["smørrebrød", "pa", "nørrebro"]);
    expect(terms("東京 2020")).toEqual(["東京", "2020"]);
  });

  it("folds plurals and possessives to one term, and leaves singular words alone", () => {
    expect(terms("studies study companies company's lanterns")).toEqual(["study", "study", "company", "company", "lantern"]);
    expect(terms("class status analysis bus gas")).toEqual(["class", "status", "analysis", "bus", "gas"]);
    expect(terms("classes")).toEqual(["class"]);
  });

  it("does not stem numbers", () => {
    expect(terms("1990s 2s")).toEqual(["1990s", "2s"]);
  });
});

describe("keyword scoring", () => {
  // Under the old scorer "art" matched inside "start" and "EU" was no term at all.
  it("matches whole tokens, never part of a word", () => {
    expect(scoreHaystack("We start the quarter early.", tokenize("art"))).toBe(0);
    expect(scoreHaystack("Street art festival", tokenize("art"))).toBeGreaterThan(0);
    expect(scoreHaystack("Revenue in the EU grew.", tokenize("EU revenue"))).toBeGreaterThan(
      scoreHaystack("Revenue grew.", tokenize("EU revenue")),
    );
  });

  it("ranks more distinct matched terms above repeats of one", () => {
    const query = tokenize("harbour lantern");
    expect(scoreHaystack("harbour lantern", query)).toBeGreaterThan(scoreHaystack("harbour harbour harbour", query));
  });
});
