import { describe, expect, it } from "vitest";
import { defaultProvider, embeddingDimension, providerOptions, providerSelection } from "./providers";

describe("provider defaults by runtime", () => {
  it("starts a Snowflake run on Cortex throughout, as the Snowflake profile does", () => {
    expect(defaultProvider("ocr", "snowflake").provider).toBe("snowflake_cortex");
    expect(defaultProvider("llm", "snowflake")).toMatchObject({ provider: "snowflake_cortex", model: "mistral-large2" });
    // Without an answer from the account, the width this deployment's tables use.
    expect(defaultProvider("embedding", "snowflake")).toMatchObject({
      provider: "snowflake_cortex",
      model: "snowflake-arctic-embed-l-v2.0",
      dimension: 1024,
    });
  });

  it("fits the Cortex embedding model to the width the graph tables store", () => {
    expect(defaultProvider("embedding", "snowflake", 768)).toMatchObject({
      model: "snowflake-arctic-embed-m-v1.5",
      dimension: 768,
    });
  });

  it("offers only Cortex inside Snowflake, where nothing else is reachable", () => {
    for (const kind of ["ocr", "llm", "embedding"] as const) {
      expect(providerOptions(kind, "snowflake").map((option) => option.name)).toEqual(["snowflake_cortex"]);
      expect(providerOptions(kind, "local").length).toBeGreaterThan(1);
    }
    expect(providerSelection("llm", "vllm_local", "snowflake").provider).toBe("snowflake_cortex");
  });

  it("keeps the local defaults unchanged", () => {
    expect(defaultProvider("ocr").provider).toBe("builtin_text");
    expect(defaultProvider("llm")).toMatchObject({ provider: "ollama", model: "qwen3:4b" });
    expect(defaultProvider("embedding")).toMatchObject({ provider: "sentence_transformers", dimension: 1024 });
  });

  it("reads a known model's width before the provider's default", () => {
    expect(embeddingDimension("snowflake_cortex", "snowflake-arctic-embed-l-v2.0", null)).toBe(1024);
    expect(embeddingDimension("snowflake_cortex", null, null)).toBe(768);
    expect(embeddingDimension("snowflake_cortex", "custom", 512)).toBe(512);
  });
});
