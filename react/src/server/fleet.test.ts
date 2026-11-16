import { describe, expect, it } from "vitest";
import { composeAgainstFleet, fleetProfileFromPayload, fleetProvider } from "./fleet";
import { fleetSelection, providerSelection } from "./providers";

const payload = {
  namespace: "flakegraph",
  config_map: "flakegraph-config",
  parsing_endpoint: "http://flakegraph-ocr:8080",
  ontology: { name: "general", mode: "hybrid" },
  config: {
    ocr: {
      provider: "fallback",
      fallback_primary_provider: "builtin_text",
      fallback_secondary_provider: "mineru_api",
      mineru_method: "ocr",
      mineru_api_url: "${KG_MINERU_API_URL}",
      timeout_seconds: 3600,
    },
    llm: { provider: "vllm_local", endpoint: "${KG_LLM_ENDPOINT}", model: "llm-batch", api_key: "${KG_LLM_API_KEY}" },
    embedding: { provider: "sentence_transformers", model: "sentence-transformers/all-MiniLM-L6-v2", dimension: 384 },
    graph: { fail_on_quality_error: true },
    writer: { provider: "local_artifacts", output_path: "/data/output" },
  },
};

describe("fleet profile", () => {
  it("carries the generic HTTP OCR schema and reads its endpoint into the form", () => {
    const contract = { endpoint: "https://ocr.example.test/parse", api_key: "${KG_OCR_KEY}", text_field: "payload.text", page_field: "payload.pages", api_key_header: "X-Key", api_key_prefix: "", timeout_seconds: 45, max_response_bytes: 1024 };
    const profile = fleetProfileFromPayload({ ...payload, config: { ...payload.config, ocr: { provider: "generic_http" }, generic_http_ocr: contract } })!;
    const selection = fleetSelection(profile, "ocr", providerSelection("ocr", "builtin_text"));
    expect(selection).toMatchObject({ provider: "generic_http", endpoint: contract.endpoint });
    const config = { generic_http_ocr: { endpoint: contract.endpoint, max_response_bytes: 4096, text_field: "wrong" } };
    composeAgainstFleet(config, profile);
    expect(config.generic_http_ocr).toEqual({ ...contract, max_response_bytes: 4096 });
    expect(fleetSelection({ config: { ocr: { provider: "mineru_api", mineru_api_url: "https://mineru.example.test" } } }, "ocr", selection).endpoint).toBe("https://mineru.example.test");
  });
  it("reads what the workers mount", () => {
    const profile = fleetProfileFromPayload(payload);
    expect(profile?.configMap).toBe("flakegraph-config");
    expect(profile?.parsingEndpoint).toBe("http://flakegraph-ocr:8080");
    expect(fleetProvider(profile!, "llm")).toEqual({ provider: "vllm_local", model: "llm-batch", dimension: null });
    expect(fleetProvider(profile!, "embedding")?.dimension).toBe(384);
  });

  it("rejects a payload without a profile", () => {
    expect(fleetProfileFromPayload({ namespace: "flakegraph" })).toBeNull();
  });

  it("composes a run the workers will claim, keeping what the digest excludes", () => {
    const profile = fleetProfileFromPayload(payload)!;
    const config: Record<string, unknown> = {
      ocr: { provider: "builtin_text", timeout_seconds: 900 },
      llm: { provider: "openai_compatible", model: "gpt", endpoint: "http://elsewhere", timeout_seconds: 120 },
      embedding: { provider: "sentence_transformers", model: "other", dimension: 768, batch_size: 8 },
      graph: { extraction_parallelism: 4, fail_on_quality_error: false },
      ontology: { profile: { entity_types: ["X"] } },
      writer: { provider: "local_artifacts", output_path: "/run" },
    };

    composeAgainstFleet(config, profile);

    expect(config.ocr).toEqual({ ...payload.config.ocr, mineru_api_url: "http://flakegraph-ocr:8080", timeout_seconds: 900 });
    expect(config.llm).toEqual({ ...payload.config.llm, timeout_seconds: 120 });
    expect(config.embedding).toEqual({ ...payload.config.embedding, batch_size: 8 });
    expect(config.graph).toEqual({ fail_on_quality_error: true, extraction_parallelism: 4 });
    // The run's own vocabulary is kept; it is the run's, not the fleet's.
    expect(config.ontology).toEqual({ profile: { entity_types: ["X"] } });
    expect(config.writer).toEqual({ provider: "local_artifacts", output_path: "/run" });
  });

  it("gives a run that names no vocabulary the one the fleet mounts", () => {
    const mounted = fleetProfileFromPayload(payload)!;
    const config: Record<string, unknown> = { ontology: { profile_path: "configs/ontologies/general.yaml" } };

    composeAgainstFleet(config, mounted);

    expect(config.ontology).toEqual({ profile: { name: "general", mode: "hybrid" } });
  });

  it("drops a path-only ontology when the fleet mounts none", () => {
    const profile = fleetProfileFromPayload({ ...payload, ontology: null })!;
    const config: Record<string, unknown> = { ontology: { profile_path: "configs/ontologies/general.yaml" } };

    composeAgainstFleet(config, profile);

    expect(config.ontology).toBeUndefined();
  });
});
