# Benchmark Datasets

Each benchmark dataset is packaged under `data/`. Public-domain sources can live
beside their annotations; externally licensed sources instead provide a downloader
that materializes ignored files locally. Keeping the catalog, annotations,
ontology, terms, and runnable profiles together makes either form portable without
redistributing content the repository does not own.

```text
data/<dataset_id>/
  README.md       scope, provenance, counts, and usage
  BENCHMARKS.md   dataset-specific measurements and protocol
  LICENSE.md      license for the dataset artifacts
  manifest.jsonl  source-file metadata and checksums
  gold.json       annotated entities, relations, and evidence
  ontology.yaml   extraction vocabulary and relation constraints
  files/          source documents consumed by FlakeGraph
  configs/        dataset-specific local and distributed profiles
  results/        compact published benchmark measurements
```

Provider examples that are useful across datasets remain in `configs/`. Profiles tied
to one benchmark corpus belong with that dataset and should use repository-relative
paths so commands work from the project root.

```mermaid
flowchart LR
    files["Source files"] --> manifest["Manifest and checksums"]
    ontology["Dataset ontology"] --> configs["Runnable profiles"]
    manifest --> configs
    configs --> extractedGraph["Extracted graph"]
    gold["Gold annotations"] --> evaluation["Evaluation"]
    extractedGraph --> evaluation
    evaluation --> results["Published results"]
```

## Available Datasets

| Dataset | Documents | Gold version | Evaluation scope | Gold graph | License |
| --- | ---: | --- | --- | ---: | --- |
| [Martial Arts History](martial_arts/README.md) | 10 | 2.1.0 | Exhaustive | 119 entities / 148 relations | CC0-1.0 |
| [Famous Deep Learning Papers](deep_learning_papers/README.md) | 49 | 1.0.1 | Reference | 657 entities / 742 relations | Source-specific |
| [Atopic Dermatitis](atopic_dermatitis/README.md) | 50 | 1.1.0 | Exhaustive | 3,462 entities / 5,762 relations | CC BY / CC0 |

## Benchmark Protocol

Comparable result records identify the dataset version, ontology, complete
processing configuration, model artifact, serving runtime, hardware, and repeat
count. A published measurement should:

- process the complete manifest with cross-run caches disabled;
- warm model services before timing;
- include at least three successful repetitions;
- report quality, evidence, topology, timing, throughput, and stability metrics;
- keep each run's output isolated; and
- omit credentials, private endpoints, usernames, and hostnames.

Run the local protocol with:

```bash
uv run flakegraph benchmark extraction \
  --config data/martial_arts/configs/local-vllm.yaml \
  --gold data/martial_arts/gold.json \
  --repeats 3 \
  --output out/benchmarks/martial-arts
```

Each dataset publishes its measurements in `BENCHMARKS.md` and compact
machine-readable records under `results/`.
