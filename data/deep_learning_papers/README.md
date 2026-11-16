# Famous Deep Learning Papers Benchmark

This benchmark uses the 49 primary papers listed by the
[Bau Lab reading list](https://papers.baulab.info/). The repository contains the
download workflow, annotations, configurations, and compact measurements, but
does not redistribute the papers. Copyright and usage terms remain with each
paper's respective authors and publisher.

## Download

```bash
uv run python data/deep_learning_papers/download.py
```

The command downloads the versioned 49-paper source catalog embedded in
`gold.json` into the ignored `files/` directory, validates PDF signatures and
reviewed checksums, and writes an ignored portable `manifest.jsonl` for dataset
inventory and manifest-source workflows. The bundled fleet profile reads the
downloaded `files/` directory directly. Existing valid files are reused; pass
`--refresh` to download them again. `--discover-index` reads the mutable live
table and is intended only for a deliberate benchmark update and review.

The downloader resolves two known stale links in the source index to the
corresponding Bottou and Mikolov PDFs hosted by the same site.

## Contents

| Path | Purpose |
| --- | --- |
| `download.py` | Discovers, validates, and downloads the non-redistributed corpus. |
| `ontology-v2.yaml` | Scientific-paper entity and relation vocabulary, reading guidance, and counter-examples. Use it with `configs/kubernetes-vllm-v2.yaml` for new runs. |
| `configs/kubernetes-vllm-v2.yaml` | Queue-backed Qwen and MinerU fleet profile using `ontology-v2.yaml`. |
| `ontology.yaml`, `configs/kubernetes-vllm.yaml` | The vocabulary and profile the published `results/` were measured with, kept unchanged so their checksums stay true. |
| `gold.json` | Selected reference graph and short evidence anchors. |
| `BENCHMARKS.md` | Controlled GPT-5.6 Sol and Qwen measurements. |
| `results/` | Compact public benchmark records without source content. |
| `files/` | Ignored downloaded PDFs. |
| `manifest.jsonl` | Ignored generated source manifest and checksums. |

## Annotation Contract

The gold graph is a selected typed reference, not an exhaustive transcription
of every paper. Each entity must be explicitly identified in a source and match
its ontology definition; bounded datasets, tasks, methods, and experimental
systems need not be proper nouns. Each relation must satisfy `ontology.yaml` and
include a phrase of at most ten words plus its physical PDF page. The current
reference contains 657 entities, 742 directed relations, and 742 independently
located evidence observations. Citation proximity alone does not establish
`BUILDS_ON`, and generic unnamed baselines are not canonical entities.

## What Is Quoted From The Papers

The papers are copyrighted and are not redistributed. The only verbatim text
`gold.json` carries is the evidence anchor of each observation: a phrase of at
most ten words that locates the fact on the cited page. The `sentence` field
holds the same anchor, not the surrounding sentence, and entity names and
aliases are FlakeGraph's own canonical labels or phrases of at most ten words.
Paper titles appear as bibliographic identifiers. A contract test
(`tests/unit/test_deep_learning_papers_contract.py`) enforces the ten-word
limit on every text field in the file. Version 1.0.1 of the gold set applied
that rule to the `sentence` field and to a handful of aliases that had exceeded
it; ids, entities, relations, anchors and page numbers are unchanged from
1.0.0, which is the version the published benchmark measurements were scored
against. See `LICENSE.md` for the terms that apply to the quoted anchors.

Changes require source-page review, endpoint and ontology validation, checksum
verification for all 49 documents, and a full benchmark rerun. Provider output
may reveal a missing valid fact, but it is never copied into the gold graph
without independent confirmation in the cited paper.
