# Famous Deep Learning Papers Benchmark

This benchmark processes all 49 papers with the same corpus, ontology, fallback
OCR policy, all-MiniLM-L6-v2 embeddings, queue-backed Kubernetes execution, and
four Spark executors. Caches are disabled. The fallback selected built-in PDF
text for 45 Qwen documents and 43 GPT documents; MinerU processed the remaining
four and six documents respectively.

Qwen is served on four NVIDIA GB10 systems with NVIDIA vLLM. GPT-5.6 Sol uses a
managed Azure OpenAI deployment while FlakeGraph extraction and Spark
finalization run on the same four-node fleet. Each result is one complete,
uncached run, so measurements are not multi-run means or stability estimates.

The Qwen row was measured on `nvidia/Qwen3.6-35B-A3B-NVFP4`, which the
repository's profiles no longer select. The figure is kept as the record of
what ran, not as a prediction of what the current profiles will produce.
Re-measure before comparing against it.

Both rows were scored against gold set 1.0.0, whose digest each result record
carries. The gold set is now 1.0.1: it shortened the quoted `sentence` fields
to their evidence anchors and dropped a few over-long aliases, leaving ids,
entities, relations, anchors and pages unchanged (see the dataset README). The
alias change can move entity matching by a fraction of a point, so treat these
figures as measured against 1.0.0.

## Results

| Result | LLM serving | Wall time | Documents/minute | Entity F1 | Triple F1 | Hard errors | Quality gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| [`dlp-v1-qwen36-35b-a3b-nvfp4-vllm-k8s-4x-gb10-20260714`](results/qwen36-35b-a3b-nvfp4-vllm-kubernetes-4x-gb10-20260714.json) | 4x GB10, vLLM | 3h 42m 47.568s | 0.220 | 0.8986 | 0.5010 | 9 | Not met |
| [`dlp-v1-gpt56-sol-azure-openai-k8s-4x-gb10-20260715`](results/gpt56-sol-azure-openai-kubernetes-4x-gb10-20260715.json) | Managed Azure OpenAI | 49m 14.681s | 0.995 | 0.9678 | 0.8042 | 0 | Not met |

GPT-5.6 Sol completed **4.52x faster** and recovered more of the selected
reference graph. Both models produced valid grounded output outside the selected
reference; those records are unscored and do not reduce reference precision.
Neither result meets every exact-reference threshold. The Qwen graph also has
eight `PUBLISHED_BY` edges and one `IS_A` edge outside the configured ontology,
which account for its nine hard errors.

## Timing

Wall time runs from durable run creation through publication of the final graph
artifact. Pre-finalization includes source preparation, OCR, chunking, embeddings,
document context, entity extraction, relation extraction, queue barriers, and the
handoff to Spark.

| Phase | Qwen 3.6 | GPT-5.6 Sol | GPT speedup |
| --- | ---: | ---: | ---: |
| Pre-finalization and handoff | 2h 50m 9.678s | 36m 1.247s | 4.72x |
| Spark graph finalization and enrichment | 52m 37.891s | 13m 13.434s | 3.98x |
| End to end | 3h 42m 47.568s | 49m 14.681s | 4.52x |

Qwen identity resolution took 2,406.139 seconds for 3,114 LLM-adjudicated
candidate pairs. GPT identity resolution took 182.749 seconds for 3,301 pairs;
its community reporting took 374.299 seconds. Pre-finalization accounted for 76%
of Qwen wall time and 73% of GPT wall time.

## Quality

The selected gold graph contains 657 entities and 742 directed relations.
Precision is measured within that reference; additional grounded output is
reported as unscored rather than incorrect.

| Metric | Qwen 3.6 | GPT-5.6 Sol |
| --- | ---: | ---: |
| Entity precision | 1.0000 | 1.0000 |
| Entity recall | 0.8158 | 0.9376 |
| Entity F1 | 0.8986 | 0.9678 |
| Triple precision | 1.0000 | 1.0000 |
| Triple recall | 0.3342 | 0.6725 |
| Triple F1 | 0.5010 | 0.8042 |
| Evidence support | 0.2264 | 0.5674 |
| Information retention | 0.5750 | 0.8051 |
| Two-hop recoverability | 0.4461 | 0.7978 |
| Components | 8,085 | 14,243 |
| Isolate ratio | 0.7302 | 0.6153 |
| Unscored entities | 10,049 | 21,637 |
| Unscored relations | 3,004 | 10,544 |

## Output Size

| Output | Qwen 3.6 | GPT-5.6 Sol |
| --- | ---: | ---: |
| Documents | 49 | 49 |
| Pages | 866 | 866 |
| Chunks | 1,369 | 1,380 |
| Canonical entities | 10,758 | 22,466 |
| Canonical relations | 3,317 | 11,164 |
| Evidence records | 20,971 | 42,534 |
| Communities | 698 | 1,428 |

## Reproduction

Download the reviewed corpus as described in [`README.md`](README.md), deploy the
fleet profile, and submit it with caches disabled. Fleet deployment, monitoring,
export, and retry commands are documented in
[`docs/kubernetes-fleet.md`](../../docs/kubernetes-fleet.md). Compact measured
results are stored in [`results/`](results/).
