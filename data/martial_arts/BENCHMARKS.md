# Martial Arts Benchmark

This benchmark processes all ten documents with built-in text extraction,
all-MiniLM-L6-v2 embeddings, disabled cross-run caches, two-pass entity and
relation extraction, and Spark GraphFrames finalization. Qwen is served through
NVIDIA vLLM; GPT-5.6 Sol uses Azure OpenAI.

The three 2026-07-15 rows each aggregate three complete queue-backed
repetitions on `nvidia/Qwen3.6-35B-A3B-NVFP4`, which the repository's profiles
no longer select. The 2026-08-27 row is a **single** repetition on the current
profile — `unsloth/Qwen3.8-27B-NVFP4` behind the priority-aware serving plane —
so it carries no stability estimate and its wall time is not comparable with the
rows above it. Read the two groups as different measurements, not as a series.

## Results on gold 2.1.0

Gold 2.1.0 is exhaustive, so these rows carry real precision and F1.

| Result | Environment | Runs | Entity P / R / F1 | Triple P / R / F1 |
| --- | --- | ---: | --- | --- |
| [`mah-v2.1-gpt6-astra-azure-openai-local-3x-20260925`](results/gpt6-astra-azure-openai-local-3x-20260925.json) | Local, Azure OpenAI | 3 (mean) | 0.471 / 0.961 / 0.632 | 0.734 / 0.933 / 0.822 |
| [`mah-v2.1-qwen38-27b-nvfp4-vllm-k8s-6x-gb10-20260925`](results/qwen38-27b-nvfp4-vllm-kubernetes-6x-gb10-20260925.json) | Kubernetes, 6x GB10, the console's sample-pack configuration | 1 | 0.794 / 0.647 / 0.713 | 0.933 / 0.667 / 0.778 |

GPT-6 astra finds nearly everything and a good deal more: around 270 entities
against 119 in the gold, many of them generic phrases the ontology does not
admit, so its entity precision is low. Qwen 3.8 is the more precise extractor
on both entities and relations and trades recall for it. Across the three
astra runs the mean pairwise overlap is 0.875 for entity names and 0.846 for
triples: on this short corpus a model repeats itself far more than on long
papers. What the astra runs found went into the gold where the text supports
it, which favours their recall somewhat.

## Results on gold 1.0.0

| Result | Environment | Mean Wall Time | Documents/Minute | Entity F1 | Triple F1 | Accepted Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [`mah-v1-qwen36-35b-a3b-nvfp4-vllm-k8s-1x-gb10-20260715`](results/qwen36-35b-a3b-nvfp4-vllm-kubernetes-1x-gb10-20260715.json) | Kubernetes, 1x GB10 | 18m 3.917s | 0.558 | 0.9909 | 0.9263 | 1/3 |
| [`mah-v1-qwen36-35b-a3b-nvfp4-vllm-k8s-4x-gb10-20260715`](results/qwen36-35b-a3b-nvfp4-vllm-kubernetes-4x-gb10-20260715.json) | Kubernetes, 4x GB10 | 7m 56.188s | 1.266 | 0.9693 | 0.9065 | 0/3 |
| [`mah-v1-gpt56-sol-azure-openai-k8s-4x-gb10-20260715`](results/gpt56-sol-azure-openai-kubernetes-4x-gb10-20260715.json) | Kubernetes, 4x GB10, Azure OpenAI | 4m 0.173s | 2.562 | 0.9955 | 0.9592 | 0/3 |
| [`mah-v1-qwen38-27b-nvfp4-vllm-k8s-1x-gb10-20260827`](results/qwen38-27b-nvfp4-vllm-kubernetes-1x-gb10-20260827.json) | Kubernetes, 1x GB10, serving plane | 58m extraction (see below) | 0.172 | recall 1.0 | recall 1.0 | 1/1 |

Four-node Qwen is **2.28x faster than one-node Qwen** end to end. GPT-5.6 Sol
is **1.98x faster than four-node Qwen** on the same processing fleet. All nine
2026-07-15 runs completed successfully; one Qwen 1x task retried after a
malformed model response. The strict per-run acceptance gate includes structural
thresholds, so mean F1 alone does not determine whether a run is accepted.

### Reading the 2026-08-27 row

It is the first run to pass every acceptance gate: 74 nodes and 104 edges
against a gold graph of exactly 74 and 104, recall 1.0 on both entities and
triples, one component, no isolates. The evaluator reports precision as not
applicable under reference entity coverage, so precision and F1 are recorded as
null rather than assumed, and only recall is quoted above.

Its timing needs three caveats. Extraction ran uninterrupted in **3,479 s**, and
that is the figure the throughput column uses. Total wall time was 6,553 s, but
finalization stalled twice on operator-fixable faults — a Spark image predating
the coordination schema, then a missing provider Secret the executors reference
— so the total measures an incident, not the fleet.

The throughput gap against the 2026-07-15 rows is mostly architectural, not a
regression in the pipeline. `Qwen3.6-35B-A3B` activates about 3B parameters per
token; `Qwen3.8-27B` is dense and activates all 27B. Decode on this part is
bound by memory bandwidth, so the newer model measured **10.7 tokens/s** where
the older one sustained several times that. Expect the same ratio on any
bandwidth-bound accelerator, and size `maxNumSeqs` and client timeouts for it.

## Protocol

| Setting | Qwen 1x GB10 | Qwen 4x GB10 | GPT-5.6 Sol |
| --- | ---: | ---: | ---: |
| Model replicas | 1 | 4 | Managed |
| Execution | Kubernetes | Kubernetes | Kubernetes |
| Peak preparation workers | 10 | 10 | 10 |
| Maximum extraction workers | 3 | 12 | 12 |
| Spark executors | 1 | 4 | 4 |
| Spark executor cores | 4 | 4 | 4 |
| vLLM GPU memory utilization | 0.50 | 0.50 | N/A |
| Preparation tasks | 10 | 10 | 10 |
| Document-context tasks | 10 | 10 | 10 |
| Entity-window tasks | 12 | 12 | 12 |
| Relation-window tasks | 12 | 12 | 12 |
| Compaction tasks | 20 | 20 | 20 |
| Finalization tasks | 1 | 1 | 1 |
| Failed task attempts | 1 | 0 | 0 |
| Repetitions | 3 | 3 | 3 |

The Qwen profiles form the controlled fleet-scaling comparison. Four-node Qwen
and GPT use the same corpus, OCR, embeddings, ontology, prompt revision, graph
settings, worker limit, Spark topology, deterministic seed, and graph identity;
only the LLM and serving platform differ.

## Timing

| Environment | Run 1 | Run 2 | Run 3 | Mean | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kubernetes, 1x GB10 | 958.972s | 1191.309s | 1101.471s | 1083.917s | 958.972-1191.309s |
| Kubernetes, 4x GB10 | 445.583s | 521.058s | 461.924s | 476.188s | 445.583-521.058s |
| GPT-5.6 Sol, Azure OpenAI | 280.312s | 191.684s | 248.523s | 240.173s | 191.684-280.312s |

| Stage | Qwen 1x GB10 | Qwen 4x GB10 | GPT-5.6 Sol | Qwen 4x Speedup | Four-Node Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Submission through preparation | 1.020s | 10.756s | 10.814s | 0.09x | 2.4% |
| Context, entity, relation, and compaction tasks | 946.478s | 369.220s | 129.392s | 2.56x | 64.1% |
| Spark graph finalization and enrichment | 136.419s | 96.212s | 99.966s | 1.42x | 35.5% |
| End to end | 1083.917s | 476.188s | 240.173s | 2.28x | 56.9% |

Preparation and extraction tasks are dynamically claimed from PostgreSQL, so
workers take the next eligible document or window instead of receiving static
partitions. Fleet scaling reduces the model-backed extraction stage most. The
remaining gap from linear scaling comes from cold autoscaling, unequal window
durations, a finite 24-window extraction workload, and graph-wide Spark joins,
shuffles, resolution, community detection, descriptions, reports, and publication
barriers. This ten-document corpus measures quality and end-to-end integration;
larger corpora are required to characterize steady-state fleet throughput.

## Quality

Values are means across three repetitions. Entity precision uses the selected
reference entity set, while relation precision uses the subgraph induced by
matched reference entities. Additional grounded output is reported as unscored.

| Metric | Qwen 1x GB10 | Qwen 4x GB10 | GPT-5.6 Sol |
| --- | ---: | ---: | ---: |
| Entity precision | 1.0000 | 1.0000 | 1.0000 |
| Entity recall | 0.9820 | 0.9414 | 0.9910 |
| Entity F1 | 0.9909 | 0.9693 | 0.9955 |
| Triple precision | 0.9539 | 0.9347 | 0.9817 |
| Triple recall | 0.9011 | 0.8828 | 0.9378 |
| Triple F1 | 0.9263 | 0.9065 | 0.9592 |
| Evidence support | 0.8974 | 0.8791 | 0.9194 |
| Information retention | 0.9416 | 0.9121 | 0.9644 |
| Components | 1.6667 | 2.0000 | 3.0000 |
| Isolate ratio | 0.0045 | 0.0093 | 0.0272 |
| Two-hop recoverability | 0.9121 | 0.8974 | 0.9524 |

## Stability

| Environment | Mean Node Jaccard | Minimum Node Jaccard | Mean Triple Jaccard | Minimum Triple Jaccard |
| --- | ---: | ---: | ---: | ---: |
| Kubernetes, 1x GB10 | 0.6006 | 0.5731 | 0.5372 | 0.5129 |
| Kubernetes, 4x GB10 | 0.5261 | 0.5204 | 0.4824 | 0.4541 |
| GPT-5.6 Sol, Azure OpenAI | 0.7794 | 0.7479 | 0.6556 | 0.5938 |

The deterministic seed governs graph algorithms, but model generation remains
stochastic. Different fleet widths also change request ordering and batching, so
quality and graph signatures are compared across repetitions instead of assumed
to be identical.

## Output Size

| Environment | Documents | Chunks | Entities | Relations | Evidence Records | Communities |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kubernetes, 1x GB10 | 10 | 32 | 341.000 | 174.000 | 680.667 | 18.667 |
| Kubernetes, 4x GB10 | 10 | 32 | 352.000 | 164.333 | 682.000 | 24.333 |
| GPT-5.6 Sol, Azure OpenAI | 10 | 32 | 312.667 | 178.000 | 650.000 | 20.667 |

## Reproduction

Deploy the pinned fleet profile, wait for all requested model replicas to become
ready, and submit the complete manifest three times with caches disabled. Fleet
deployment, monitoring, export, and retry commands are documented in
[`docs/kubernetes-fleet.md`](../../docs/kubernetes-fleet.md). Compact public
measurements are stored in [`results/`](results/).
