# Atopic Dermatitis Benchmark

All rows are scored against `gold.json` 1.1.0, which is exhaustive within the
ontology, so precision and F1 are measured as well as recall.

## Results

| Result | Environment | Runs | Entity P / R / F1 | Triple P / R / F1 | Evidence |
| --- | --- | ---: | --- | --- | ---: |
| [`adt-v1.1-gpt6-astra-azure-openai-local-3x-20260925`](results/gpt6-astra-azure-openai-local-3x-20260925.json) | Local, Azure OpenAI | 3 (mean) | 0.684 / 0.798 / 0.737 | 0.655 / 0.566 / 0.607 | 0.53 |
| [`adt-v1.1-qwen38-27b-nvfp4-vllm-k8s-6x-gb10-20260924`](results/qwen38-27b-nvfp4-vllm-kubernetes-6x-gb10-20260924.json) | Kubernetes, 6x GB10, from the console's sample pack | 1 | 0.678 / 0.539 / 0.600 | 0.600 / 0.233 / 0.336 | 0.14 |

Neither meets every acceptance gate. What the three astra runs found went
into the gold when the text supports it, which favours their recall somewhat;
the Qwen run contributed nothing to the gold.

## Reading the rows

The two models extract entities with similar precision; GPT-6 astra finds
more of them. The gap is in relations: astra recovers 0.566 of the gold's
relations, Qwen 3.8 0.233, and only 0.14 of Qwen's required relations carry an
evidence quote the evaluator can match. Most relations Qwen misses have both
ends in its graph with no edge between them.

Repeated runs of the same model differ more on relations than on entities.
Across the three astra runs the mean pairwise overlap is 0.763 for entity
names and 0.479 for triples: a single run finds under half of the relations
another run finds. Compare models with repeated runs, and read a single-run
difference in relation scores of a few points as noise.
