# Atopic Dermatitis Treatment

A benchmark corpus of 50 open-access research articles and reviews on the
treatment of atopic dermatitis: biologics, JAK inhibitors and topical agents,
their molecular targets, the trials that tested them, and the conditions
around the disease. Every article is published under CC BY (or dedicated under
CC0), so it is redistributed here, converted to Markdown, with attribution in
[LICENSE.md](LICENSE.md).

## Contents

| Path | Purpose |
| --- | --- |
| `files/` | 50 articles as Markdown: title, abstract, body sections, figure captions and tables. |
| `papers.json` | The selection: PMCID, DOI, title, authors, journal, year, license, file. |
| `build.py` | Rebuilds `files/`, `manifest.jsonl` and `LICENSE.md` from `papers.json`; `--select` chooses afresh. |
| `manifest.jsonl` | Relative file paths, source URIs, SHA-256 checksums, byte sizes, and MIME types. |
| `ontology.yaml` | Drugs, drug classes, diseases, molecules, cells, microorganisms, studies, outcome measures, organizations and locations, and 15 relations between them. |
| `gold.json` | The reference graph the Quality tab scores against. |
| `configs/kubernetes-azure-openai.yaml` | The distributed profile for a reference extraction with an Azure OpenAI deployment. |
| `results/` | Recorded quality measurements on this gold. |
| `LICENSE.md` | Per-article attribution and licenses; CC0 for the annotations. |

## How the corpus was chosen

`build.py --select` queries Europe PMC for articles with "atopic dermatitis" or
"atopic eczema" in the title and a named treatment in the abstract, published
2019-2026, open access, and licensed CC BY or CC0. Case reports, letters,
protocols, errata and commentaries are excluded. The rest are ranked by
citations, with at most four per journal, and kept when their converted text is
between 1,500 and 7,000 words. The committed `papers.json` fixes the result.

Conversion keeps what states treatment facts and drops what adds citation noise:
reference lists, citation markers, author lists, acknowledgements, funding,
competing-interest, ethics and data-availability sections.

## How the gold is made

The gold is exhaustive within the ontology: every entity and relation of the
listed types that the articles state, not a selection, so the evaluator scores
precision and F1 as well as recall. It holds 3,462 entities, 5,762 relations
and 7,857 evidence observations, and is built from several extractions of the
corpus, each claim judged against the source text:

1. GPT-6 astra extracted the corpus four times and Qwen 3.8 once, from the
   console's sample pack.
2. Every entity and relation on which astra's first extraction and Qwen's
   disagreed was judged by a model that extracted neither.
3. The entities and relations of all four astra extractions were
   deduplicated (the same typed name or alias is one entity) and each was
   validated against the text on its own. An item already judged in step 2
   keeps that judgement; one never judged is kept or removed on this
   validation; a new one is added when the text states it.
4. A name an extraction used for an entity that the gold lists under another
   name is kept as an alias.

What no extraction found is not in the gold; each further extraction judged
the same way adds to it.
