# Third-Party Notices

This file lists the third-party software, container images and model weights
that FlakeGraph depends on, downloads, or references, with their licences and
the obligations that follow when a package or image is redistributed. It is
informational and is not a substitute for legal review.

FlakeGraph source code is licensed under the Apache License 2.0 (see `LICENSE`
and `NOTICE`). That licence applies to this repository's application code, the
console, configuration examples, prompt templates, documentation and tests.
The original benchmark dataset under `data/martial_arts/` is separately
dedicated under CC0-1.0, and the annotations under `data/deep_learning_papers/`
carry the terms in that directory's `LICENSE.md`. Neither dedication
re-licenses third-party packages, provider services, downloaded model weights
or operating-system packages installed into a container image.

## How This File Is Maintained

The Python tables were generated from `uv.lock` with `pip-licenses` over the
locked environment (core dependencies, then every optional extra); the console
table from `react/package.json` and `react/bun.lock`. A contract test
(`tests/unit/test_documentation_contracts.py`) checks that every direct
dependency declared in `pyproject.toml` appears here, so adding a dependency
without a notice fails the suite. Regenerate the inventory with:

```bash
uv run --isolated --no-dev --all-extras --with pip-licenses pip-licenses --format=markdown --with-urls
```

The source repository vendors no Python packages, JavaScript packages or model
weights. Dependencies are resolved from `pyproject.toml`/`uv.lock` and
`react/package.json`/`react/bun.lock`; container builds install them at build
time. If you redistribute a built image or packaged artifact, include the
licence and notice files required by the installed packages, model artifacts
and system packages in that artifact.

## Copyleft Summary

FlakeGraph's required dependencies are all under permissive licences (MIT,
BSD, Apache-2.0, ISC, PSF, MPL-2.0 file-level) with two weak-copyleft
exceptions and one opt-in strong-copyleft extra:

- `psycopg`, `psycopg-binary` and `psycopg-pool` are LGPL-3.0. They are used
  as an ordinary library through their public API and are not modified;
  redistributed images must keep the LGPL text and allow the library to be
  replaced. Source: https://www.psycopg.org/psycopg3/docs/basic/license.html
- `certifi` is MPL-2.0, which applies file by file to the bundle it ships and
  imposes nothing on FlakeGraph's own code.
- **Leiden community detection is an opt-in extra and is GPL.** `igraph`
  (GPL-2.0-or-later) and `leidenalg` (GPL-3.0-or-later) are installed only by
  `flakegraph[leiden]` and used only when `graph.community_algorithm: leiden`
  is configured. The default partition, Louvain, comes from NetworkX
  (BSD-3-Clause). Installing the extra makes the combined installation a work
  that must be distributed under the GPL's terms; do not build a
  redistributable image with it unless that is acceptable. Sources:
  https://igraph.org/c/html/latest/igraph-License.html and
  https://github.com/vtraag/leidenalg/blob/main/LICENSE

MinerU's own licence and the AGPL model weights it downloads are covered under
"Models" below; they matter to anyone offering MinerU-backed OCR as a service.

## Python Package: Required Dependencies

Direct dependencies declared in `pyproject.toml`, as resolved by `uv.lock`.

| Package | Licence | Role |
| --- | --- | --- |
| `azure-identity` | MIT | Azure credential chain for Blob and Azure OpenAI |
| `azure-storage-blob` | MIT | Azure Blob Storage file source |
| `boto3` (with `botocore`, `s3transfer`) | Apache-2.0 | S3-compatible file source and artifact store |
| `defusedxml` | PSF-2.0 | Safe XML parsing of Office documents |
| `fastapi` (with `starlette`) | MIT / BSD-3-Clause | Inference sidecar and OCR shim HTTP services |
| `httpx` (with `httpcore`, `h11`, `anyio`) | BSD-3-Clause / MIT | Provider HTTP client |
| `networkx` | BSD-3-Clause | Graph structure, Louvain community detection, quality metrics |
| `numpy` | BSD-3-Clause (with 0BSD, MIT, Zlib, CC0-1.0 components) | Vector arithmetic |
| `pandas` | BSD-3-Clause | Tabular artifacts and exports |
| `plotly` | MIT | Interactive graph explorer; the self-contained HTML files embed Plotly.js so they open without a network connection |
| `prometheus-client` | Apache-2.0 (with BSD-2-Clause components) | Metrics exposed by the sidecar and shim |
| `psycopg[binary,pool]` | LGPL-3.0 | PostgreSQL task coordination (see Copyleft Summary) |
| `pyarrow` | Apache-2.0 | Parquet artifacts |
| `pydantic` (with `pydantic-core`) | MIT | Settings and domain models |
| `pypdf` | BSD-3-Clause | Built-in PDF text extraction |
| `pypdfium2` | BSD-3-Clause and Apache-2.0; bundles Google PDFium (BSD-3-Clause) with its third-party notices | Built-in PDF text and rendering fallback; preserve the `PDFIUM_THIRD_PARTY` notices when redistributing |
| `python-multipart` | Apache-2.0 | Multipart uploads to the OCR shim |
| `pyyaml` | MIT | Configuration profiles |
| `rich` | MIT | Terminal progress |
| `snowflake-connector-python` | Apache-2.0 | Snowflake connectivity; Cortex and Snowpark Container Services usage is governed by the target account's terms |
| `typer` (with `click`, `shellingham`) | MIT / BSD-3-Clause / ISC | Command-line interface |
| `uvicorn` | BSD-3-Clause | ASGI server for the sidecar and shim |

Transitive packages in the locked core environment and their licences:
`annotated-doc` (MIT), `annotated-types` (MIT), `asn1crypto` (MIT),
`azure-core` (MIT), `certifi` (MPL-2.0), `cffi` (MIT-0),
`charset-normalizer` (MIT), `cryptography` (Apache-2.0 OR BSD-3-Clause),
`filelock` (MIT), `idna` (BSD-3-Clause), `isodate` (BSD-3-Clause),
`jmespath` (MIT), `markdown-it-py` (MIT), `mdurl` (MIT), `msal` (MIT),
`msal-extensions` (MIT), `narwhals` (MIT), `packaging` (Apache-2.0 OR
BSD-2-Clause), `platformdirs` (MIT), `pycparser` (BSD-3-Clause), `Pygments`
(BSD-2-Clause), `PyJWT` (MIT), `pyOpenSSL` (Apache-2.0), `python-dateutil`
(Apache-2.0 / BSD-3-Clause), `pytz` (MIT), `requests` (Apache-2.0), `six`
(MIT), `sortedcontainers` (Apache-2.0), `tomlkit` (MIT), `typing-extensions`
(PSF-2.0), `typing-inspection` (MIT), `tzdata` (Apache-2.0), `urllib3` (MIT).

## Python Package: Optional Extras

| Extra | Package | Licence | Notes |
| --- | --- | --- | --- |
| `local-embeddings` | `sentence-transformers` | Apache-2.0 | Local embedding provider |
| `local-embeddings` | `torch` | BSD-3-Clause (with Apache-2.0, LLVM-exception and BSD-2-Clause components) | Linux builds resolve PyTorch's CPU wheel index; review PyTorch's notices when redistributing images |
| `local-embeddings`, `extract-gliner` | `transformers` | Apache-2.0 | Model loading; brings `huggingface-hub`, `tokenizers`, `safetensors`, `hf-xet`, `regex` (all Apache-2.0), `tqdm` (MPL-2.0 and MIT), `sentencepiece` (Apache-2.0) |
| `extract-gliner` | `gliner` | Apache-2.0 | Optional local entity extraction; brings `scikit-learn`, `scipy`, `joblib`, `threadpoolctl` (BSD-3-Clause) |
| `distributed-spark` | `pyspark` | Apache-2.0 | Partitioned finalization; brings `py4j` (BSD-3-Clause) |
| `distributed-spark` | `graphframes-py` | Apache-2.0 | Label-propagation communities on Spark |
| `leiden` | `igraph` | GPL-2.0-or-later | Opt-in Leiden partition (see Copyleft Summary); brings `texttable` (MIT) |
| `leiden` | `leidenalg` | GPL-3.0-or-later | Opt-in Leiden partition (see Copyleft Summary) |
| `sso-cache` | `snowflake-connector-python[secure-local-storage]` | Apache-2.0 | Brings `keyring` (MIT) and the `jaraco.*` helpers (MIT), `more-itertools` (MIT) |

Development-only tools (`ruff`, `mypy`, `pytest`, `pytest-cov`, `pandas-stubs`,
`boto3-stubs`, `types-*`, `python-docx`) are not installed into images or the
wheel and are not listed.

GLiREL is not bundled because its published code/model licence includes
non-commercial and share-alike restrictions; it can only be connected as an
externally reviewed custom relation adapter. Source:
https://github.com/jackboyla/GLiREL

## Console (`react/`)

The console is a Next.js application. Its direct runtime dependencies, from
`react/package.json` and `react/bun.lock`:

| Package | Licence |
| --- | --- |
| `next`, `react`, `react-dom` | MIT |
| `@ai-sdk/azure`, `@ai-sdk/openai`, `@ai-sdk/provider`, `@ai-sdk/react`, `ai` | Apache-2.0 |
| `@effect/platform`, `@effect/platform-node`, `effect` | MIT |
| `@hookform/resolvers`, `react-hook-form` | MIT |
| `@kubernetes/client-node` | Apache-2.0 |
| `@tanstack/react-query` | MIT |
| `@trpc/client`, `@trpc/react-query`, `@trpc/server` | MIT |
| `class-variance-authority` | Apache-2.0 |
| `client-only`, `server-only` | MIT |
| `clsx`, `cmdk`, `date-fns`, `nuqs`, `sonner`, `superjson`, `tailwind-merge`, `tw-animate-css` | MIT |
| `drizzle-orm` | Apache-2.0 |
| `geist` | MIT (package); the Geist typefaces it bundles are SIL Open Font License 1.1 |
| `hyparquet` | MIT |
| `lucide-react` | ISC |
| `postgres` | Unlicense |
| `radix-ui` | MIT |
| `react-markdown`, `remark-gfm` | MIT |
| `snowflake-sdk` | Apache-2.0 |
| `yaml` | ISC |
| `zod` | MIT |

Development dependencies (`@playwright/test` and `typescript`, Apache-2.0;
`@tailwindcss/postcss`, `tailwindcss`, `@types/*`, `drizzle-kit`, `eslint`,
`eslint-config-next`, `jsdom`, `vite-tsconfig-paths`, `vitest`, MIT) are used
to build and test the console and are not shipped in the production bundle,
except for the CSS that Tailwind generates. The transitive tree is recorded in
`react/bun.lock`; `bunx license-checker-rseidelsohn --production` in `react/`
lists it in full.

## Models

FlakeGraph downloads no model weights into the repository. The models below are
the ones its defaults, profiles, images or chart name; each is fetched from its
publisher at install or run time, under the publisher's terms. Replace any of
them in configuration and review the replacement's model card.

| Model | Licence | Where it is used |
| --- | --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` | Apache-2.0 | Default embedding model (`embedding.model`); the production image preloads a pinned revision so executors never download mutable weights mid-run. https://huggingface.co/Qwen/Qwen3-Embedding-0.6B |
| `qwen3:8b` (Ollama build of `Qwen/Qwen3-8B`) | Apache-2.0 | Default LLM in `configs/app-defaults.yaml`, served by Ollama (Ollama itself is MIT and obtained separately). https://huggingface.co/Qwen/Qwen3-8B |
| `Qwen/Qwen3-8B` | Apache-2.0 | The GPU profile `configs/local-vllm-mineru-oss.yaml`, served by vLLM |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | Embedding model of the benchmark dataset profiles under `data/*/configs/` |
| `urchade/gliner_multi-v2.1` | Apache-2.0 | Default GLiNER model for the optional `extract-gliner` entity extractor. https://huggingface.co/urchade/gliner_multi-v2.1 |
| MinerU pipeline models | AGPL-3.0 | The `mineru_internal` and `fallback` OCR providers run the MinerU CLI, which on first use downloads its pipeline models itself, at MinerU 3.4.4 from `opendatalab/PDF-Extract-Kit-1.0` (Hugging Face, or `OpenDataLab/PDF-Extract-Kit-1.0` on ModelScope when `MINERU_MODEL_SOURCE=modelscope`). Those layout, formula, table and OCR weights are AGPL-3.0: offering them as a network service is what triggers the AGPL's source obligations. https://github.com/opendatalab/PDF-Extract-Kit |
| `unsloth/Qwen3.8-27B-NVFP4` | Apache-2.0 (quantisation of Alibaba's `Qwen/Qwen3.8-27B`) | The Helm chart's serving profile and the DGX Spark reference example under `deploy/examples`; NVFP4 needs a Blackwell-class GPU and is not used by any laptop or GPU profile in `configs/`. https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4 |
| DFlash2 draft model | Reference deployment only; no public source | The chart's speculative-decoding drafter (`speculativeMethod: dflash`) reads a draft checkpoint from an operator-supplied image or path. No public release of that checkpoint exists; the example values name a private registry placeholder, and operators must supply and license their own drafter or select `mtp`, which needs no download. |

MinerU itself is licensed under the MinerU Open Source License, which is based
on Apache-2.0 with additional terms: an attribution obligation for online
services and commercial thresholds. It is installed by the quick start and the
default image as an isolated Python 3.13 tool (`mineru[pipeline]`) rather than
into FlakeGraph's own environment, and FlakeGraph strips provider and cloud
credentials from the subprocess environment it runs in as defence in depth.
Review the licence before offering MinerU-backed services. Source:
https://github.com/opendatalab/MinerU/blob/master/LICENSE.md

## Container Images

The production image (`Dockerfile`) is built from `python:3.14-slim` (PSF
licence for Python, Debian packages under their respective licences), copies
`kubectl` from `registry.k8s.io/kubectl` (Apache-2.0) and builds the console
with `oven/bun` (MIT) and `node` (MIT). It installs the Python dependencies
above, optionally MinerU, and optionally system packages:

- `tesseract-ocr` is installed only when `KG_INSTALL_TESSERACT=true`; Tesseract
  is Apache-2.0. Source: https://tesseract-ocr.github.io/tessdoc/
- `poppler-utils` is installed only when `KG_INSTALL_TESSERACT=true`; Debian's
  poppler packages include GPL/LGPL/MIT licensed components. Review this
  optional image variant before redistribution.

The Spark image (`Dockerfile.spark`) copies runtime components from the
pinned `apache/spark` image and resolves Apache Spark, Hadoop AWS and
GraphFrames JVM packages (all Apache-2.0) during the build. Preserve their
licence and notice files when redistributing that image.

## Deployment: Images And Charts The Helm Chart References

The chart under `deploy/helm/flakegraph` references these published images by
tag or digest and does not redistribute them:

- vLLM (`vllm/vllm-openai`) serves the chart-managed model. vLLM is Apache-2.0;
  the image also contains NVIDIA CUDA and other runtime components governed by
  their own notices. Earlier profiles used NVIDIA's vLLM container from NGC,
  whose NGC terms apply if you select it instead. Source:
  https://github.com/vllm-project/vllm/blob/main/LICENSE
- LiteLLM (`ghcr.io/berriai/litellm`) provides the API gateway: virtual keys,
  budgets, spend records and model access control. The proxy is MIT with
  additional terms covering its enterprise features; review those before
  enabling enterprise functionality. Source:
  https://github.com/BerriAI/litellm/blob/main/LICENSE
- Envoy (`docker.io/envoyproxy/envoy`) provides the inference listener and is
  Apache-2.0. Source: https://github.com/envoyproxy/envoy/blob/main/LICENSE
- The endpoint picker is llm-d's inference scheduler
  (`ghcr.io/llm-d/llm-d-router-endpoint-picker`), Apache-2.0 and derived from
  the Kubernetes Gateway API Inference Extension. The chart adapts its
  upstream Envoy and plugin configuration. Sources:
  https://github.com/llm-d/llm-d-router and
  https://github.com/kubernetes-sigs/gateway-api-inference-extension
- oauth2-proxy (`quay.io/oauth2-proxy/oauth2-proxy`) is the sign-in gate in
  front of the console and dashboards and is MIT. Source:
  https://github.com/oauth2-proxy/oauth2-proxy/blob/master/LICENSE
- KEDA provides queue-driven autoscaling and is installed separately from its
  upstream Helm chart; it is Apache-2.0 and not embedded in the package or
  images. Source: https://github.com/kedacore/keda
- CloudNativePG (PostgreSQL operator, Apache-2.0), MinIO (S3-compatible object
  storage, AGPL-3.0 for the server; the chart only talks to it over S3 and does
  not install or embed it), kube-prometheus-stack and Grafana (Apache-2.0 and
  AGPL-3.0 respectively) and the NVIDIA Kubernetes device plugin (Apache-2.0)
  are optional infrastructure referenced by the fleet guide. They are not
  installed by the FlakeGraph chart; review and pin their upstream licences,
  images and charts before operating or redistributing a complete platform
  bundle.

## Provider Services

FlakeGraph can call OpenAI-compatible APIs, Azure OpenAI, Ollama, vLLM
endpoints, Snowflake Cortex, generic HTTP OCR services, MinerU API services and
local models. The configured provider or model owner controls the applicable
terms, data-usage policy, allowed outputs, region availability and commercial
permissions.

Before production use, approve every configured provider endpoint and model
identifier for licensing, data residency, security and acceptable-use posture.
