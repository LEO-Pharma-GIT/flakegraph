# Keep the container base aligned with pyproject/mypy/ruff and the local validation
# runtime. Every base below is pinned by digest (the multi-architecture index,
# so one pin serves amd64 and arm64) because a tag is a moving target and a
# build should be reproducible from the Dockerfile alone.
#
# kubectl supports one minor either side of the API server, no further, so
# the console's kubectl has to sit within one minor of the cluster it will
# read. Override with --build-arg KG_KUBECTL_VERSION=v1.xx.y to match yours;
# the digest pin below is for the default version only, so an override drops
# it and pins the tag.
ARG KG_KUBECTL_VERSION=v1.36.5
FROM registry.k8s.io/kubectl:${KG_KUBECTL_VERSION} AS kubectl

# The console is a Next.js application built once here and served by the Node
# runtime copied below. Building it in its own stage keeps its toolchain out of
# the image everything else runs; the standalone output it produces carries the
# server and the exact node_modules it traced, nothing more.
FROM oven/bun:1.3-slim@sha256:d56a2534ffd262e92c12fd3249d3924d296d97086da773f821d7d0477435ea04 AS console-build
WORKDIR /console
COPY react/package.json react/bun.lock ./
# A build host whose egress is TLS-intercepted hands its trust bundle in as a
# build secret (--secret id=ca-bundle,src=...); it is read for this one
# install and never lands in a layer. Without one, the registry is trusted as
# shipped.
RUN --mount=type=secret,id=ca-bundle \
    if [ -f /run/secrets/ca-bundle ]; then \
        bun install --frozen-lockfile --cafile /run/secrets/ca-bundle; \
    else \
        bun install --frozen-lockfile; \
    fi
COPY react/ ./
RUN bun run build

FROM node:22-bookworm-slim@sha256:de170b39439eb0e5a3444ad7d60b015614a381e10fecae2e30ec9e735122f38d AS node

FROM python:3.14.9-slim-trixie@sha256:b347e2dcbc7f0b6f9fb7182e9da6a187fb10ae715e0692518c1aed0eec94a700

ARG KG_INSTALL_MINERU=true
ARG KG_INSTALL_TESSERACT=false
ARG KG_INSTALL_LOCAL_EMBEDDINGS=true
ARG KG_INSTALL_GLINER=false
# Leiden community detection (`graph.community_algorithm: leiden`) needs the
# GPL-licensed igraph and leidenalg packages, so the public image leaves them
# out; a deployment that wants Leiden builds with this on and accepts that
# licence for its own image.
ARG KG_INSTALL_LEIDEN=false
ARG KG_PRELOAD_LOCAL_EMBEDDING=true
ARG KG_LOCAL_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
ARG KG_LOCAL_EMBEDDING_REVISION=97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
# Bounded, because the failure this guards against does not announce itself. The
# accelerated Hub transfer client waits on an interception proxy instead of
# failing, so without a limit the build stops making progress and reports
# nothing - for as long as anyone is willing to wait.
ARG KG_PRELOAD_TIMEOUT_SECONDS=600
ARG UV_VERSION=0.13.5

# HF_HUB_DISABLE_TELEMETRY: the Hub client tags every download with an
# identifying user agent unless told not to; nothing this image downloads is
# anybody else's business. Set at build time so the preload below and every
# runtime download alike stay quiet.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    XDG_CACHE_HOME=/home/kgprocessor/.cache \
    HF_HOME=/home/kgprocessor/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/kgprocessor/.cache/sentence_transformers
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# The worker should behave the same in Docker and SPCS: application code is
# immutable, while output and provider model caches live in writable paths owned
# by the unprivileged runtime user.
RUN useradd --create-home --shell /usr/sbin/nologin kgprocessor \
    && mkdir -p \
        /home/kgprocessor/.cache/huggingface \
        /home/kgprocessor/.cache/mineru \
        /home/kgprocessor/.cache/sentence_transformers \
        /home/kgprocessor/.cache/torch \
        /app/out \
    && chown -R kgprocessor:kgprocessor /home/kgprocessor /app/out

# Dependency resolution depends only on package metadata and the lockfile. Keep
# frequently edited documentation out of this layer so README changes do not
# reinstall OCR, embedding, and provider dependencies.
COPY pyproject.toml uv.lock /app/

RUN if [ "$KG_INSTALL_TESSERACT" = "true" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            poppler-utils \
            tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# MinerU's pipeline backend imports six without declaring it in the published
# extra. Keep it in the isolated tool environment so scanned-PDF OCR works
# without leaking MinerU dependencies into the FlakeGraph environment.
RUN if [ "$KG_INSTALL_MINERU" = "true" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential \
            libgl1 \
            libglib2.0-0 \
            libgomp1; \
    fi \
    && python -m pip install "uv==$UV_VERSION" \
    && if [ "$KG_INSTALL_MINERU" = "true" ]; then \
        uv python install 3.13 \
        && uv tool install --python 3.13 --with "six==1.17.0" "mineru[pipeline]==3.4.4"; \
    fi \
    && set -- \
    && if [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
        set -- "$@" --extra local-embeddings; \
    fi \
    && if [ "$KG_INSTALL_GLINER" = "true" ]; then set -- "$@" --extra extract-gliner; fi \
    && if [ "$KG_INSTALL_LEIDEN" = "true" ]; then set -- "$@" --extra leiden; fi \
    && uv sync --locked --no-dev --no-editable --no-install-project "$@" \
    && if [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ] \
        && [ "$KG_PRELOAD_LOCAL_EMBEDDING" = "true" ]; then \
        timeout "$KG_PRELOAD_TIMEOUT_SECONDS" /opt/venv/bin/python -c \
          "from sentence_transformers import SentenceTransformer; SentenceTransformer('$KG_LOCAL_EMBEDDING_MODEL', revision='$KG_LOCAL_EMBEDDING_REVISION')" \
        || { \
            echo "ERROR: could not fetch $KG_LOCAL_EMBEDDING_MODEL within ${KG_PRELOAD_TIMEOUT_SECONDS}s." >&2; \
            echo "The Hub serves weights from a separate CDN host. A network that" >&2; \
            echo "resolves that host to an interception proxy substitutes its own" >&2; \
            echo "certificate, which a current Python rejects - and the accelerated" >&2; \
            echo "transfer client stalls rather than reporting it. Trusting the" >&2; \
            echo "proxy's CA is not sufficient if its chain omits the Authority Key" >&2; \
            echo "Identifier, which strict X.509 verification requires." >&2; \
            echo "Reach the CDN host directly, or build with" >&2; \
            echo "--build-arg KG_PRELOAD_LOCAL_EMBEDDING=false and mount the model" >&2; \
            echo "into the deployment instead (docs/kubernetes-fleet.md, Scaling)." >&2; \
            exit 1; \
        } \
        # A download pinned to a revision records no branch, and offline
        # resolution of the bare model id looks the branch up. Recording it
        # is what lets a deployment name the model without a revision.
        && model_cache="$SENTENCE_TRANSFORMERS_HOME/models--$(echo "$KG_LOCAL_EMBEDDING_MODEL" | sed 's#/#--#g')" \
        && mkdir -p "$model_cache/refs" \
        && printf '%s' "$KG_LOCAL_EMBEDDING_REVISION" > "$model_cache/refs/main"; \
    fi \
    && uv cache clean \
    && chown -R kgprocessor:kgprocessor /home/kgprocessor/.cache \
    && if [ "$KG_INSTALL_MINERU" = "true" ]; then \
        apt-get purge -y --auto-remove build-essential \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# The CLI reads the cluster through kubectl rather than a client library.
# Taken from the upstream release image rather than fetched: this base carries
# no curl, and that image is multi-arch, so the copy is right on arm64 without
# asking which architecture is being built - about 50 MB.
COPY --from=kubectl /bin/kubectl /usr/local/bin/kubectl

# --chown, and the chmod below, because COPY preserves the *build host's* file
# modes. A source tree synced onto a build machine under a restrictive umask
# produced a 0640 root-owned application file, which the container - running as
# an unprivileged user - could not read, and the image failed at runtime with a
# permission error rather than at build. What the image contains should not
# depend on the umask of whoever last copied the tree onto the builder.
COPY --chown=kgprocessor:kgprocessor README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /app/
COPY --chown=kgprocessor:kgprocessor src /app/src

# Installed environments are self-contained. Removing uv's wheel and source
# cache avoids shipping several gigabytes of duplicate build inputs to every
# worker without changing either the application or MinerU environments.
RUN set -- \
    && if [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
        set -- "$@" --extra local-embeddings; \
    fi \
    && if [ "$KG_INSTALL_GLINER" = "true" ]; then set -- "$@" --extra extract-gliner; fi \
    && if [ "$KG_INSTALL_LEIDEN" = "true" ]; then set -- "$@" --extra leiden; fi \
    && uv sync --locked --no-dev --no-editable "$@" \
    && uv cache clean

# Config examples are runtime inputs for local, on-prem, and SPCS execution.
# Test data and docs stay outside the image and are mounted or staged instead,
# except the sample packs the console offers: the small martial-arts corpus
# for a first run and the atopic dermatitis papers (CC BY, 1.5 MB of Markdown)
# for a realistic one. Their gold sets and recorded results are what the
# Quality tab scores a graph against and compares it with, so a fleet with
# nothing else to hand can still prove the pipeline. The deep-learning papers
# corpus is downloaded on demand and is not shipped.
COPY --chown=kgprocessor:kgprocessor configs /app/configs
COPY --chown=kgprocessor:kgprocessor data/martial_arts/files /app/data/martial_arts/files
COPY --chown=kgprocessor:kgprocessor data/martial_arts/gold.json data/martial_arts/ontology.yaml data/martial_arts/manifest.jsonl data/martial_arts/LICENSE.md /app/data/martial_arts/
COPY --chown=kgprocessor:kgprocessor data/martial_arts/results /app/data/martial_arts/results
COPY --chown=kgprocessor:kgprocessor data/atopic_dermatitis /app/data/atopic_dermatitis

# The console: the Node runtime and the standalone server the build stage
# produced. Node is one binary; the copies are multi-arch, so they are right on
# arm64 without asking. The control plane is a component of the deployment, not
# a script an operator runs on a node by hand, so the image serves it like
# anything else - about 200 MB.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=console-build --chown=kgprocessor:kgprocessor /console/.next/standalone /app/react
COPY --from=console-build --chown=kgprocessor:kgprocessor /console/.next/static /app/react/.next/static
COPY --from=console-build --chown=kgprocessor:kgprocessor /console/public /app/react/public

# Normalise read bits for the same reason: ownership alone still leaves a 0600
# file unreadable to anything but its owner, and these are read-only inputs.
RUN chmod -R a+rX /app/README.md /app/LICENSE /app/NOTICE /app/THIRD_PARTY_NOTICES.md /app/src /app/configs /app/react

USER kgprocessor

ENTRYPOINT ["flakegraph"]
CMD ["--help"]
