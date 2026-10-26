# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=vllm/vllm-openai:v0.21.0
FROM rust:1.95.0-slim-bookworm AS rust-toolchain

FROM ${BASE_IMAGE}

# Reuse the CUDA/Torch/vLLM combination maintained by the upstream GPU image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mineru-src
COPY pyproject.toml README.md LICENSE.md ./
COPY mineru ./mineru

# Mount Rust only during compilation; do not ship its toolchain in the final image.
# The explicit Git requirement also works with pip, which ignores tool.uv.sources.
RUN --mount=from=rust-toolchain,source=/usr/local,target=/opt/rust,ro \
    export PATH="/opt/rust/cargo/bin:$PATH" \
           RUSTUP_HOME=/opt/rust/rustup RUSTUP_TOOLCHAIN=1.95.0 DOCVORTEX_BUILD_NATIVE=1 \
    && python3 -m pip freeze | grep -E '^(torch|torchvision|torchaudio|vllm)==' > /tmp/base-constraints.txt \
    && DOCVORTEX_REQUIREMENT="$(python3 -c 'import tomllib; s=tomllib.load(open("pyproject.toml", "rb"))["tool"]["uv"]["sources"]["docvortex"]; print("docvortex @ git+" + s["git"] + "@" + s["rev"])')" \
    && VLLM_REQUIREMENT="$(python3 -c 'import importlib.metadata; print("vllm==" + importlib.metadata.version("vllm"))')" \
    && python3 -m pip install --no-cache-dir --constraint /tmp/base-constraints.txt \
         '.[torch]' "$DOCVORTEX_REQUIREMENT" "$VLLM_REQUIREMENT" \
    && python3 -c 'import importlib.metadata as m, json, tomllib; s=tomllib.load(open("pyproject.toml", "rb"))["tool"]["uv"]["sources"]["docvortex"]; d=json.loads(m.distribution("docvortex").read_text("direct_url.json")); assert d["url"] == s["git"] and d["vcs_info"]["commit_id"] == s["rev"]' \
    && python3 -c 'from docvortex._compute_backend import get_native; assert get_native() is not None; from mineru_llama_cpp import Engine' \
    && rm -rf /root/.cargo /root/.cache /tmp/base-constraints.txt

COPY docker/rest-api/entrypoint.py /opt/mineru-rest/entrypoint.py

ENV PYTHONUNBUFFERED=1 \
    MINERU_HOME=/data/mineru \
    HF_HOME=/data/huggingface \
    XDG_CACHE_HOME=/data/cache \
    MINERU_MODEL_SOURCE=huggingface \
    MINERU_DEVICE_MODE=cuda \
    MINERU_SERVER_TIER=standard \
    MINERU_SERVER_CONCURRENCY=1 \
    MINERU_SERVER_IMAGE_ANALYSIS=0 \
    MINERU_API_RETENTION_SECONDS=86400

LABEL org.opencontainers.image.source="https://github.com/phipi-a/MinerU"
EXPOSE 8000
VOLUME ["/data"]
WORKDIR /data

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=5 \
    CMD python3 -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/v1/health", timeout=4).close()'

ENTRYPOINT ["python3", "/opt/mineru-rest/entrypoint.py"]
