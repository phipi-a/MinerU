# Fork REST API: NVIDIA GPU and CPU

This deployment runs only the REST API, not Gradio or a separate VLM HTTP server.
GPU mode uses CUDA/Torch for small models and in-process vLLM for the VLM. CPU
mode uses ONNX small models and in-process llama.cpp. Both modes use the same
image and persistent data volume; switching modes requires recreating the
container, not rebuilding the image.

Build from the `phipi-a/MinerU` checkout containing Markdown mapping and the
visibility fix. The Dockerfile installs this checkout and the exact DocVortex
Git commit declared in `pyproject.toml`, including its native Rust extension.
It verifies the installed DocVortex commit during the build. Agent files,
Git metadata, local environments, credentials and generated outputs are excluded
from the allowlisted build context.

## Requirements and validation status

- Docker Engine 28.3 or newer with BuildKit and a current Docker Compose plugin
  on Linux x86-64. GPU access uses CDI device selectors.
- For GPU mode: a compatible NVIDIA GPU/driver and NVIDIA Container Toolkit,
  configured to generate CDI devices for Docker. The default base is `vllm/vllm-openai:v0.21.0`.
  Your GPU and driver must support that image's CUDA/vLLM combination. Use
  `MINERU_BASE_IMAGE` to select a compatible upstream variant when necessary.
- CPU mode does not require NVIDIA hardware or the NVIDIA Container Toolkit.
  It still uses the large CUDA-capable image; it is not a minimal CPU image.
- Internet access during the build and initial model downloads. Plan for multiple
  gigabytes of model storage, substantial image/build disk space, and enough
  host RAM/VRAM for the selected tier. Standard CPU parsing can be very slow.

The launcher and configuration have automated tests. The image has also been
built successfully and tested on an AMD Ryzen 7 5800X with an NVIDIA RTX 3090
(24 GiB, driver 595.71.05). Both CPU and GPU REST modes parsed all 15 pages of
Attention Is All You Need at the standard tier, with cold and warm requests and
verified paired Markdown/map hashes. The warm GPU request took 9.31 seconds
versus 63.28 seconds for CPU on this host; these are individual measurements of
different backend/model-format combinations, not a general performance guarantee.
Validate a real parse on your own target hardware as well.

Run the following commands from the repository root.

## 1. Configure

```bash
cp docker/rest-api/.env.example docker/rest-api/.env
```

Edit `.env`. Set a strong `MINERU_API_KEY` before exposing the service. The default
host binding is loopback (`127.0.0.1:8000`). Changing `MINERU_BIND_ADDRESS` to
`0.0.0.0` exposes the API on all host interfaces; use authentication, a firewall
and a TLS reverse proxy. Local filesystem sources and plain-HTTP URL sources
remain disabled. Do not commit the `.env` file.

`MINERU_SERVER_TIER=standard` enables the standard server capability set, including
Advanced requests. For a lighter CPU deployment, set it to `basic`; this avoids
the VLM path but restricts the available tiers. See `/v1/tiers` for capabilities.
Requests without a tier use `standard`; pass `tier=advanced` per request for
VLM layout analysis.

`MINERU_SERVER_IMAGE_ANALYSIS=0` (default) disables VLM image analysis, which
only runs for Advanced requests. Images and charts then stay images, while
detected tables are still recognized as tables. With `1`, Advanced additionally
attaches estimated data tables to charts and converts images of tables or
formulas into table/equation blocks.

Models are downloaded lazily on the first request into the persistent volume.
`MINERU_SERVER_PRELOAD=1` initializes the selected model stack before startup
completes. Increase startup/health-check allowances if model downloads are slow.
The image contains neither model weights nor secrets. GPU and CPU modes use
different model bundles and may each require an initial download.

## 2. Build and start with NVIDIA GPU

```bash
docker compose --env-file docker/rest-api/.env -p mineru-rest \
  -f docker/rest-api/compose.yaml -f docker/rest-api/compose.gpu.yaml \
  up --build -d

docker compose --env-file docker/rest-api/.env -p mineru-rest \
  -f docker/rest-api/compose.yaml -f docker/rest-api/compose.gpu.yaml \
  logs -f api
```

`NVIDIA_GPU_ID=0` selects the CDI device `nvidia.com/gpu=0`. On NixOS, enable
`hardware.nvidia-container-toolkit.enable` alongside the host NVIDIA driver;
verify `nvidia-smi` and the `nvidia-container-toolkit-cdi-generator` service after
rebooting. This configuration does not use the deprecated Docker NVIDIA runtime
or legacy `--gpus` reservations. Inside the container, the selected GPU is
addressed as CUDA device 0. GPU mode refuses to start if Torch cannot access CUDA;
it does not silently fall back to CPU. All inference runs inside this container.

The upstream CUDA/Torch/vLLM stack is reused; its installed Torch-family versions
are constrained during installation rather than silently replaced by pip.
The Rust toolchain is mounted only while compiling DocVortex and is not included
in the resulting application layers. Other package dependencies are resolved
at build time; this is not a fully locked or bit-reproducible image.

## 3. Switch to CPU

Stop the GPU container without deleting its volume, then recreate using only the
base Compose file:

```bash
docker compose --env-file docker/rest-api/.env -p mineru-rest \
  -f docker/rest-api/compose.yaml -f docker/rest-api/compose.gpu.yaml down

docker compose --env-file docker/rest-api/.env -p mineru-rest \
  -f docker/rest-api/compose.yaml up -d
```

For the first CPU-only installation, use the second command with `--build`.
CPU mode hides CUDA devices and explicitly selects ONNX plus llama.cpp. It does
not request GPUs from Docker. Set `MINERU_CPU_NUM_THREADS` to limit Torch CPU
threads; llama.cpp and ONNX have their own threading behavior.

To switch back, stop the CPU deployment with the base file and repeat the GPU
startup command. Do not run both variants simultaneously with the same project
name or upload directory.

## 4. Check the API and parse a PDF

```bash
curl --fail http://127.0.0.1:8000/v1/health
curl --fail http://127.0.0.1:8000/v1/tiers

# Replace the key with the one configured in .env, or omit the header if unset.
curl --fail-with-body \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -F 'files=@report.pdf' \
  'http://127.0.0.1:8000/v1/file_parse?tier=basic' \
  -o result.json
```

A healthy HTTP endpoint does not prove that model downloads or inference have
completed successfully. Verify a real parse in both modes you intend to use.
The synchronous wrapper can return HTTP 202 when its wait budget expires; follow
its task/status URL instead of submitting the file again.

For Markdown mapping, use the upload flow and `POST /v1/parse/jobs` with
`"output_formats": ["markdown", "markdown_map"]`. The newer convenience
`/v1/tasks` and `/v1/file_parse` endpoints do not currently accept `markdown_map`.
Download the paired output files and validate their hash as described in
[Markdown Map](../../docs/en/reference/markdown_map.md). This image also includes
the [visibility fix](../../docs/en/reference/pdf_text_visibility.md).

## Storage, health and shutdown

- `/data/mineru`: MinerU configuration and model bundles.
- `/data/huggingface` and `/data/cache`: download/runtime caches.
- `/data/uploads`: API uploads and generated artifacts.
- Artifacts expire according to `MINERU_API_RETENTION_SECONDS` (default 86400;
  `0` disables collection). Download results before expiry.
- Job IDs and upload/file indexes are in-memory state. A volume preserves bytes
  and model caches, **not resumable jobs or durable API file IDs** across restarts.
- `docker compose ... down` preserves the named volume. `down -v` destroys it,
  including downloaded models. Use the same `-p mineru-rest` for both modes.
- The API process receives termination signals through Docker's init process.
- The upstream GPU base runs as root. Do not mount sensitive host directories,
  expose the Docker socket, or enable local sources for untrusted clients.

For Docker secrets, mount a key file read-only and set `MINERU_API_KEY_FILE` in a
local Compose override; leave `MINERU_API_KEY` empty. Do not bake keys into images.
The CLI receives the resolved key as a process argument, so host/container
administrators can still inspect it.
