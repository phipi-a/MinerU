"""Launch the REST API with an explicit GPU or CPU model stack."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


def server_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Select both inference backends together, without implicit GPU fallback."""
    result = dict(environ)
    device = result.get("MINERU_DEVICE_MODE", "cuda")
    if device not in {"cuda", "cpu"}:
        raise ValueError("MINERU_DEVICE_MODE must be cuda or cpu")
    result["MINERU_DEVICE_MODE"] = device
    result["MINERU_MODEL_SMALL_BACKEND"] = "torch" if device == "cuda" else "onnx"
    result["MINERU_MODEL_VLM_ENGINE"] = "vllm" if device == "cuda" else "llama-cpp"
    if device == "cpu":
        result["CUDA_VISIBLE_DEVICES"] = ""
    return result


def server_command(environ: Mapping[str, str]) -> list[str]:
    """Build CLI arguments without interpreting environment values as shell code."""
    tier = environ.get("MINERU_SERVER_TIER", "standard")
    if tier not in {"flash", "basic", "standard"}:
        raise ValueError("MINERU_SERVER_TIER must be flash, basic, or standard")
    concurrency = int(environ.get("MINERU_SERVER_CONCURRENCY", "1"))
    retention = int(environ.get("MINERU_API_RETENTION_SECONDS", "86400"))
    if concurrency < 1 or retention < 0:
        raise ValueError("Concurrency must be positive and retention must be nonnegative")
    command = [
        "mineru-kit", "api-server", "--host", "0.0.0.0", "--port", "8000",
        "--tier", tier, "--concurrency", str(concurrency),
        "--retention-seconds", str(retention), "--upload-dir", "/data/uploads",
    ]
    key = environ.get("MINERU_API_KEY", "")
    key_file = environ.get("MINERU_API_KEY_FILE", "")
    if key and key_file:
        raise ValueError("Set only one of MINERU_API_KEY and MINERU_API_KEY_FILE")
    if key_file:
        key = Path(key_file).read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("MINERU_API_KEY_FILE must contain a nonempty API key")
    if key:
        command.extend(["--api-key", key])
    preload = environ.get("MINERU_SERVER_PRELOAD", "0")
    if preload not in {"0", "1"}:
        raise ValueError("MINERU_SERVER_PRELOAD must be 0 or 1")
    if preload == "1":
        command.append("--preload-models")
    image_analysis = environ.get("MINERU_SERVER_IMAGE_ANALYSIS", "0")
    if image_analysis not in {"0", "1"}:
        raise ValueError("MINERU_SERVER_IMAGE_ANALYSIS must be 0 or 1")
    if image_analysis == "0":
        command.append("--disable-image-analysis")
    return command


def main() -> None:
    """Replace the entrypoint process so termination reaches the API server."""
    try:
        environ = server_environment(os.environ)
        command = server_command(environ)
        if environ["MINERU_DEVICE_MODE"] == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver/toolkit or use the CPU Compose file.")
        print(
            f"Starting REST API: device={environ['MINERU_DEVICE_MODE']}, "
            f"small_backend={environ['MINERU_MODEL_SMALL_BACKEND']}, "
            f"vlm_engine={environ['MINERU_MODEL_VLM_ENGINE']}",
            flush=True,
        )
        os.execvpe(command[0], command, environ)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"REST API startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
