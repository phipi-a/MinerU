"""Validate the container launcher without Docker, model downloads, or a GPU."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "docker/rest-api"


@pytest.fixture
def launcher() -> dict[str, Any]:
    return runpy.run_path(str(DEPLOYMENT / "entrypoint.py"))


def test_gpu_stack_is_explicit(launcher: dict[str, Any]) -> None:
    source = {"MINERU_DEVICE_MODE": "cuda", "MINERU_MODEL_VLM_ENGINE": "llama-cpp"}
    result = launcher["server_environment"](source)
    assert result["MINERU_MODEL_SMALL_BACKEND"] == "torch"
    assert result["MINERU_MODEL_VLM_ENGINE"] == "vllm"
    assert source["MINERU_MODEL_VLM_ENGINE"] == "llama-cpp"


def test_cpu_stack_hides_inherited_cuda(launcher: dict[str, Any]) -> None:
    result = launcher["server_environment"]({"MINERU_DEVICE_MODE": "cpu", "CUDA_VISIBLE_DEVICES": "0"})
    assert result["MINERU_MODEL_SMALL_BACKEND"] == "onnx"
    assert result["MINERU_MODEL_VLM_ENGINE"] == "llama-cpp"
    assert result["CUDA_VISIBLE_DEVICES"] == ""


def test_default_command_is_rest_only(launcher: dict[str, Any]) -> None:
    command = launcher["server_command"]({})
    assert command[:2] == ["mineru-kit", "api-server"]
    assert command[command.index("--tier") + 1] == "standard"
    assert command[command.index("--upload-dir") + 1] == "/data/uploads"
    assert "--allow-local-source" not in command
    assert "--allow-http-source" not in command
    assert "--preload-models" not in command
    assert "--disable-image-analysis" in command


def test_image_analysis_can_be_enabled(launcher: dict[str, Any]) -> None:
    command = launcher["server_command"]({"MINERU_SERVER_IMAGE_ANALYSIS": "1"})
    assert "--disable-image-analysis" not in command


def test_secret_and_preload_options(launcher: dict[str, Any], tmp_path: Path) -> None:
    secret = tmp_path / "api-key"
    secret.write_text("secret-key\n")
    command = launcher["server_command"](
        {"MINERU_API_KEY_FILE": str(secret), "MINERU_SERVER_PRELOAD": "1", "MINERU_API_RETENTION_SECONDS": "0"}
    )
    assert command[command.index("--api-key") + 1] == "secret-key"
    assert command[command.index("--retention-seconds") + 1] == "0"
    assert "--preload-models" in command


@pytest.mark.parametrize(
    "environment",
    [
        {"MINERU_SERVER_TIER": "advanced"},
        {"MINERU_SERVER_CONCURRENCY": "0"},
        {"MINERU_API_RETENTION_SECONDS": "-1"},
        {"MINERU_SERVER_CONCURRENCY": "not-an-integer"},
        {"MINERU_SERVER_PRELOAD": "yes"},
        {"MINERU_SERVER_IMAGE_ANALYSIS": "true"},
        {"MINERU_API_KEY": "one", "MINERU_API_KEY_FILE": "two"},
    ],
)
def test_invalid_options_fail_closed(launcher: dict[str, Any], environment: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        launcher["server_command"](environment)


def test_invalid_device_is_rejected(launcher: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="cuda or cpu"):
        launcher["server_environment"]({"MINERU_DEVICE_MODE": "auto"})


def test_cpu_main_executes_with_selected_environment(launcher: dict[str, Any]) -> None:
    captured = {}

    def execute(file: str, arguments: list[str], environment: dict[str, str]) -> None:
        captured.update(file=file, arguments=arguments, environment=environment)

    main = launcher["main"]
    main.__globals__["os"] = SimpleNamespace(environ={"MINERU_DEVICE_MODE": "cpu"}, execvpe=execute)
    main()
    assert captured["file"] == "mineru-kit"
    assert captured["environment"]["MINERU_MODEL_VLM_ENGINE"] == "llama-cpp"


def test_gpu_main_refuses_silent_cpu_fallback(
    launcher: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    launcher["main"].__globals__["os"] = SimpleNamespace(environ={"MINERU_DEVICE_MODE": "cuda"})
    with pytest.raises(SystemExit) as error:
        launcher["main"]()
    assert error.value.code == 1
    assert "CUDA is unavailable" in capsys.readouterr().err


def test_compose_cpu_does_not_reserve_gpus() -> None:
    config = yaml.safe_load((DEPLOYMENT / "compose.yaml").read_text())
    api = config["services"]["api"]
    assert set(config["services"]) == {"api"}
    assert "deploy" not in api and "gpus" not in api and "devices" not in api
    assert api["environment"]["MINERU_DEVICE_MODE"] == "cpu"
    assert api["environment"]["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert api["volumes"] == ["data:/data"]


def test_compose_gpu_overrides_same_service() -> None:
    config = yaml.safe_load((DEPLOYMENT / "compose.gpu.yaml").read_text())
    api = config["services"]["api"]
    assert set(config["services"]) == {"api"}
    assert "build" not in api and "image" not in api
    assert api["environment"]["MINERU_DEVICE_MODE"] == "cuda"
    assert api["devices"] == ["nvidia.com/gpu=${NVIDIA_GPU_ID:-0}"]
    assert "deploy" not in api
