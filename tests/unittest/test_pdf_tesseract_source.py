"""Independent OCR candidates, producer geometry, and public text-source routing."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
from docvortex.schema import DocumentProperties
from fastapi.testclient import TestClient
from PIL import Image

from mineru.backend import analyze
from mineru.backend.analysis.contracts import AnalysisResult
from mineru.backend.analysis.pdf import pipeline, window
from mineru.backend.analysis.pdf.text import tesseract
from mineru.backend.analysis.pdf.text.selection import SELECTION_FIELD, prepare_text_candidates, select_text_candidates
from mineru.backend.analysis.pdf.text.tesseract import TesseractWord, fill_tesseract_candidates
from mineru.parser import MinerUParser, api_server
from mineru.parser.api_client import MinerUApiParser
from mineru.parser.base import ParseResult
from mineru.parser.task_api import UploadTaskOptions


def block(text: str, box: list[float] | None = None) -> dict[str, Any]:
    return {"type": "text", "bbox": box or [0.1, 0.1, 0.9, 0.3], "content": text}


def words(text: str, y: float = 0.15) -> list[TesseractWord]:
    return [
        TesseractWord(value, (0.15 + i * 0.12, y, 0.25 + i * 0.12, y + 0.03), 95.0, (1, 1, 1))
        for i, value in enumerate(text.split())
    ]


def test_tsv_retains_word_geometry_and_rejects_invalid_records() -> None:
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    records = [
        "5\t1\t2\t3\t4\t1\t10\t20\t30\t10\t96.5\tQuantum",
        "4\t1\t2\t3\t4\t0\t10\t20\t30\t10\t-1\t",
        "5\t1\t2\t3\t4\t2\t90\t20\t30\t10\t96\tout-of-bounds",
        "5\t1\t2\t3\t4\t2\t0\t20\t30\t10\tNaN\tbad-score",
        "5\t1\t2\t3\t4\t2\t0\t20\t0\t10\t90\tzero-width",
    ]
    parsed = tesseract.parse_tsv(header + "\n".join(records), (100, 200))
    assert parsed == [TesseractWord("Quantum", (0.1, 0.1, 0.4, 0.15), 96.5, (2, 3, 4))]


def test_independent_candidates_accept_only_matching_text_and_keep_own_boxes() -> None:
    good = block("The quantum particle moves.")
    bad = block("Energy 900", [0.1, 0.4, 0.9, 0.6])
    original_bad = deepcopy(bad)
    candidates = prepare_text_candidates([[good, bad]])
    source_words = words("The quantum particle moves.") + words("Energy 100", y=0.45)
    fill_tesseract_candidates(candidates, source_words)
    select_text_candidates(candidates, text_source="tesseract")
    assert good[SELECTION_FIELD]["source"] == "tesseract"
    assert [t["bbox"] for t in good["_position_tokens"]] == [list(w.bbox) for w in source_words[:4]]
    assert bad[SELECTION_FIELD]["source"] == "vlm"
    assert {k: v for k, v in bad.items() if k != SELECTION_FIELD} == original_bad
    assert good["_position_tokens"][1]["start"] == 4


def test_each_word_belongs_to_only_one_block_and_lines_keep_tsv_order() -> None:
    a, b = block("Quantum theory"), block("Quantum theory")
    candidates = prepare_text_candidates([[a, b]])
    source_words = [words("Quantum")[0], TesseractWord("theory", (0.1, 0.2, 0.3, 0.25), 90, (1, 1, 2))]
    fill_tesseract_candidates(candidates, source_words)
    assert a["content"] == "Quantum\ntheory"
    assert b["content"] == ""
    assert len(a["_position_lines"]) == 2


@pytest.mark.parametrize(
    "content",
    [r"Energy \(E_1\) is conserved.", [{"type": "text", "content": "Energy "}, {"type": "equation_inline", "content": "E_1"}]],
)
def test_inline_formulas_are_kept_with_vlm(content: Any) -> None:
    original = block(content)
    original["lines"] = [{"bbox": [0.1, 0.1, 0.9, 0.3]}]
    expected = deepcopy(original)
    candidates = prepare_text_candidates([[original]])
    fill_tesseract_candidates(candidates, words("Energy is conserved."))
    select_text_candidates(candidates, text_source="tesseract")
    assert original[SELECTION_FIELD]["reason"] == "inline_math_preserved"
    assert {k: v for k, v in original.items() if k != SELECTION_FIELD} == expected


def test_tesseract_positions_and_diagnostics_survive_normalization_and_export() -> None:
    from mineru.backend.analysis.pdf.normalization import _normalize_pdf_model_list
    from docvortex.postprocess.document import model_json_to_middle_json

    page = block("The quantum particle moves.")
    candidates = prepare_text_candidates([[page]])
    fill_tesseract_candidates(candidates, words("The quantum particle moves."))
    select_text_candidates(candidates, text_source="tesseract")
    result = AnalysisResult([[page]], "xhigh", "all", 0.1, text_source="tesseract")
    _normalize_pdf_model_list(result.model_list)
    model = analyze._build_model_json(result, "pdf", [526], DocumentProperties())
    assert model.extensions["mineru"] == {"tier": "advanced", "parse_mode": "all"}
    assert model.extensions["mineru_text_selection"]["text_source"] == "tesseract"
    assert model.extensions["mineru_position_sources"]["blocks"][0]["page_idx"] == 526
    parsed = ParseResult(model_json_to_middle_json(model), model)
    markdown, mapping = parsed.markdown_with_map()
    assert markdown == "The quantum particle moves."
    assert mapping["pages"][0]["page_idx"] == 526
    elements = mapping["pages"][0]["blocks"][0]["elements"]
    assert any(element["level"] == "word" for element in elements)


def test_finish_all_uses_tesseract_without_native_text_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    page = block("The quantum particle moves.")
    state = SimpleNamespace(
        images_list=[],
        window_pages=[],
        images_pil_list=[],
        np_images=[],
        images_layout_res=[],
        vl_style_layout_blocks=[],
        page_text_geometries=None,
        page_vector_geometries=None,
        page_snapshots=None,
        window=SimpleNamespace(start=0),
        tesseract_words=[words("The quantum particle moves.")],
    )
    monkeypatch.setattr(window, "_convert_vlm_results_to_model_list", lambda value: value)
    monkeypatch.setattr(window, "_normalize_xhigh_vlm_blocks", lambda *_: None)
    monkeypatch.setattr(window, "_apply_layout_title_split", lambda *_: None)
    monkeypatch.setattr(window, "_supplement_missing_image_block_containers", lambda *_: None)
    monkeypatch.setattr(window, "_attach_visual_block_images", lambda *_, **__: None)
    seen = []

    def process(*args: Any, **kwargs: Any) -> list[Any]:
        seen.append(args[3])
        assert args[2][0][0]["content"] == "The quantum particle moves."
        return args[2]

    monkeypatch.setattr(window, "_process_text_and_formulas", process)
    result = window._finish_pdf_window(
        state, [[page]], effort="xhigh", parse_mode="all", hybrid_model=object(), positions=True, text_source="tesseract"
    )
    assert seen == ["ocr"]
    assert result[0][0][SELECTION_FIELD]["source"] == "tesseract"


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("text_source", ["tesseract", "both"])
def test_pipeline_forwards_text_source_without_changing_all(
    monkeypatch: pytest.MonkeyPatch, use_async: bool, text_source: str
) -> None:
    calls = []
    monkeypatch.setattr(tesseract, "check_tesseract", lambda: "/bin/tesseract")

    def prepare(state: Any, *_args: Any) -> None:
        state.parse_mode = "all"
        state.document = state.hybrid_model = state.predictor = object()

    def process(*_args: Any, **kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return []

    async def async_process(*args: Any, **kwargs: Any) -> list[Any]:
        return process(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_prepare_analysis", prepare)
    monkeypatch.setattr(pipeline, "_close_analysis", lambda _state: None)
    monkeypatch.setattr(pipeline, "process_pdf_windows", process)
    monkeypatch.setattr(pipeline, "aio_process_pdf_windows", async_process)
    monkeypatch.setattr(pipeline, "_build_pdf_analysis_result", lambda *_args: AnalysisResult([], "xhigh", "all", 0.0))
    if use_async:
        asyncio.run(pipeline.aio_analyze_pdf(b"pdf", effort="xhigh", parse_mode="all", text_source=text_source))
    else:
        pipeline.analyze_pdf(b"pdf", effort="xhigh", parse_mode="all", text_source=text_source)
    assert calls[0]["text_source"] == text_source
    assert calls[0]["positions"] is True
    assert calls[0]["parse_mode"] == "all"


@pytest.mark.parametrize("tier,mode", [("basic", "all"), ("advanced", "txt"), ("advanced", "ocr"), ("advanced", "auto")])
@pytest.mark.parametrize("text_source", ["tesseract", "both"])
def test_unsupported_combinations_fail_before_model_loading(tier: str, mode: str, text_source: str) -> None:
    with pytest.raises(ValueError, match="requires"):
        MinerUParser(tier=tier, parse_mode=mode, text_source=text_source)


@pytest.mark.parametrize("text_source", ["tesseract", "both"])
def test_api_client_and_task_options_expose_text_source(text_source: str) -> None:
    parser = MinerUApiParser(tier="advanced", ocr_mode="all", text_source=text_source)
    assert parser._build_payload({"type": "file_id", "file_id": "file-1"}, "r1")["text_source"] == text_source
    assert "text_source" not in MinerUApiParser()._build_payload({}, "")
    assert UploadTaskOptions(text_source=text_source).text_source == text_source


def test_subprocess_timeout_reports_page_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tesseract, "check_tesseract", lambda: "tesseract")

    def timeout(*args: Any, **kwargs: Any) -> None:
        assert kwargs["env"]["OMP_THREAD_LIMIT"] == "1"
        raise subprocess.TimeoutExpired("tesseract", 180)

    monkeypatch.setattr(tesseract.subprocess, "run", timeout)
    with Image.new("RGB", (20, 20)) as image, pytest.raises(RuntimeError, match="page 65"):
        tesseract.recognize_pages([image], page_start=64)


def test_missing_executable_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tesseract.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="requires Tesseract"):
        pipeline.analyze_pdf(b"not PDF", effort="xhigh", parse_mode="all", text_source="tesseract")


@pytest.mark.parametrize("route", ["jobs", "tasks", "file_parse"])
@pytest.mark.parametrize("text_source", ["tesseract", "both"])
def test_api_forwards_text_source_for_json_and_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, text_source: str
) -> None:
    import time
    from test_markdown_map import make_result

    seen = []

    async def parse(_path: str, **options: Any) -> ParseResult:
        seen.append(options)
        return make_result([{"type": "text", "content": "Verified source."}])

    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(api_server, "parse_async", parse)
    app = api_server.create_app(upload_dir=str(tmp_path), tier="standard")
    with TestClient(app) as client:
        if route == "jobs":
            response = client.post(
                "/v1/parse/jobs",
                json={
                    "files": [{"source": {"type": "inline", "name": "page.pdf", "data": "JVBERg=="}}],
                    "tier": "advanced",
                    "ocr_mode": "all",
                    "text_source": text_source,
                },
            )
            assert response.status_code == 202, response.text
            job_id = response.json()["job_id"]
        else:
            response = client.post(
                f"/v1/{route}",
                params={"tier": "advanced", "text_source": text_source, "filename": "page.pdf"},
                content=b"%PDF-test",
                headers={"Content-Type": "application/octet-stream"},
            )
            assert response.status_code in {200, 202}, response.text
            job_id = response.json()["task_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"/v1/parse/jobs/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed", job
    assert seen[0]["text_source"] == text_source
    assert seen[0]["ocr_mode"] == "all"


def test_api_rejects_tesseract_with_txt_before_job_creation(tmp_path: Path) -> None:
    app = api_server.create_app(upload_dir=str(tmp_path), tier="standard")
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "inline", "name": "page.pdf", "data": "JVBERg=="}}],
                "ocr_mode": "txt",
                "text_source": "tesseract",
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "text_source"
    assert not app.state.job_store._jobs


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("text_source", ["tesseract", "both"])
def test_window_runs_tesseract_outside_model_lock_and_closes_state(
    monkeypatch: pytest.MonkeyPatch, use_async: bool, text_source: str
) -> None:
    from contextlib import contextmanager

    locked = False
    seen = []
    closed = []
    state = SimpleNamespace(
        images_pil_list=[], vl_style_layout_blocks=[], high_vlm_blocks=[], close=lambda: closed.append(True)
    )

    @contextmanager
    def lock(_device: str) -> Any:
        nonlocal locked
        locked = True
        try:
            yield
        finally:
            locked = False

    def prepare(*_args: Any, **options: Any) -> Any:
        assert locked
        assert options["text_source"] == text_source
        return state

    def recognize(*_args: Any, **options: Any) -> list[Any]:
        assert not locked
        seen.append("ocr")
        return [[TesseractWord("Quantum", (0.1, 0.1, 0.2, 0.2), 90, (1, 1, 1))]]

    def finish(_state: Any, _result: Any, **options: Any) -> list[Any]:
        assert locked
        assert options["text_source"] == text_source
        assert state.tesseract_words[0][0].text == "Quantum"
        seen.append("finish")
        return [[]]

    def predict(**options: Any) -> list[Any]:
        assert "not_extract_list" not in options
        seen.append("vlm")
        return [[]]

    async def async_predict(**options: Any) -> list[Any]:
        return predict(**options)

    monkeypatch.setattr(window, "local_model_stage", lock)
    monkeypatch.setattr(window, "_prepare_pdf_window", prepare)
    monkeypatch.setattr(window, "_finish_pdf_window", finish)
    monkeypatch.setattr(window, "recognize_pages", recognize)
    monkeypatch.setattr(window, "get_document_render_session", lambda _: None)
    monkeypatch.setattr(window, "trim_process_heap", lambda: None)
    model = SimpleNamespace(device="cpu")
    predictor = SimpleNamespace(batch_two_step_extract=predict, aio_batch_two_step_extract=async_predict)
    options = {
        "effort": "xhigh",
        "parse_mode": "all",
        "image_analysis": False,
        "hybrid_model": model,
        "vlm_predictor": predictor,
        "text_source": text_source,
    }
    if use_async:
        asyncio.run(window.aio_process_pdf_windows(b"pdf", SimpleNamespace(page_count=1), **options))
    else:
        window._process_pdf_window(b"pdf", object(), window._ProcessingWindow(0, 1, 0, 1), page_count=1, **options)
    assert seen == ["ocr", "vlm", "finish"]
    assert closed == [True]


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("text_source", ["tesseract", "both"])
def test_cli_passes_all_and_tesseract_to_local_and_remote_parsers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool, text_source: str
) -> None:
    from typer.testing import CliRunner
    from mineru.kit.commands import parse as command
    from mineru.kit.main import app
    from test_markdown_map import make_result

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    seen = []
    result = make_result([{"type": "text", "content": "Verified CLI."}])

    def local(_path: Path, **options: Any) -> ParseResult:
        seen.append(options)
        return result

    class Remote:
        def __init__(self, **options: Any) -> None:
            seen.append(options)

        def parse(self, _path: Path, **options: Any) -> ParseResult:
            return result

    monkeypatch.setattr(command, "local_parse", local)
    monkeypatch.setattr(command, "MinerUApiParser", Remote)
    args = [
        "parse",
        str(source),
        "-o",
        str(tmp_path / "book.md"),
        "--tier",
        "advanced",
        "--ocr-mode",
        "all",
        "--text-source",
        text_source,
    ]
    if remote:
        args.extend(["--remote-url", "http://localhost:8000"])
    response = CliRunner().invoke(app, args)
    assert response.exit_code == 0, response.output
    assert seen[0]["text_source"] == text_source
    assert seen[0]["ocr_mode"] == "all"


def test_tesseract_window_uses_300_dpi_without_reading_native_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    image = Image.new("RGB", (300, 300))
    page = MagicMock()
    monkeypatch.setattr(window, "_get_window_pdf_pages", lambda *_: [page])
    seen = []

    def render(**options: Any) -> list[Any]:
        seen.append(options["dpi"])
        return [{"img_pil": image, "scale": 300 / 72}]

    monkeypatch.setattr(window, "load_images_from_pdf_bytes_range", render)
    model = MagicMock()
    model.layout_model.batch_predict.return_value = [[]]
    state = window._prepare_pdf_window(
        b"",
        object(),
        window._ProcessingWindow(0, 1, 0, 1),
        page_count=1,
        effort="xhigh",
        parse_mode="all",
        hybrid_model=model,
        text_source="tesseract",
    )
    assert seen == [300]
    assert state.page_snapshots is None
    assert state.page_text_geometries is None
    page.get_chars_with_geometry.assert_not_called()
    page.get_snapshot.assert_not_called()
    state.close()
