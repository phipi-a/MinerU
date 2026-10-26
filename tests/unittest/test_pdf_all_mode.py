"""Block-level PDF/VLM text selection and its public API contract."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from docvortex.schema import DocumentProperties
from fastapi.testclient import TestClient

from mineru.backend import analyze
from mineru.backend.analysis.contracts import AnalysisResult
from mineru.backend.analysis.pdf import normalization, pipeline, window
from mineru.backend.analysis.pdf.text.selection import (
    SELECTION_FIELD,
    prepare_text_candidates,
    select_text_candidates,
)
from mineru.integrations.docvortex import MinerUMetadata
from mineru.parser.api_server import CreateJobRequest, create_app
from mineru.parser.base import ParseResult
from mineru.parser.task_api import UploadTaskOptions


def make_block(text: str) -> dict[str, Any]:
    return {"type": "text", "bbox": [0.1, 0.1, 0.9, 0.2], "content": text}


def set_native(block: dict[str, Any], text: str, *, words: bool = True) -> None:
    block["content"] = text
    block["_position_content"] = text
    block["_position_lines"] = [{"line_id": 0, "bbox": block["bbox"]}]
    if words:
        block["_position_tokens"] = [{"type": "word", "level": "word", "text": text, "bbox": block["bbox"]}]


def test_selection_is_per_block_and_restores_rejected_vlm() -> None:
    good = make_block("The quantum particle passes through the opening.")
    bad = make_block("zero matrix 476")
    original = deepcopy(bad)
    formula = {"type": "equation", "content": "x=1", "bbox": [0.1, 0.3, 0.9, 0.4]}
    candidates = prepare_text_candidates([[good, bad, formula]])
    assert formula["content"] == "x=1"
    set_native(good, "The quantum particle passes through the opening.")
    set_native(bad, "4\x1f,) '\x1b.,#2 jml")
    bad["style"] = {"font_name": "bad candidate"}
    select_text_candidates(candidates)
    assert good[SELECTION_FIELD]["source"] == "txt"
    assert good["_position_tokens"]
    assert bad[SELECTION_FIELD]["source"] == "vlm"
    assert bad[SELECTION_FIELD]["reason"] == "invalid_native_characters"
    assert {key: value for key, value in bad.items() if key != SELECTION_FIELD} == original


@pytest.mark.parametrize(
    ("native", "vlm", "accepted"),
    [
        ("The magniﬁcent particle moves.", "The magnificent particle moves.", True),
        ("The particle crosses a nar-\nrow opening.", "The particle crosses a narrow opening.", True),
        ("This is a totally different paragraph.", "The particle passes through the opening.", False),
        ("energy 100", "energy 900", False),
        ("", "Valid reference paragraph.", False),
        ("Native text without a reference.", "", False),
    ],
)
def test_similarity_and_short_text_guards(native: str, vlm: str, accepted: bool) -> None:
    block = make_block(vlm)
    candidates = prepare_text_candidates([[block]])
    set_native(block, native)
    select_text_candidates(candidates)
    assert block[SELECTION_FIELD]["source"] == ("txt" if accepted else "vlm")
    if not accepted:
        assert "_position_tokens" not in block


def test_styles_and_latex_spelling_do_not_create_false_text_conflicts() -> None:
    block = make_block("The energy \\(E_1\\) is the ground state energy.")
    candidates = prepare_text_candidates([[block]])
    set_native(block, "The energy is the ground state energy.")
    block["content"] = [
        {"type": "text", "content": "The energy ", "styles": ["bold"]},
        {"type": "equation_inline", "content": "E _ {1}"},
        {"type": "text", "content": " is the ground state energy."},
    ]
    select_text_candidates(candidates)
    assert block[SELECTION_FIELD]["source"] == "txt"


def test_matching_ocr_text_without_native_word_positions_uses_vlm() -> None:
    block = make_block("Readable words but no trustworthy word geometry.")
    candidates = prepare_text_candidates([[block]])
    set_native(block, "Readable words but no trustworthy word geometry.", words=False)
    select_text_candidates(candidates)
    assert block[SELECTION_FIELD]["reason"] == "missing_native_word_positions"
    assert "_position_lines" not in block


def test_all_mode_metadata_and_diagnostics_roundtrip_without_native_boxes() -> None:
    block = make_block("zero matrix 476")
    candidates = prepare_text_candidates([[block]])
    set_native(block, "4\x1f,) '\x1b.,#2 jml")
    select_text_candidates(candidates)
    result = AnalysisResult([[block]], "xhigh", "all", 0.1)
    normalization._normalize_pdf_model_list(result.model_list)
    model = analyze._build_model_json(result, "pdf", [526], DocumentProperties())
    assert model.extensions["mineru"] == {"tier": "advanced", "parse_mode": "all"}
    assert "mineru_position_sources" not in model.extensions
    decision = model.extensions["mineru_text_selection"]["blocks"][0]
    assert decision["page_idx"] == 526
    assert decision["index"] == 0
    assert decision["source"] == "vlm"
    assert SELECTION_FIELD not in model.pages[0][0]

    from docvortex.postprocess.document import model_json_to_middle_json

    middle = model_json_to_middle_json(model)
    parsed = ParseResult(middle, model)
    restored = ParseResult.from_json(parsed.to_json())
    assert restored.middle_json.extensions["mineru_text_selection"] == model.extensions["mineru_text_selection"]
    markdown, mapping = parsed.markdown_with_map()
    assert markdown == "zero matrix 476"
    assert mapping["pages"][0]["page_idx"] == 526
    assert mapping["pages"][0]["blocks"][0]["elements"][0]["level"] == "block"


def test_all_keeps_full_vlm_recognition_enabled() -> None:
    from types import SimpleNamespace

    state = SimpleNamespace(images_pil_list=[], high_vlm_blocks=[])
    assert "not_extract_list" not in window._inference_options(state, "xhigh", "all", False)
    assert "not_extract_list" in window._inference_options(state, "xhigh", "txt", False)


@pytest.mark.parametrize("use_async", [False, True])
def test_all_collects_positions_even_without_a_map_request(monkeypatch: pytest.MonkeyPatch, use_async: bool) -> None:
    calls = []

    def prepare(state: Any, *_args: Any) -> None:
        state.parse_mode = "all"
        state.document = object()
        state.hybrid_model = object()
        state.predictor = object()

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
        result = asyncio.run(pipeline.aio_analyze_pdf(b"pdf", effort="xhigh", parse_mode="all"))
    else:
        result = pipeline.analyze_pdf(b"pdf", effort="xhigh", parse_mode="all")
    assert result.parse_mode == "all"
    assert calls[0]["positions"] is True
    assert calls[0]["parse_mode"] == "all"


def test_api_models_accept_all_mode() -> None:
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "inline", "name": "page.pdf", "data": "JVBERg=="}}],
            "tier": "advanced",
            "ocr_mode": "all",
        }
    )
    assert request.ocr_mode == "all"
    assert UploadTaskOptions(ocr_mode="all").ocr_mode == "all"
    assert MinerUMetadata(tier="advanced", parse_mode="all").parse_mode == "all"


def test_all_rejects_non_vlm_tiers_before_loading_models() -> None:
    with pytest.raises(ValueError, match="standard or advanced"):
        pipeline._prepare_analysis(pipeline._PDFAnalysis(), b"not pdf", "flash", "all", None)
    with pytest.raises(ValueError, match="PDF input"):
        analyze._validate_analyze("xhigh", "html", None, "all")


def test_api_rejects_all_on_flash_before_creating_jobs(tmp_path: Path) -> None:
    app = create_app(upload_dir=str(tmp_path), tier="flash")
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "inline", "name": "page.pdf", "data": "JVBERg=="}}],
                "tier": "flash",
                "ocr_mode": "all",
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "ocr_mode"
    assert app.state.job_store._jobs == {}
