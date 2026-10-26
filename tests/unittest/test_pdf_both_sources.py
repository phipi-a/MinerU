"""Rank independent PDF and OCR candidates against the original VLM reference."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from docvortex.schema import DocumentProperties

from mineru.backend import analyze
from mineru.backend.analysis.contracts import AnalysisResult
from mineru.backend.analysis.pdf import window
from mineru.backend.analysis.pdf.text.selection import (
    SELECTION_FIELD,
    prepare_text_candidates,
    select_best_text_candidates,
)
from mineru.backend.analysis.pdf.text.tesseract import fill_tesseract_candidates
from test_pdf_all_mode import make_block, set_native
from test_pdf_tesseract_source import words

REFERENCE = "The quantum particle passes through the narrow opening."


@pytest.mark.parametrize(
    ("pdf_text", "ocr_text", "expected"),
    [
        ("The quantum particle passes through a narrow opening.", REFERENCE, "tesseract"),
        (REFERENCE, "The quantum particle passes through a narrow opening.", "txt"),
        (REFERENCE, REFERENCE, "txt"),
        ("Corrupted native PDF content.", "Unrelated OCR paragraph.", "vlm"),
        ("", REFERENCE, "tesseract"),
        (REFERENCE, "", "txt"),
        ("", "", "vlm"),
        ("Energy 100", "Energy 200", "vlm"),
    ],
)
def test_best_matching_eligible_source_wins(pdf_text: str, ocr_text: str, expected: str) -> None:
    reference = "Energy 900" if pdf_text == "Energy 100" else REFERENCE
    live = make_block(reference)
    ocr = deepcopy(live)
    native_candidates = prepare_text_candidates([[live]])
    ocr_candidates = prepare_text_candidates([[ocr]])
    set_native(live, pdf_text)
    set_native(ocr, ocr_text)
    live["_position_tokens"][0]["bbox"] = [0.1, 0.1, 0.2, 0.2]
    ocr["_position_tokens"][0]["bbox"] = [0.4, 0.1, 0.5, 0.2]
    select_best_text_candidates(native_candidates, ocr_candidates)
    decision = live[SELECTION_FIELD]
    assert decision["source"] == expected
    assert set(decision["candidates"]) == {"pdf", "tesseract"}
    if expected == "vlm":
        assert live["content"] == reference
        assert "_position_tokens" not in live
    else:
        assert live["content"] == (pdf_text if expected == "txt" else ocr_text)
        assert live["_position_tokens"][0]["bbox"] == ([0.1, 0.1, 0.2, 0.2] if expected == "txt" else [0.4, 0.1, 0.5, 0.2])


def test_higher_similarity_without_own_word_geometry_does_not_displace_valid_source() -> None:
    live = make_block(REFERENCE)
    ocr = deepcopy(live)
    native = prepare_text_candidates([[live]])
    tess = prepare_text_candidates([[ocr]])
    set_native(live, "The quantum particle passes through a narrow opening.")
    set_native(ocr, REFERENCE, words=False)
    select_best_text_candidates(native, tess)
    decision = live[SELECTION_FIELD]
    assert decision["source"] == "txt"
    assert decision["candidates"]["tesseract"]["similarity"] == 1.0
    assert decision["candidates"]["tesseract"]["reason"] == "missing_native_word_positions"


def test_both_sources_remain_independent_of_native_mutations() -> None:
    live = make_block("The quantum particle moves.")
    original = deepcopy(live)
    ocr_candidates = prepare_text_candidates([[deepcopy(live)]])
    native = prepare_text_candidates([[live]])
    set_native(live, "Corrupted extracted text.")
    live["style"] = {"font_name": "PDF-only font"}
    fill_tesseract_candidates(ocr_candidates, words("The quantum particle moves."))
    select_best_text_candidates(native, ocr_candidates)
    assert live[SELECTION_FIELD]["source"] == "tesseract"
    assert "style" not in live
    assert live["content"] == original["content"]
    assert ocr_candidates[0].vlm_block == original


def test_inline_formula_guards_apply_to_tesseract_when_both_enabled() -> None:
    live = make_block(r"The energy \(E_1\) is the ground state energy.")
    original = deepcopy(live)
    ocr = deepcopy(live)
    native = prepare_text_candidates([[live]])
    tess = prepare_text_candidates([[ocr]])
    set_native(live, "Corrupted extraction.")
    set_native(ocr, r"The energy \(E_1\) is the ground state energy.")
    select_best_text_candidates(native, tess)
    assert live[SELECTION_FIELD]["source"] == "vlm"
    assert live[SELECTION_FIELD]["candidates"]["tesseract"]["reason"] == "inline_math_preserved"
    assert {k: v for k, v in live.items() if k != SELECTION_FIELD} == original


def test_both_rejects_unpaired_blocks() -> None:
    with pytest.raises(ValueError, match="candidate count mismatch"):
        select_best_text_candidates(prepare_text_candidates([[make_block(REFERENCE)]]), [])


def test_window_compares_both_sources_against_untouched_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.backend.analysis.pdf.normalization import _normalize_pdf_model_list
    from docvortex.postprocess.document import model_json_to_middle_json
    from mineru.parser.base import ParseResult

    page = make_block("The quantum particle moves.")
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

    def native_fill(*args: Any, **kwargs: Any) -> list[Any]:
        assert args[3] == "txt"
        assert page["content"] == ""
        set_native(page, "Corrupted native text.")
        return args[2]

    monkeypatch.setattr(window, "_process_text_and_formulas", native_fill)
    pages = window._finish_pdf_window(
        state, [[page]], effort="xhigh", parse_mode="all", hybrid_model=object(), positions=True, text_source="both"
    )
    assert page[SELECTION_FIELD]["source"] == "tesseract"
    _normalize_pdf_model_list(pages)
    result = AnalysisResult(pages, "xhigh", "all", 0.1, text_source="both")
    model = analyze._build_model_json(result, "pdf", [20], DocumentProperties())
    diagnostic = model.extensions["mineru_text_selection"]
    assert diagnostic["text_source"] == "both"
    assert diagnostic["blocks"][0]["candidates"]["tesseract"]["similarity"] == 1.0
    parsed = ParseResult(model_json_to_middle_json(model), model)
    markdown, mapping = parsed.markdown_with_map()
    assert markdown == "The quantum particle moves."
    assert mapping["pages"][0]["page_idx"] == 20
    assert any(element["level"] == "word" for element in mapping["pages"][0]["blocks"][0]["elements"])


def test_each_block_can_choose_a_different_producer() -> None:
    originals = [make_block(REFERENCE) for _ in range(3)]
    ocr_blocks = deepcopy(originals)
    native = prepare_text_candidates([originals])
    tess = prepare_text_candidates([ocr_blocks])
    for index, (pdf, ocr) in enumerate(zip(originals, ocr_blocks, strict=True)):
        set_native(pdf, REFERENCE if index == 0 else "Corrupted native extraction.")
        set_native(ocr, REFERENCE if index == 1 else "Unrelated OCR output.")
    select_best_text_candidates(native, tess)
    assert [block[SELECTION_FIELD]["source"] for block in originals] == ["txt", "tesseract", "vlm"]
    assert all(block["content"] == REFERENCE for block in originals)
