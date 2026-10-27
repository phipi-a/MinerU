"""Exact Markdown bytes, semantic style ranges, and conservative fallbacks."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from copy import deepcopy
from typing import Any

from mineru.parser import api_server
from mineru.parser.base import DEFAULT_RESULT_FILES, ParseResult
from mineru.types import MiddleJson, ModelJson


def visible_text(span: dict[str, Any]) -> str:
    if span["type"] == "hyperlink":
        return "".join(visible_text(child) for child in span["content"])
    return span["content"] if span["type"] == "text" else f"\\({span['content']}\\)"


def make_result(content: list[dict[str, Any]]) -> ParseResult:
    text = "".join(visible_text(s) for s in content)
    box = [0.1, 0.1, 0.8, 0.2]
    block = {"type": "text", "index": 1, "bbox": box, "content": content}
    tokens = [
        {"type": "word", "level": "word", "text": m.group(), "bbox": [0.1 + i * 0.1, 0.1, 0.2 + i * 0.1, 0.2]}
        for i, m in enumerate(re.finditer(r"\S+", text))
    ]
    payload = {
        "schema": "docvortex.middle",
        "schema_version": "2.0",
        "is_full_document": False,
        "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "4.0.10"}},
        "extensions": {
            "mineru": {"tier": "basic", "parse_mode": "txt"},
            "mineru_position_sources": {
                "version": 1,
                "blocks": [{"page_idx": 5, "index": 1, "bbox": box, "content": deepcopy(content), "tokens": tokens}],
            },
        },
        "pages": [{"page_idx": 5, "blocks": [block]}],
    }
    return ParseResult(MiddleJson.from_dict(payload))


def regions(source_map: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        element
        for page in source_map["pages"]
        for block in page["blocks"]
        for element in block["elements"] + [e for line in block["lines"] for e in line["elements"]]
    ]


def slices(markdown: str, source_map: dict[str, Any]) -> list[str]:
    return [markdown[b["markdown_char_start"] : b["markdown_char_end"]] for b in regions(source_map)]


def test_superscript_word_includes_complete_markup_and_punctuation() -> None:
    result = make_result(
        [
            {"type": "text", "content": "Stefan Gugler"},
            {"type": "text", "content": "1,2†", "styles": ["superscript"]},
            {"type": "text", "content": ", Max"},
        ]
    )
    before = result.to_dict()
    markdown, mapping = result.markdown_with_map()
    assert markdown == result.markdown()
    assert slices(markdown, mapping) == ["Stefan", "Gugler<sup>1,2†</sup>,", "Max"]
    assert mapping["markdown_sha256"] == hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    assert "markdown" not in mapping and "extensions" not in mapping
    assert result.to_dict() == before


def test_repeated_words_use_their_own_ordered_boxes() -> None:
    result = make_result([{"type": "text", "content": "same same same"}])
    markdown, mapping = result.markdown_with_map()
    blocks = regions(mapping)
    assert [b["markdown_char_start"] for b in blocks] == [0, 5, 10]
    assert len({tuple(b["bbox"]) for b in blocks}) == 3
    assert slices(markdown, mapping) == ["same"] * 3


def test_unicode_offsets_are_codepoints_not_bytes_or_utf16() -> None:
    markdown, mapping = make_result([{"type": "text", "content": "😀 café"}]).markdown_with_map()
    assert mapping["offset_units"] == "unicode_codepoints"
    assert mapping["pages"][0]["page_idx"] == 5
    assert [(b["markdown_char_start"], b["markdown_char_end"]) for b in regions(mapping)] == [(0, 1), (2, 6)]
    assert slices(markdown, mapping) == ["😀", "café"]


def test_style_wrappers_at_span_edges_and_spaces() -> None:
    markdown, mapping = make_result([{"type": "text", "content": " alpha beta ", "styles": ["bold"]}]).markdown_with_map()
    assert markdown == " **alpha beta** "
    assert slices(markdown, mapping) == ["**alpha", "beta**"]


@pytest.mark.parametrize("pure_markdown", [False, True])
def test_link_keeps_word_ranges_and_edge_words_own_link_syntax(pure_markdown: bool) -> None:
    url = "https://github.com/microsoft/LoRA"
    result = make_result(
        [
            {"type": "text", "content": "GPT-2 at "},
            {"type": "hyperlink", "url": url, "content": [{"type": "text", "content": url}]},
            {"type": "text", "content": ". See "},
            {
                "type": "hyperlink",
                "url": "https://e.x",
                "content": [{"type": "text", "content": "the *repo*", "styles": ["bold"]}],
            },
        ]
    )
    markdown, mapping = result.markdown_with_map(pure_markdown=pure_markdown)
    assert markdown == f"GPT-2 at [{url}]({url}). See [**the \\*repo\\***](https://e.x)"
    assert slices(markdown, mapping) == ["GPT-2", "at", f"[{url}]({url}).", "See", "[**the", "\\*repo\\***](https://e.x)"]
    assert all(region["level"] == "word" for region in regions(mapping))


def test_formula_is_atomic_and_has_its_real_delimiters() -> None:
    result = make_result([{"type": "text", "content": "Value "}, {"type": "equation_inline", "content": "x_1"}])
    tokens = result.middle_json.extensions["mineru_position_sources"]["blocks"][0]["tokens"]
    tokens[-1] = {"type": "inline_math", "latex": "x_1", "level": "formula", "bbox": [0.3, 0.1, 0.4, 0.2]}
    markdown, mapping = result.markdown_with_map()
    assert regions(mapping)[-1]["level"] == "formula"
    assert slices(markdown, mapping)[-1] == markdown[len("Value ") :]


def test_missing_provenance_falls_back_to_existing_block_box() -> None:
    result = make_result([{"type": "text", "content": "Plain text"}])
    result.middle_json.extensions.pop("mineru_position_sources")
    markdown, mapping = result.markdown_with_map()
    assert slices(markdown, mapping) == [markdown]
    assert mapping["pages"][0]["blocks"][0]["level"] == "block"


@pytest.mark.parametrize("pure_markdown", [False, True])
@pytest.mark.parametrize("empty_only", [False, True])
def test_empty_pages_preserve_source_map_page_offsets(pure_markdown: bool, empty_only: bool) -> None:
    from mineru.parser.positioned import positioned_document
    from mineru.types import PageInfo

    result = make_result([{"type": "text", "content": "Plain text"}])
    text_page = result.pages[0]
    result.middle_json.pages = [PageInfo(page_idx=0, blocks=[])]
    if not empty_only:
        result.middle_json.pages.extend([text_page, PageInfo(page_idx=6, blocks=[])])
    before = result.to_dict()
    assert "blocks" not in before["pages"][0]

    positioned = positioned_document(result.middle_json, None)
    markdown, mapping = result.markdown_with_map(pure_markdown=pure_markdown)

    assert markdown == result.markdown(pure_markdown=pure_markdown)
    assert mapping["markdown_sha256"] == hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    assert mapping["pages"][0] == {"page_idx": 0, "blocks": []}
    if empty_only:
        assert markdown == ""
        assert positioned["extensions"]["mineru_positions"]["entries"] == []
        assert len(mapping["pages"]) == 1
    else:
        assert [page["page_idx"] for page in mapping["pages"]] == [0, 5, 6]
        assert mapping["pages"][2]["blocks"] == []
        assert mapping["pages"][1]["blocks"][0]["element"] == "/pages/1/blocks/0"
        assert positioned["extensions"]["mineru_positions"]["entries"][0]["element"] == "/pages/1/blocks/0"
        assert slices(markdown, mapping) == ["Plain", "text"]
    assert result.to_dict() == before


def test_context_sensitive_escaping_is_not_guessed() -> None:
    result = make_result([{"type": "text", "content": "<b> text </b>"}])
    markdown, mapping = result.markdown_with_map()
    assert markdown == result.markdown()
    for block in mapping["pages"][0]["blocks"]:
        assert 0 <= block["markdown_char_start"] < block["markdown_char_end"] <= len(markdown)


def test_boundary_rewrites_still_use_explicit_group_ranges() -> None:
    result = make_result([{"type": "text", "content": "micro-"}])
    block = result.pages[0].blocks[0]
    other = block.model_copy(deep=True, update={"index": 2, "continues_prev": True})
    other.content[0].content = "kinetics"
    result.pages[0].blocks.append(other)
    markdown, mapping = result.markdown_with_map()
    assert markdown == result.markdown()
    assert all(b["level"] == "group" for b in regions(mapping))
    assert slices(markdown, mapping) == [markdown, markdown]


@pytest.mark.parametrize("include_map", [False, True])
@pytest.mark.parametrize("pure_markdown", [False, True])
def test_saved_map_preserves_model_position_provenance(tmp_path: Path, include_map: bool, pure_markdown: bool) -> None:
    from mineru.parser.writer import FileBasedDataWriter

    result = make_result([{"type": "text", "content": "😀 café", "styles": ["underline"]}])
    result._model_output = ModelJson(
        pages=[],
        page_index_map=[],
        metadata=result.middle_json.metadata,
        extensions={"mineru_position_sources": result.middle_json.extensions.pop("mineru_position_sources")},
    )
    files = DEFAULT_RESULT_FILES | ({"markdown_map"} if include_map else set())
    result.save(FileBasedDataWriter(str(tmp_path)), files=files, pure_markdown=pure_markdown)
    assert (tmp_path / "markdown_map.json").exists() is include_map
    markdown_bytes = (tmp_path / "markdown.md").read_bytes()
    assert markdown_bytes.decode() == ("😀 café" if pure_markdown else "<u>😀 café</u>")
    if include_map:
        mapping = json.loads((tmp_path / "markdown_map.json").read_bytes())
        assert hashlib.sha256(markdown_bytes).hexdigest() == mapping["markdown_sha256"]
        assert slices(markdown_bytes.decode("utf-8"), mapping) == (["😀", "café"] if pure_markdown else ["<u>😀", "café</u>"])
        assert all(region["level"] == "word" for region in regions(mapping))


@pytest.mark.parametrize("include_markdown", [False, True])
@pytest.mark.parametrize("include_zip", [False, True])
@pytest.mark.parametrize("pure_markdown", [False, True])
def test_rest_map_is_a_separate_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_markdown: bool,
    include_zip: bool,
    pure_markdown: bool,
) -> None:
    result = make_result([{"type": "text", "content": "😀 café", "styles": ["underline"] if pure_markdown else []}])
    payload = result.middle_json.to_dict(skip_defaults=False)
    image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
    payload["pages"][0]["blocks"].insert(
        0,
        {
            "type": "image",
            "index": 0,
            "bbox": [0.1, 0.3, 0.8, 0.5],
            "content": [
                {
                    "type": "image_body",
                    "index": 0,
                    "content": "",
                    "image_base64": f"data:image/png;base64,{image_base64}",
                }
            ],
        },
    )
    result = ParseResult.from_dict(payload)
    original = result.middle_json.to_dict(skip_defaults=False)
    calls = []

    async def fake_parse(path: str, **kwargs: Any) -> ParseResult:
        calls.append(kwargs)
        return result

    monkeypatch.setattr(api_server, "parse_async", fake_parse)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    app = api_server.create_app(upload_dir=str(tmp_path / "uploads"), tier="flash", allow_local_source=True)
    with TestClient(app) as client:
        assert "markdown_map" in client.get("/v1/health").json()["features"]["output_formats"]
        response = client.post(
            "/v1/parse/jobs",
            json={
                "files": [{"source": {"type": "local", "path": str(source)}}],
                "tier": "flash",
                "pure_markdown": pure_markdown,
                "output_formats": (
                    ["markdown_map"] + (["markdown"] if include_markdown else []) + (["zip"] if include_zip else [])
                ),
            },
        )
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        for _ in range(200):
            job = client.get(f"/v1/parse/jobs/{job_id}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert job["status"] == "completed", job
        outputs = job["files"][0]["output_files"]
        ref = outputs["markdown_map"]
        mapping = client.get(f"/v1/files/{ref['file_id']}/content").json()
        assert mapping == result.markdown_map(pure_markdown=pure_markdown)
        assert not outputs.get("middle_json")
        assert calls[0]["positions"] is True
        stored_map = app.state.file_store.get_file(ref["file_id"])
        assert stored_map.job_id == job_id
        assert stored_map.expires_at is not None
        if include_markdown:
            ref = outputs["markdown"]
            markdown_bytes = client.get(f"/v1/files/{ref['file_id']}/content").content
            assert hashlib.sha256(markdown_bytes).hexdigest() == mapping["markdown_sha256"]
        if include_zip:
            ref = outputs["zip"]
            archive_bytes = client.get(f"/v1/files/{ref['file_id']}/content").content
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                bundled_map = json.loads(archive.read("markdown_map.json"))
                images = [name for name in archive.namelist() if name.startswith("images/")]
                if not include_markdown:
                    # Only requested files are bundled; the map alone references no images.
                    assert archive.namelist() == ["markdown_map.json"]
                    assert bundled_map["markdown_sha256"] != mapping["markdown_sha256"]
                    return
                assert sorted(set(archive.namelist()) - set(images)) == ["markdown.md", "markdown_map.json"]
                markdown_bytes = archive.read("markdown.md")
                assert len(images) == 1
                assert archive.read(images[0]) == base64.b64decode(image_base64)
                markdown = markdown_bytes.decode("utf-8")
                assert f"]({images[0]})" in markdown
                assert "data:image" not in markdown
                assert hashlib.sha256(markdown_bytes).hexdigest() == bundled_map["markdown_sha256"]
                assert bundled_map["markdown_sha256"] != mapping["markdown_sha256"]
                assert slices(markdown, bundled_map)[-2:] == ["😀", "café"]
                if pure_markdown:
                    assert not re.search(r"</?[A-Za-z]", markdown)
        assert result.middle_json.to_dict(skip_defaults=False) == original


def test_pure_map_translates_inline_markup_and_unicode() -> None:
    result = make_result(
        [
            {"type": "text", "content": "😀A"},
            {"type": "text", "content": "1", "styles": ["superscript"]},
            {"type": "text", "content": " "},
            {"type": "text", "content": "café", "styles": ["underline", "bold"]},
        ]
    )
    before = result.middle_json.to_dict(skip_defaults=False)
    markdown, mapping = result.markdown_with_map(pure_markdown=True)
    assert markdown == result.markdown(pure_markdown=True)
    assert markdown == "😀A^1 **café**"
    assert slices(markdown, mapping) == ["😀A^1", "**café**"]
    assert mapping["markdown_sha256"] == hashlib.sha256(markdown.encode()).hexdigest()
    assert result.middle_json.to_dict(skip_defaults=False) == before


def test_pure_map_flattens_complex_table_conservatively() -> None:
    result = make_result([{"type": "text", "content": "After table"}])
    payload = result.middle_json.to_dict(skip_defaults=False)
    payload["pages"][0]["blocks"].insert(
        0,
        {
            "type": "table",
            "index": 0,
            "bbox": [0.1, 0.3, 0.8, 0.5],
            "content": [
                {
                    "type": "table_body",
                    "index": 0,
                    "content": ('<table><tr><th colspan="2">Title</th></tr><tr><td>A</td><td>B</td></tr></table>'),
                }
            ],
        },
    )
    result = ParseResult.from_dict(payload)
    markdown, mapping = result.markdown_with_map(pure_markdown=True)
    assert markdown == result.markdown(pure_markdown=True)
    assert "<table" not in markdown
    assert "| Title |" in markdown
    block = mapping["pages"][0]["blocks"][0]
    assert block["level"] == "block"
    assert block["lines"] == []
    # A flattened table owns only its whole block range; no cell positions are invented.
    assert [(e["level"], e["markdown_char_start"], e["markdown_char_end"]) for e in block["elements"]] == [
        ("block", block["markdown_char_start"], block["markdown_char_end"])
    ]
    assert "| Title |" in markdown[block["markdown_char_start"] : block["markdown_char_end"]]
    assert slices(markdown, mapping)[-2:] == ["After", "table"]


@pytest.mark.parametrize("output_format", ["markdown", "zip"])
def test_pure_markdown_job_without_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_format: str) -> None:
    result = make_result([{"type": "text", "content": "Plain café", "styles": ["underline"]}])

    async def parse(path: str, **kwargs: Any) -> ParseResult:
        assert "positions" not in kwargs
        return result

    monkeypatch.setattr(api_server, "parse_async", parse)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    request = api_server.CreateJobRequest(
        files=[{"source": {"type": "local", "path": str(source)}}],
        tier="flash",
        output_formats=[output_format],
        pure_markdown=True,
    )
    file_store = api_server.FileStore(tmp_path / "store")
    record = api_server.JobStore().create(request, file_store)
    asyncio.run(api_server._run_job(record, request, file_store, image_analysis=False, allow_local_source=True))
    assert record.status == "completed"
    outputs = record.files[0].output_files
    assert outputs.markdown_map is None
    if output_format == "markdown":
        assert file_store.read_file_data(outputs.markdown.file_id).decode() == "Plain café"
    else:
        with zipfile.ZipFile(io.BytesIO(file_store.read_file_data(outputs.zip.file_id))) as archive:
            # Requesting only zip keeps the self-contained bundle.
            assert archive.read("markdown.md").decode() == "Plain café"
            assert sorted(archive.namelist()) == ["markdown.md", "middle_json.json", "structured_content.json"]


@pytest.mark.parametrize(
    ("output_formats", "expected"),
    [
        (["zip", "markdown"], ["markdown.md"]),
        (["zip", "middle_json", "structured_content"], ["middle_json.json", "structured_content.json"]),
        (["zip", "markdown", "markdown_map"], ["markdown.md", "markdown_map.json"]),
    ],
)
def test_zip_bundles_only_requested_formats(output_formats: list[str], expected: list[str]) -> None:
    result = make_result([{"type": "text", "content": "Plain café"}])
    files = api_server._zip_result_files(output_formats)
    with zipfile.ZipFile(io.BytesIO(api_server._build_self_contained_zip_output(result, files=files))) as archive:
        assert sorted(archive.namelist()) == expected


def test_result_save_rejects_unknown_files() -> None:
    with pytest.raises(ValueError, match="Unsupported result files"):
        make_result([{"type": "text", "content": "x"}]).save(api_server._ZipDataWriter(None), files=["html"])


def test_flat_reference_list_preserves_leaf_word_ranges() -> None:
    result = make_result([{"type": "text", "content": "[1] Author"}])
    payload = result.to_dict()
    child = payload["pages"][0]["blocks"][0]
    child["type"] = "ref_text"
    source = payload["extensions"]["mineru_position_sources"]["blocks"][0]
    source["content"] = deepcopy(child["content"])
    payload["pages"][0]["blocks"] = [
        {"type": "list", "index": 0, "bbox": child["bbox"], "sub_type": "ref_text", "content": [child]}
    ]
    result = ParseResult.from_dict(payload)
    markdown, mapping = result.markdown_with_map()
    assert markdown == result.markdown()
    assert slices(markdown, mapping) == ["[1]", "Author"]
    assert all(b["level"] == "word" for b in regions(mapping))


def test_hierarchy_uses_only_producer_lines() -> None:
    result = make_result([{"type": "text", "content": "same same"}])
    source = result.middle_json.extensions["mineru_position_sources"]["blocks"][0]
    source["lines"] = [{"line_id": 0, "bbox": [0.1, 0.1, 0.8, 0.2]}, {"line_id": 3, "bbox": [0.1, 0.3, 0.8, 0.4]}]
    for token, line_id in zip(source["tokens"], [0, 3]):
        token["source_span"] = {"line": line_id, "span": 0, "text": "same"}
    markdown, mapping = result.markdown_with_map()
    assert mapping["version"] == 2
    block = mapping["pages"][0]["blocks"][0]
    assert block["type"] == "text"
    assert [line["line_id"] for line in block["lines"]] == [0, 3]
    assert [line["bbox"] for line in block["lines"]] == [line["bbox"] for line in source["lines"]]
    assert slices(markdown, mapping) == ["same", "same"]
    assert block["elements"] == []


def test_cross_line_word_has_shared_range_and_local_boxes() -> None:
    result = make_result([{"type": "text", "content": "microkinetics"}])
    source = result.middle_json.extensions["mineru_position_sources"]["blocks"][0]
    boxes = [[0.1, 0.1, 0.4, 0.2], [0.1, 0.3, 0.5, 0.4]]
    source["lines"] = [{"line_id": i, "bbox": box} for i, box in enumerate(boxes)]
    parts = [{"source_span": {"line": i, "span": 0}, "start": 0, "end": 5 + i * 3, "bbox": box} for i, box in enumerate(boxes)]
    source["tokens"][0].update(source_parts=parts, bboxes=boxes)
    markdown, mapping = result.markdown_with_map()
    children = regions(mapping)
    assert slices(markdown, mapping) == ["microkinetics"] * 2
    assert children[0]["token_id"] == children[1]["token_id"]
    assert [child["bbox"] for child in children] == boxes
    assert all(child["source_parts"] == parts for child in children)


def test_no_line_provenance_does_not_invent_lines() -> None:
    _, mapping = make_result([{"type": "text", "content": "Plain text"}]).markdown_with_map()
    block = mapping["pages"][0]["blocks"][0]
    assert block["lines"] == []
    assert [element["level"] for element in block["elements"]] == ["word", "word"]


def test_producer_line_provenance_reaches_map() -> None:
    from mineru.backend.analysis.pdf.text.models import _AnalyzeLine, _AnalyzeSpan
    from mineru.backend.analysis.pdf.text.positions import capture_block_positions, capture_words

    result = make_result([{"type": "text", "content": "microkinetics"}])
    first = _AnalyzeSpan("text", (10, 10, 70, 20), "micro-")
    second = _AnalyzeSpan("text", (10, 30, 90, 40), "kinetics")
    for span in (first, second):
        x, y, _, _ = span.bbox
        capture_words(span, [(char, (x + i * 10, y, x + (i + 1) * 10, y + 10)) for i, char in enumerate(span.content)])
    raw = {"type": "text", "content": "microkinetics"}
    capture_block_positions(raw, [_AnalyzeLine(first.bbox, [first]), _AnalyzeLine(second.bbox, [second])], (100, 100))
    source = result.middle_json.extensions["mineru_position_sources"]["blocks"][0]
    source.update(tokens=raw["_position_tokens"], lines=raw["_position_lines"])
    before = result.to_dict()
    markdown, mapping = result.markdown_with_map()
    assert result.to_dict() == before
    assert markdown == result.markdown()
    lines = mapping["pages"][0]["blocks"][0]["lines"]
    assert [line["bbox"] for line in lines] == [[0.1, 0.1, 0.7, 0.2], [0.1, 0.3, 0.9, 0.4]]
    assert slices(markdown, mapping) == ["microkinetics"] * 2
    assert lines[0]["elements"][0]["bbox"] == [0.1, 0.1, 0.6, 0.2]


def test_formula_with_line_owner_is_atomic() -> None:
    result = make_result([{"type": "equation_inline", "content": "x_1 + y_2"}])
    source = result.middle_json.extensions["mineru_position_sources"]["blocks"][0]
    source["lines"] = [{"line_id": 0, "bbox": source["bbox"]}]
    source["tokens"] = [
        {
            "type": "inline_math",
            "latex": "x_1 + y_2",
            "level": "formula",
            "bbox": source["bbox"],
            "source_span": {"line": 0, "span": 0},
        }
    ]
    markdown, mapping = result.markdown_with_map()
    assert len(regions(mapping)) == 1
    assert regions(mapping)[0]["level"] == "formula"
    assert slices(markdown, mapping) == [markdown]


def test_display_formula_stays_a_separate_atomic_block() -> None:
    result = make_result([{"type": "text", "content": "unused"}])
    payload = result.to_dict()
    payload["pages"][0]["blocks"][0].update(type="equation", content="x^2 + y^2")
    payload["extensions"].pop("mineru_position_sources")
    result = ParseResult.from_dict(payload)
    markdown, mapping = result.markdown_with_map()
    block = mapping["pages"][0]["blocks"][0]
    assert block["type"] == "equation" and block["level"] == "formula"
    assert block["lines"] == []
    assert slices(markdown, mapping) == [markdown]


def test_unknown_line_identity_remains_unassigned() -> None:
    result = make_result([{"type": "text", "content": "word"}])
    source = result.middle_json.extensions["mineru_position_sources"]["blocks"][0]
    source["lines"] = [{"line_id": 0, "bbox": source["bbox"]}]
    source["tokens"][0]["source_span"] = {"line": 99, "span": 0}
    _, mapping = result.markdown_with_map()
    block = mapping["pages"][0]["blocks"][0]
    assert block["lines"] == []
    assert len(block["elements"]) == 1


def continued_result() -> ParseResult:
    """Physical source blocks repeat words and local line IDs across pages."""
    result = make_result([{"type": "text", "content": "same same"}])
    first = result.to_dict()
    second = deepcopy(first["pages"][0])
    second["page_idx"] = 9
    second["blocks"][0]["continues_prev"] = True
    image = {
        "type": "image",
        "index": 0,
        "bbox": [0.1, 0.3, 0.9, 0.6],
        "content": [{"type": "image_body", "index": 0, "content": "", "image_path": "figure.png"}],
    }
    second["blocks"].insert(0, image)
    first["pages"].append(second)
    sources = first["extensions"]["mineru_position_sources"]["blocks"]
    sources.append(deepcopy(sources[0]))
    sources[1]["page_idx"] = 9
    for source in sources:
        source["lines"] = [{"line_id": 0, "bbox": source["bbox"]}]
        for token in source["tokens"]:
            token["source_span"] = {"line": 0, "span": 0}
    return ParseResult.from_dict(first)


def test_continued_paragraph_keeps_physical_lines_across_intervening_image() -> None:
    result = continued_result()
    before = result.to_dict()
    markdown, mapping = result.markdown_with_map()
    assert markdown == result.markdown()
    assert result.to_dict() == before
    assert mapping["markdown_sha256"] == hashlib.sha256(markdown.encode()).hexdigest()
    assert [page["page_idx"] for page in mapping["pages"]] == [5, 9]
    first, second = mapping["pages"]
    assert [block["type"] for block in second["blocks"]] == ["image", "text"]
    assert len(first["blocks"][0]["lines"]) == 1
    text = next(block for block in second["blocks"] if block["type"] == "text")
    assert len(text["lines"]) == 1
    words = [region for region in regions(mapping) if region["level"] == "word"]
    assert [markdown[word["markdown_char_start"] : word["markdown_char_end"]] for word in words] == ["same"] * 4
    assert [word["markdown_char_start"] for word in words] == [0, 5, 10, 15]
    assert words[0]["token_id"] != words[2]["token_id"]
    image = next(block for block in second["blocks"] if block["type"] == "image")
    fragment = markdown[image["markdown_char_start"] : image["markdown_char_end"]]
    assert fragment == "![](figure.png)"
    assert image["level"] == "block" and image["lines"] == []
    assert all("group_id" not in block for page in mapping["pages"] for block in page["blocks"])


def test_continuation_does_not_invent_missing_source_words() -> None:
    result = continued_result()
    result.middle_json.extensions["mineru_position_sources"]["blocks"].pop()
    markdown, mapping = result.markdown_with_map()
    second = next(block for block in mapping["pages"][1]["blocks"] if block["type"] == "text")
    assert second["lines"] == []
    assert second["elements"][0]["level"] == "block"
    assert markdown[second["markdown_char_start"] : second["markdown_char_end"]] == "same same"


@pytest.mark.parametrize("kind", ["image", "table", "chart"])
@pytest.mark.parametrize("with_source", [False, True])
def test_captions_are_nested_and_keep_their_own_ranges(kind: str, with_source: bool) -> None:
    result = make_result([{"type": "text", "content": "😀 same same"}])
    payload = result.to_dict()
    source = payload["extensions"]["mineru_position_sources"]["blocks"][0]
    source.update(index=2, lines=[{"line_id": 0, "bbox": source["bbox"]}])
    for token in source["tokens"]:
        token["source_span"] = {"line": 0, "span": 0}
    # Identical body and caption text must not result in a source-text match.
    body = {"type": f"{kind}_body", "index": 0, "content": "😀 same same"}
    caption = {"type": f"{kind}_caption", "index": 2, "bbox": source["bbox"], "content": source["content"]}
    payload["pages"][0]["blocks"] = [{"type": kind, "index": 0, "bbox": [0.1, 0.3, 0.8, 0.6], "content": [body, caption]}]
    if not with_source:
        payload["extensions"].pop("mineru_position_sources")
    result = ParseResult.from_dict(payload)
    before = result.to_dict()
    markdown, mapping = result.markdown_with_map()
    assert markdown == result.markdown()
    assert result.to_dict() == before
    assert mapping["markdown_sha256"] == hashlib.sha256(markdown.encode()).hexdigest()
    parent = mapping["pages"][0]["blocks"][0]
    child = parent["children"][0]
    assert child["element"] == "/pages/0/blocks/0/content/1"
    assert child["type"] == f"{kind}_caption"
    assert child["bbox"] == source["bbox"]
    assert markdown[child["markdown_char_start"] : child["markdown_char_end"]] == "😀 same same"
    assert child["markdown_char_start"] > parent["markdown_char_start"]
    if with_source:
        assert len(child["lines"]) == 1
        assert [markdown[e["markdown_char_start"] : e["markdown_char_end"]] for e in child["lines"][0]["elements"]] == [
            "😀",
            "same",
            "same",
        ]
        assert child["elements"] == []
    else:
        assert child["lines"] == []
        assert child["elements"][0]["level"] == "block"


def test_caption_survives_intervening_image_in_continued_paragraph() -> None:
    payload = continued_result().to_dict()
    payload["pages"][1]["blocks"][0]["content"].append(
        {
            "type": "image_caption",
            "index": 2,
            "bbox": [0.1, 0.61, 0.9, 0.65],
            "content": [{"type": "text", "content": "same same"}],
        }
    )
    result = ParseResult.from_dict(payload)
    markdown, mapping = result.markdown_with_map()
    image = mapping["pages"][1]["blocks"][0]
    caption = image["children"][0]
    assert markdown[caption["markdown_char_start"] : caption["markdown_char_end"]] == "same same"
    assert caption["level"] == "block"
    assert len(mapping["pages"][1]["blocks"][1]["lines"]) == 1


def test_caption_renderer_visits_child_lines() -> None:
    from PIL import Image
    from scripts.render_markdown_map import render_page

    page = {
        "blocks": [
            {
                "type": "image",
                "lines": [],
                "elements": [],
                "children": [
                    {
                        "type": "image_caption",
                        "lines": [
                            {
                                "line_id": 0,
                                "bbox": [0.1, 0.7, 0.8, 0.8],
                                "elements": [{"bbox": [0.1, 0.7, 0.3, 0.8], "level": "word"}],
                            }
                        ],
                        "elements": [],
                        "children": [],
                    }
                ],
            }
        ]
    }
    with Image.new("RGB", (600, 800), "white") as image:
        rendered, counts = render_page(image, page)
        try:
            assert counts["captions"] == 1 and counts["lines"] == 1 and counts["word"] == 1
            assert rendered.getpixel((60, 560)) != (255, 255, 255)
        finally:
            rendered.close()


def test_caption_shared_group_fallback_does_not_invent_lines() -> None:
    payload = continued_result().to_dict()
    payload["pages"][0]["blocks"][0]["content"][0]["content"] = "micro-"
    payload["pages"][1]["blocks"][1]["content"][0]["content"] = "kinetics"
    payload["pages"][1]["blocks"][0]["content"].insert(
        0,
        {
            "type": "image_caption",
            "index": 2,
            "bbox": [0.1, 0.61, 0.9, 0.65],
            "content": [{"type": "text", "content": "Caption before image"}],
        },
    )
    result = ParseResult.from_dict(payload)
    _, mapping = result.markdown_with_map()
    image = mapping["pages"][1]["blocks"][0]
    caption = image["children"][0]
    assert caption["type"] == "image_caption"
    assert caption["level"] == "group" and caption["lines"] == []
    assert caption["markdown_char_start"] == image["markdown_char_start"]
    assert caption["markdown_char_end"] == image["markdown_char_end"]


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("tight_spacing", [False, True])
def test_word_provenance_tracks_generated_cjk_space_removal(generated: bool, tight_spacing: bool) -> None:
    """Keep glyph ownership aligned with upstream's generated-space policy."""
    from docvortex.document.pdf import Bbox
    from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
    from mineru.backend.analysis.pdf.text.native import chars_to_content

    source = "\u4e2d \u6587"
    chars = [
        {
            "char": char,
            "char_idx": index,
            "bbox": Bbox([index * 5, 0, index * 5 + 10, 10]),
            "tight_bbox": (index * 5, 0, index * 5 + 10, 10),
            "origin": (index * 5, 10),
            "writing_angle": 0.0,
            "is_generated": generated and char == " ",
        }
        for index, char in enumerate(source)
    ]
    span = _AnalyzeSpan(
        "text",
        (0, 0, 30, 10),
        metadata={"chars": chars, "_collect_positions": True, "_native_tight_spacing": tight_spacing},
    )
    chars_to_content(span, detect_scripts=False)
    expected = source.replace(" ", "") if generated and tight_spacing else source
    assert span.content == expected
    assert span.metadata["_position_words"]["text"] == expected
    assert [word["text"] for word in span.metadata["_position_words"]["words"]] == expected.split()
    assert len(span.metadata["_position_char_boxes"]) == len(expected)


def test_probe_documents_do_not_copy_document_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.parser import markdown_map

    result = make_result([{"type": "text", "content": "Probe documents stay small."}])
    middle = result.middle_json
    middle.extensions["mineru_text_selection"] = {"version": 1, "text_source": "pdf", "blocks": [{"index": 1}] * 1000}
    expected = result.markdown_with_map()
    rendered = []
    render = markdown_map.render_markdown

    def spy(document: MiddleJson, **options: Any) -> str:
        rendered.append(document)
        return render(document, **options)

    monkeypatch.setattr(markdown_map, "render_markdown", spy)
    assert result.markdown_with_map() == expected
    assert rendered[0] is middle
    assert len(rendered) > 1
    assert all(document.extensions == {} for document in rendered[1:])
