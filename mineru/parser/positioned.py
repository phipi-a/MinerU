"""Conservative positional export without text search or geometric rematching."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..types import MiddleJson, ModelJson, parse_inline_spans

__all__ = ["positioned_document"]

SOURCE_EXTENSION = "mineru_position_sources"
OUTPUT_EXTENSION = "mineru_positions"


def positioned_document(middle: MiddleJson, model: ModelJson | None) -> dict[str, Any]:
    """Add JSON-pointer-linked positions; changed content falls back to its own block.

    Raw block indices are producer-assigned identities, not nearest-text matches.
    Equality only checks that postprocessing preserved the referenced content.
    No source geometry is attached to rewritten content at a finer granularity.
    """
    result = middle.to_dict(
        skip_defaults=True,
        exclude_block_fields={"image_base64"} if middle.metadata.file_suffix == "pdf" else None,
    )
    if OUTPUT_EXTENSION in result["extensions"] and SOURCE_EXTENSION not in result["extensions"] and model is None:
        return result
    sources = middle.extensions.get(SOURCE_EXTENSION, {})
    if model is not None:
        sources = model.extensions.get(SOURCE_EXTENSION, sources)
    records = sources.get("blocks", []) if isinstance(sources, dict) else []
    by_identity = {(r["page_idx"], r["index"]): r for r in records}
    entries: list[dict[str, Any]] = []
    for page_offset, page in enumerate(result["pages"]):
        # Default-skipping serialization omits the blocks array on empty pages.
        for block_offset, block in enumerate(page.get("blocks", [])):
            _visit_block(block, page["page_idx"], f"/pages/{page_offset}/blocks/{block_offset}", by_identity, entries)
    result["extensions"].pop(SOURCE_EXTENSION, None)
    result["extensions"][OUTPUT_EXTENSION] = {
        "version": 1,
        "coordinate_system": "top_left_xyxy",
        "bbox_units": "normalized",
        "offset_units": "unicode_codepoints",
        "entries": entries,
    }
    return result


def _visit_block(
    block: dict[str, Any],
    page_idx: int,
    pointer: str,
    sources: dict[tuple[int, int], dict[str, Any]],
    entries: list[dict[str, Any]],
) -> None:
    content = block.get("content")
    if isinstance(content, list) and any(isinstance(c, dict) and "index" in c for c in content):
        for offset, child in enumerate(content):
            _visit_block(child, page_idx, f"{pointer}/content/{offset}", sources, entries)
        return
    source = sources.get((page_idx, block.get("index")))
    # An identity may survive a merge or be reused by a visual container: verify
    # its payload and box before using the producer's fine-grained positions.
    source_content = source["content"] if source else None
    if isinstance(source_content, list) and all(
        c.get("type") in {"text", "equation_inline", "code_inline", "hyperlink"} for c in source_content
    ):
        source_content = [span.model_dump(mode="json", exclude_defaults=True) for span in parse_inline_spans(source_content)]
    unchanged = source_content == content
    if isinstance(source_content, list) and isinstance(content, list):
        from ..backend.analysis.pdf.text.positions import content_signature

        unchanged = content_signature(source_content) == content_signature(content)
    if source and unchanged and source["bbox"] == block.get("bbox"):
        tokens = deepcopy(source["tokens"])
        level = "mixed"
        origin = "analysis"
    else:
        level = "block" if block.get("bbox") is not None else "unavailable"
        if block.get("type") == "equation" and block.get("bbox") is not None:
            level = "formula"
            tokens = [{"type": "display_math", "latex": content, "bbox": block["bbox"], "level": level}]
        else:
            tokens = [{"content": deepcopy(content), "bbox": block.get("bbox"), "level": level}]
        origin = "middle_json"
    entries.append(
        {
            "element": pointer,
            "page_idx": page_idx,
            "level": level,
            "source": origin,
            "tokens": tokens,
            "lines": deepcopy(source.get("lines", [])) if origin == "analysis" else [],
        }
    )
