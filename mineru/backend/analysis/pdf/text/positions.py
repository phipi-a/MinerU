"""Carry producer geometry through text construction, never align it afterwards."""

from __future__ import annotations

import math
import re
from typing import Any

from docvortex.content.text import resolve_text_line_boundary

from .models import _AnalyzeLine, _AnalyzeSpan

__all__ = ["capture_words", "capture_block_positions", "content_signature"]


def content_signature(content: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Visible semantic stream; style splitting/merging does not alter ownership."""
    result: list[tuple[str, str]] = []
    for span in content:
        kind = span["type"]
        value = span["content"]
        if kind == "hyperlink":
            pieces = content_signature(value)
        else:
            pieces = [(kind, value)]
        for kind, text in pieces:
            if result and kind == "text" and result[-1][0] == kind:
                result[-1] = (kind, result[-1][1] + text)
            else:
                result.append((kind, text))
    return result


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def capture_words(span: _AnalyzeSpan, records: list[tuple[str, Any]]) -> None:
    """Apply the same character cleanup as the producer, retaining glyph owners."""
    from .native import __replace_ligatures, __replace_unicode

    normalized = [(__replace_ligatures(__replace_unicode(text)), box) for text, box in records]
    text = "".join(part for part, _ in normalized)
    if text.strip() != span.content:
        return
    leading = len(text) - len(text.lstrip())
    characters = [box for part, box in normalized for _ in part][leading : leading + len(span.content)]
    words = []
    for match in re.finditer(r"\S+", span.content):
        boxes = characters[match.start() : match.end()]
        complete = bool(boxes) and all(_valid_box(box) for box in boxes)
        words.append(
            {
                "type": "word",
                "text": match.group(),
                "level": "word" if complete else "span",
                "start": match.start(),
                "end": match.end(),
                "bbox": _union(boxes) if complete else list(span.bbox),
            }
        )
    span.metadata["_position_words"] = {"text": span.content, "words": words}
    # Private, same-producer glyph map permits removal of a line-end hyphen and
    # splitting ligatures without retaining its removed ink in the output box.
    span.metadata["_position_char_boxes"] = characters


def _union(boxes: list[Any]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _unit(box: Any, size: tuple[float, float]) -> list[float] | None:
    if not _valid_box(box):
        return None
    result = [box[0] / size[0], box[1] / size[1], box[2] / size[0], box[3] / size[1]]
    return result if all(0 <= x <= 1 for x in result) else None


def _line_stream(
    line: _AnalyzeLine,
    line_index: int,
    size: tuple[float, float],
) -> tuple[str, list[dict[str, Any] | None]]:
    text = ""
    owners: list[dict[str, Any] | None] = []
    for span_index, span in enumerate(line.spans):
        value = str(span.content or "").strip()
        if not value or span.type not in {"text", "inline_equation"}:
            continue
        if text:
            text += " "
            owners.append(None)
        source = {"line": line_index, "span": span_index, "text": span.content}
        box = _unit(span.bbox, size)
        if box is None:
            return "", []
        identity = (line_index, span_index)
        if span.type == "inline_equation":
            rendered = f"\\({value}\\)"
            owner = {
                "type": "inline_math",
                "latex": value,
                "level": "formula",
                "bbox": box,
                "source_span": source,
                "identity": identity,
            }
            owners.extend([owner] * len(rendered))
            text += rendered
            continue
        glyphs = span.metadata.get("_position_char_boxes")
        words = span.metadata.get("_position_words")
        fine = words is not None and words["text"] == span.content and glyphs is not None
        leading = len(span.content) - len(span.content.lstrip())
        for offset, char in enumerate(value, leading):
            glyph_box = _unit(glyphs[offset], size) if fine else None
            owners.append(
                {
                    "type": "text",
                    "level": "word" if glyph_box is not None else "line" if len(line.spans) == 1 else "span",
                    "bbox": glyph_box or box,
                    "source_span": source,
                    "offset": offset,
                    "identity": identity,
                }
            )
        text += value
    return text, owners


def _tokenize(text: str, owners: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """Segment the constructed stream; source character owners stay attached."""
    result = []
    offset = 0
    while offset < len(text):
        owner = owners[offset]
        if owner is None or text[offset].isspace():
            offset += 1
            continue
        end = offset + 1
        if owner["type"] == "inline_math":
            while end < len(text) and owners[end] is owner:
                end += 1
            result.append({k: v for k, v in owner.items() if k != "identity"})
        else:
            while end < len(text) and not text[end].isspace() and owners[end] is not None and owners[end]["type"] == "text":
                end += 1
            members = owners[offset:end]
            parts = []
            cursor = 0
            while cursor < len(members):
                first = members[cursor]
                finish = cursor + 1
                while finish < len(members) and members[finish]["identity"] == first["identity"]:
                    finish += 1
                group = members[cursor:finish]
                parts.append(
                    {
                        "source_span": first["source_span"],
                        "start": first["offset"],
                        "end": group[-1]["offset"] + 1,
                        "bbox": _union([m["bbox"] for m in group]),
                    }
                )
                cursor = finish
            boxes = [p["bbox"] for p in parts]
            level = (
                "line"
                if any(m["level"] == "line" for m in members)
                else "span"
                if any(m["level"] == "span" for m in members)
                else "word"
            )
            token = {
                "type": "word" if level == "word" else "text",
                "text": text[offset:end],
                "level": level,
                "bbox": _union(boxes),
            }
            if len(parts) == 1:
                token.update({k: v for k, v in parts[0].items() if k != "bbox"})
            else:
                token.update({"bboxes": boxes, "source_parts": parts})
            # An OCR-only line stays a line token rather than receiving repeated
            # pretend word boxes. Never split a lexical word at a source boundary.
            previous = result[-1] if result else None
            if (
                level != "word"
                and previous is not None
                and previous.get("level") == level
                and len(parts) == 1
                and previous.get("source_span") == token.get("source_span")
                and previous.get("bbox") == token["bbox"]
                and "bboxes" not in previous
            ):
                previous["text"] += text[previous["_stream_end"] : offset] + token["text"]
                previous["end"] = token["end"]
                previous["_stream_end"] = end
            else:
                token["_stream_end"] = end
                result.append(token)
        offset = end
    for token in result:
        token.pop("_stream_end", None)
    return result


def capture_block_positions(block: dict[str, Any], lines: list[_AnalyzeLine], size: tuple[float, float]) -> None:
    """Use the producer's public boundary operation while carrying its source map."""
    text = ""
    owners: list[dict[str, Any] | None] = []
    for line_index, line in enumerate(lines):
        current, current_owners = _line_stream(line, line_index, size)
        if not current:
            continue
        if text:
            if block.get("type") in {"index", "code", "algorithm"}:
                processed, separator = text, "\n"
            else:
                processed, separator = resolve_text_line_boundary(text, next_content=current)
            # The shared operation deletes only a trailing boundary here. Any
            # other rewrite has no supported provenance operation: fail closed.
            if not text.startswith(processed):
                return
            owners = owners[: len(processed)] + [None] * len(separator)
            text = processed + separator
        text += current
        owners.extend(current_owners)
    if text == block.get("content") and text:
        block["_position_lines"] = [{"line_id": i, "bbox": _unit(line.bbox, size)} for i, line in enumerate(lines)]
        block["_position_tokens"] = _tokenize(text, owners)
        block["_position_content"] = text
