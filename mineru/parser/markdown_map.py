"""Markdown source ranges composed from public DocVortex rendering operations.

Never locate a word in a completed Markdown file. Compose ordered render groups,
verify their exact serialization, and refine only reversible inline emissions.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

from docvortex.render.fragments import render_inline_content
from docvortex.render.markdown import render_single_block

from ..config import config
from ..render.markdown import render_markdown
from ..types import MiddleJson, PAGE_AUXILIARY_BLOCK_TYPES, TextSpan

__all__ = ["render_markdown_map"]

_PROBE = "MINERUMARKDOWNSOURCEMAPPROBE0123456789"


def _document(middle: MiddleJson, blocks: list[tuple[int, int, Any]]) -> MiddleJson:
    pages = []
    for page_offset, page in enumerate(middle.pages):
        owned = [block for p, _, block in blocks if p == page_offset]
        if owned:
            pages.append(page.model_copy(update={"blocks": owned}))
    extensions = {k: v for k, v in middle.extensions.items() if k not in {"mineru_position_sources", "mineru_positions"}}
    return middle.model_copy(update={"pages": pages, "extensions": extensions})


def _groups(middle: MiddleJson) -> list[list[tuple[int, int, Any]]]:
    """Keep potential continuation dependencies together, without implementing merges.

    Grouping is deliberately broader than the renderer's planner. The shared
    renderer alone decides whether/how continuations actually merge.
    """
    blocks = [(p, b, block) for p, page in enumerate(middle.pages) for b, block in enumerate(page.blocks)]
    boundaries = set(range(1, len(blocks)))
    previous: dict[str, int] = {}
    for offset, (_, _, block) in enumerate(blocks):
        kind = str(block.type)
        if getattr(block, "continues_prev", False) and kind in previous:
            boundaries.difference_update(range(previous[kind] + 1, offset + 1))
        previous[kind] = offset
    groups: list[list[tuple[int, int, Any]]] = []
    for offset, block in enumerate(blocks):
        if offset == 0 or offset in boundaries:
            groups.append([])
        groups[-1].append(block)
    return groups


def _template(rendered: str) -> tuple[str, str] | None:
    # This is a controlled rendering probe, NOT a search for source text.
    if rendered.count(_PROBE) != 1:
        return None
    prefix, _, suffix = rendered.partition(_PROBE)
    return prefix, suffix


def _text_boundaries(span: TextSpan, cuts: set[int], *, pure: bool) -> tuple[str, dict[int, int], dict[int, int]] | None:
    """Locate owned character boundaries through exact compositional render checks."""
    delimiters = config.render.latex_delimiters

    def emit(spans: list[Any]) -> str:
        return render_inline_content(spans, delimiters, pure_markdown=pure)

    rendered = emit([span])
    plain = span.model_copy(update={"styles": []})
    raw = emit([plain])
    template = _template(emit([span.model_copy(update={"content": _PROBE})]))
    if template is None:
        return None
    opening, closing = template
    if rendered == raw:
        opening = closing = ""
    leading = 0 if not opening else len(span.content) - len(span.content.lstrip(" \t"))
    core_end = len(span.content) if not opening else len(span.content.rstrip(" \t"))
    prefix = emit([plain.model_copy(update={"content": span.content[:leading]})])
    core = emit([plain.model_copy(update={"content": span.content[leading:core_end]})])
    suffix = emit([plain.model_copy(update={"content": span.content[core_end:]})])
    if prefix + core + suffix != raw or prefix + opening + core + closing + suffix != rendered:
        return None  # whitespace rewriting, context-sensitive escaping, or unsupported styles
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    for cut in cuts | {0, len(span.content)}:
        left = emit([plain.model_copy(update={"content": span.content[:cut]})])
        right = emit([plain.model_copy(update={"content": span.content[cut:]})])
        if left + right != raw:
            return None  # no guess about an escape that crosses this source boundary
        position = len(left)
        if cut > leading:
            position += len(opening)
        if cut > core_end:
            position += len(closing)
        starts[cut] = position
        ends[cut] = position + (len(closing) if cut == core_end else 0)
    return rendered, starts, ends


def _fine_ranges(
    block: Any,
    tokens: list[dict[str, Any]],
    rendered: str,
    *,
    pure: bool,
    inline_only: bool = False,
) -> list[dict[str, Any]] | None:
    content = block.content
    if (
        not isinstance(content, list)
        or not content
        or any(getattr(span, "type", None) not in {"text", "equation_inline"} for span in content)
    ):
        return None
    delimiters = config.render.latex_delimiters
    logical = "".join(span.content if span.type == "text" else f"\\({span.content}\\)" for span in content)
    # Walk the producer's ordered stream at its current cursor, never search for
    # the next occurrence. A changed stream makes the entire refinement fail.
    cursor = 0
    intervals = []
    for token in tokens:
        while cursor < len(logical) and logical[cursor].isspace():
            cursor += 1
        value = f"\\({token['latex']}\\)" if token.get("type") == "inline_math" else token.get("text")
        if not isinstance(value, str) or not value or logical[cursor : cursor + len(value)] != value:
            return None
        intervals.append((cursor, cursor + len(value), token))
        cursor += len(value)
    if logical[cursor:].strip():
        return None
    if inline_only:
        template = ("", "")
    else:
        probe = block.model_copy(update={"content": [TextSpan(type="text", content=_PROBE)]})
        try:
            template = _template(
                render_single_block(probe, delimiters=delimiters, asset_base_url="", pure_markdown=pure).strip("\n")
            )
        except TypeError:
            return None
    if template is None:
        return None
    opening, closing = template
    inline = render_inline_content(content, delimiters, pure_markdown=pure)
    if opening + inline + closing != rendered:
        return None
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    logical_cursor = 0
    emitted_cursor = len(opening)
    cuts = {position for start, end, _ in intervals for position in (start, end)}
    pieces = []
    for span in content:
        length = len(span.content) if span.type == "text" else len(f"\\({span.content}\\)")
        local_cuts = {cut - logical_cursor for cut in cuts if logical_cursor <= cut <= logical_cursor + length}
        if span.type == "text":
            traced = _text_boundaries(span, local_cuts, pure=pure)
            if traced is None:
                return None
            piece, local_starts, local_ends = traced
        else:
            piece = render_inline_content([span], delimiters, pure_markdown=pure)
            if local_cuts - {0, length}:
                return None  # formulas are atomic
            local_starts = local_ends = {0: 0, length: len(piece)}
        for cut, offset in local_starts.items():
            starts[logical_cursor + cut] = emitted_cursor + offset
        for cut, offset in local_ends.items():
            ends.setdefault(logical_cursor + cut, emitted_cursor + offset)
        pieces.append(piece)
        logical_cursor += length
        emitted_cursor += len(piece)
    if "".join(pieces) != inline:
        return None
    ranges = []
    for start, end, token in intervals:
        if token.get("bbox") is None:
            continue
        item = {
            "bbox": token["bbox"],
            "level": token["level"],
            "markdown_char_start": starts[start],
            "markdown_char_end": ends[end],
        }
        for key in ("type", "bboxes", "source_span", "source_parts"):
            if key in token:
                item[key] = deepcopy(token[key])
        # Keep producer identities and offsets, not redundant source text copies.
        for source in ([item["source_span"]] if "source_span" in item else []) + [
            part["source_span"] for part in item.get("source_parts", [])
        ]:
            source.pop("text", None)
        ranges.append(item)
    return ranges


def _list_ranges(
    visible: list[tuple[int, int, Any]],
    text: str,
    entries: dict[str, Any],
    *,
    pure: bool,
) -> list[tuple[int, dict[str, Any]]] | None:
    """Refine flat list emissions, including verified continued reference lists.

    Do not reproduce list-marker or nesting rules. Accept only if the public
    inline emissions compose exactly to the shared renderer's complete output.
    """
    if not visible or any(block.type != "list" for _, _, block in visible):
        return None
    leaves = []
    for p, b, block in visible:
        for c, child in enumerate(block.content):
            if child.type == "list":
                return None
            piece = render_inline_content(child.content, config.render.latex_delimiters, pure_markdown=pure)
            if not piece:
                continue
            leaves.append((p, child, f"/pages/{p}/blocks/{b}/content/{c}", piece))
    if "\n".join(piece for _, _, _, piece in leaves) != text:
        return None
    mapped = []
    cursor = 0
    for offset, (p, child, pointer, piece) in enumerate(leaves):
        if offset:
            cursor += 1
        source = entries.get(pointer)
        fine = _fine_ranges(child, source["tokens"], piece, pure=pure) if source and source["source"] == "analysis" else None
        if not fine and child.bbox is not None:
            fine = [{"bbox": list(child.bbox), "level": "block", "markdown_char_start": 0, "markdown_char_end": len(piece)}]
        for item in fine or []:
            item["markdown_char_start"] += cursor
            item["markdown_char_end"] += cursor
            item["_block_range"] = [cursor, cursor + len(piece)]
            item["element"] = pointer
            mapped.append((p, item))
        cursor += len(piece)
    return mapped


def _marker_intervals(probe: str, text: str, identifiers: set[int]) -> list[tuple[int, int, int]] | None:
    """Decode controlled markers only after exact full-emission verification."""
    pattern = re.compile(re.escape(_PROBE) + r"([BE])(\d+)X")
    pieces = []
    intervals = []
    opened: dict[int, int] = {}
    finished: set[int] = set()
    previous = cursor = 0
    for match in pattern.finditer(probe):
        piece = probe[previous : match.start()]
        pieces.append(piece)
        cursor += len(piece)
        kind, identifier = match.group(1), int(match.group(2))
        if identifier not in identifiers:
            return None
        if kind == "B":
            if identifier in opened or identifier in finished:
                return None
            opened[identifier] = cursor
        else:
            if identifier not in opened:
                return None
            intervals.append((opened.pop(identifier), cursor, identifier))
            finished.add(identifier)
        previous = match.end()
    pieces.append(probe[previous:])
    if opened or finished != identifiers or "".join(pieces) != text:
        return None
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        return None
    return intervals


def _continuation_ranges(
    middle: MiddleJson,
    group: list[tuple[int, int, Any]],
    text: str,
    entries: dict[str, Any],
    *,
    pure: bool,
) -> list[tuple[int, dict[str, Any]]] | None:
    """Trace physical paragraph fragments through the public document renderer.

    Controlled boundary markers are not source-text searches. Accept ownership
    only when removing every marker reproduces the complete original emission.
    The renderer alone decides continuation order and intervening image placement.
    Boundary rewrites (for example dehyphenation) that markers disturb fail closed.
    """
    if len(group) < 2 or _PROBE in text:
        return None
    marked = []
    traced: dict[int, tuple[int, int, Any]] = {}
    opaque = []
    for index, (p, b, block) in enumerate(group):
        if block.type in {"text", "ref_text"} and block.content:
            opening = TextSpan(type="text", content=f"{_PROBE}B{index}X")
            closing = TextSpan(type="text", content=f"{_PROBE}E{index}X")
            marked.append((p, b, block.model_copy(update={"content": [opening, *block.content, closing]})))
            traced[index] = (p, b, block)
        else:
            marked.append((p, b, block))
            piece = render_markdown(_document(middle, [(p, b, block)]), pure_markdown=pure)
            if piece:
                opaque.append((p, b, block, piece))
    if not traced:
        return None
    probe = render_markdown(_document(middle, marked), pure_markdown=pure)
    intervals = _marker_intervals(probe, text, set(traced))
    if intervals is None:
        return None
    # Account for all unmarked emissions in their producer order at the current
    # cursor. Do not search for an image/table/string inside the final Markdown.
    physical = []
    opaque_index = cursor = 0
    for start, end, identifier in [*intervals, (len(text), len(text), -1)]:
        gap = text[cursor:start]
        while gap and not gap.isspace():
            gap = gap.lstrip("\n")
            if opaque_index == len(opaque):
                return None
            p, b, block, piece = opaque[opaque_index]
            if not gap.startswith(piece):
                return None
            offset = start - len(gap)
            physical.append((offset, offset + len(piece), p, b, block))
            gap = gap[len(piece) :]
            opaque_index += 1
        if identifier != -1:
            p, b, block = traced[identifier]
            physical.append((start, end, p, b, block))
        cursor = end
    if opaque_index != len(opaque):
        return None
    mapped = []
    # Markdown continuation order can differ from physical page reading order.
    for start, end, p, b, block in sorted(physical, key=lambda item: (item[2], item[3])):
        if start == end:
            continue
        pointer = f"/pages/{p}/blocks/{b}"
        entry = entries.get(pointer)
        fine = (
            _fine_ranges(block, entry["tokens"], text[start:end], pure=pure)
            if entry and entry["source"] == "analysis"
            else None
        )
        if not fine and block.bbox is not None:
            fine = [
                {
                    "bbox": list(block.bbox),
                    "level": "formula" if block.type == "equation" else "block",
                    "markdown_char_start": 0,
                    "markdown_char_end": end - start,
                }
            ]
        for item in fine or []:
            item["markdown_char_start"] += start
            item["markdown_char_end"] += start
            item["_block_range"] = [start, end]
            item["element"] = pointer
            mapped.append((p, item))
    return mapped


def render_markdown_map(
    middle: MiddleJson, positioned: dict[str, Any], *, pure_markdown: bool = False
) -> tuple[str, dict[str, Any]]:
    """Return Markdown and its standalone box/range map.

    With pure_markdown every probe uses the shared renderer's HTML-free dialect,
    so ranges are verified directly against the emitted GFM.

    Continuations lacking a public ownership trace use an explicitly marked group
    range: all boxes in that group refer to the shared logical output, not to an
    invented word subdivision. No private DocVortex implementation is imported.
    """
    markdown = render_markdown(middle, pure_markdown=pure_markdown)
    groups = _groups(middle)
    emissions = [render_markdown(_document(middle, group), pure_markdown=pure_markdown) for group in groups]
    if "\n\n".join(part for part in emissions if part) != markdown:
        # Global anchors or other context-sensitive document operations cannot
        # safely be split through the installed dependency's public interface.
        groups = [[item for group in groups for item in group]]
        emissions = [markdown]
    entries = {entry["element"]: entry for entry in positioned["extensions"]["mineru_positions"]["entries"]}
    pages = [{"page_idx": page.page_idx, "blocks": []} for page in middle.pages]
    cursor = 0
    emitted = False
    for group_index, (group, text) in enumerate(zip(groups, emissions)):
        if not text:
            continue
        if emitted:
            cursor += 2
        start, end = cursor, cursor + len(text)
        visible = [(p, b, block) for p, b, block in group if block.type not in PAGE_AUXILIARY_BLOCK_TYPES]
        list_ranges = _continuation_ranges(middle, group, text, entries, pure=pure_markdown) or _list_ranges(
            visible, text, entries, pure=pure_markdown
        )
        fine = list_ranges
        if list_ranges:
            for p, item in list_ranges:
                item["markdown_char_start"] += start
                item["markdown_char_end"] += start
                item["_block_range"] = [offset + start for offset in item["_block_range"]]
                pages[p]["blocks"].append(item)
        elif len(visible) == 1:
            p, b, block = visible[0]
            entry = entries.get(f"/pages/{p}/blocks/{b}")
            if entry is not None and entry["source"] == "analysis":
                fine = _fine_ranges(block, entry["tokens"], text, pure=pure_markdown)
            if fine:
                for item in fine:
                    item["markdown_char_start"] += start
                    item["markdown_char_end"] += start
                    item["_block_range"] = [start, end]
                    item["element"] = f"/pages/{p}/blocks/{b}"
                pages[p]["blocks"].extend(fine)
        if not fine:
            for p, b, block in visible:
                if block.bbox is None:
                    continue
                item = {
                    "element": f"/pages/{p}/blocks/{b}",
                    "bbox": list(block.bbox),
                    "level": ("formula" if block.type == "equation" else "block") if len(visible) == 1 else "group",
                    "markdown_char_start": start,
                    "markdown_char_end": end,
                }
                if len(visible) > 1:
                    item["group_id"] = group_index
                pages[p]["blocks"].append(item)
        cursor = end
        emitted = True
    assert cursor == len(markdown)
    _hierarchy(pages, middle, entries)
    _caption_children(pages, middle, entries, markdown, pure=pure_markdown)
    mapping = {
        "version": 2,
        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "offset_units": "unicode_codepoints",
        "coordinate_system": "top_left_xyxy",
        "bbox_units": "normalized",
        "pages": pages,
    }
    return markdown, mapping


def _hierarchy(pages: list[dict[str, Any]], middle: MiddleJson, entries: dict[str, Any]) -> None:
    """Group by producer identities; duplicate multi-line tokens with shared IDs.

    Each duplicate has the complete Markdown range and source parts, but a local
    box. Unknown line ownership stays explicitly in the block's elements array.
    """
    for page in pages:
        blocks: dict[str, dict[str, Any]] = {}
        for item in page["blocks"]:
            pointer = item.pop("element")
            block_start, block_end = item.pop("_block_range", [item["markdown_char_start"], item["markdown_char_end"]])
            if pointer not in blocks:
                block: Any = middle
                for component in pointer.strip("/").split("/"):
                    block = block[int(component)] if isinstance(block, list) else getattr(block, component)
                blocks[pointer] = {
                    "element": pointer,
                    "type": str(block.type),
                    "bbox": list(block.bbox) if block.bbox is not None else None,
                    "level": item["level"] if item["level"] in {"block", "group", "formula"} else "block",
                    "markdown_char_start": block_start,
                    "markdown_char_end": block_end,
                    "lines": [],
                    "elements": [],
                    "children": [],
                }
            parent = blocks[pointer]
            parent["markdown_char_start"] = min(parent["markdown_char_start"], block_start)
            parent["markdown_char_end"] = max(parent["markdown_char_end"], block_end)
            if "group_id" in item:
                parent["group_id"] = item["group_id"]
            parts = item.get("source_parts") or (
                [{"source_span": item["source_span"], "bbox": item["bbox"]}] if "source_span" in item else []
            )
            known = {line["line_id"]: line for line in entries.get(pointer, {}).get("lines", [])}
            owned = {part["source_span"]["line"] for part in parts}
            if not owned or not owned.issubset(known):
                parent["elements"].append(item)
                continue
            token_id = f"{pointer}/tokens/{item['markdown_char_start']}:{item['markdown_char_end']}"
            for line_id in sorted(owned):
                line = next((line for line in parent["lines"] if line["line_id"] == line_id), None)
                if line is None:
                    line = {**deepcopy(known[line_id]), "level": "line", "elements": []}
                    parent["lines"].append(line)
                child = deepcopy(item)
                child["token_id"] = token_id
                boxes = [part["bbox"] for part in parts if part["source_span"]["line"] == line_id]
                child["bbox"] = [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ]
                if len(owned) > 1:
                    child["source_parts"] = deepcopy(parts)
                    child["bboxes"] = [part["bbox"] for part in parts]
                line["elements"].append(child)
                for key, operation in (("markdown_char_start", min), ("markdown_char_end", max)):
                    line[key] = operation(line.get(key, item[key]), item[key])
        page["blocks"] = list(blocks.values())
        for parent in page["blocks"]:
            parent["lines"].sort(key=lambda line: line["line_id"])


def _caption_children(
    pages: list[dict[str, Any]],
    middle: MiddleJson,
    entries: dict[str, Any],
    markdown: str,
    *,
    pure: bool,
) -> None:
    """Retain caption identity under its visual parent, with verified ownership.

    Probe the visual container rather than guessing where its caption text occurs.
    Unsupported container contexts keep explicit shared parent/group ranges.
    """
    caption_types = {"image_caption", "table_caption", "chart_caption"}
    for p, page in enumerate(pages):
        for parent in page["blocks"]:
            if parent["type"] not in {"image", "table", "chart"}:
                continue
            b = int(parent["element"].split("/")[-1])
            block = middle.pages[p].blocks[b]
            captions = {c: child for c, child in enumerate(block.content) if child.type in caption_types and child.content}
            if not captions:
                continue
            start, end = parent["markdown_char_start"], parent["markdown_char_end"]
            text = markdown[start:end]
            intervals = None
            if parent["level"] != "group" and _PROBE not in text:
                children = list(block.content)
                for c, child in captions.items():
                    children[c] = child.model_copy(
                        update={
                            "content": [
                                TextSpan(type="text", content=f"{_PROBE}B{c}X"),
                                *child.content,
                                TextSpan(type="text", content=f"{_PROBE}E{c}X"),
                            ]
                        }
                    )
                marked = block.model_copy(update={"content": children})
                probe = render_markdown(_document(middle, [(p, b, marked)]), pure_markdown=pure)
                intervals = _marker_intervals(probe, text, set(captions))
            owned = {c: (left + start, right + start) for left, right, c in intervals or []}
            mapped = []
            for c, child in captions.items():
                pointer = f"{parent['element']}/content/{c}"
                left, right = owned.get(c, (start, end))
                if left == right:
                    continue
                entry = entries.get(pointer)
                fine = None
                if c in owned and entry and entry["source"] == "analysis":
                    fine = _fine_ranges(child, entry["tokens"], markdown[left:right], pure=pure, inline_only=True)
                if not fine:
                    fine = [
                        {
                            "bbox": list(child.bbox) if child.bbox is not None else None,
                            "level": ("block" if child.bbox is not None else "unavailable") if c in owned else "group",
                            "markdown_char_start": 0,
                            "markdown_char_end": right - left,
                        }
                    ]
                for item in fine:
                    item["element"] = pointer
                    item["markdown_char_start"] += left
                    item["markdown_char_end"] += left
                    item["_block_range"] = [left, right]
                    if c not in owned:
                        item["group_id"] = parent.get("group_id", parent["element"])
                    mapped.append(item)
            child_pages = [{"page_idx": page["page_idx"], "blocks": mapped}]
            _hierarchy(child_pages, middle, entries)
            parent["children"] = child_pages[0]["blocks"]
