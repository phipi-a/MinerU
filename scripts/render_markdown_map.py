"""Visualize producer PDF lines and word boxes from a version-2 Markdown map."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import html
import io
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from docvortex.document.pdf import PDFDocument
from PIL import Image, ImageDraw, ImageFont


def color_for(index: int) -> tuple[int, int, int]:
    """Use contrasting deterministic colors for successive physical lines."""
    return tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb((index * 0.618034) % 1, 0.8, 0.68))


def iter_nodes(blocks: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Visit visual captions as well as their owning top-level blocks."""
    for block in blocks:
        yield block
        yield from iter_nodes(block.get("children", []))


def render_page(image: Image.Image, page: dict[str, Any]) -> tuple[Image.Image, dict[str, int]]:
    """Draw only saved producer geometry; never discover or assign PDF lines."""
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=max(12, round(image.width / 95)))
    counts: Counter[str] = Counter()
    number = 0

    def rectangle(box: list[float], color: tuple[int, int, int], *, tint: bool = False) -> tuple[float, ...]:
        if len(box) != 4 or not all(0 <= x <= 1 for x in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid normalized box: {box}")
        pixels = (box[0] * image.width, box[1] * image.height, box[2] * image.width, box[3] * image.height)
        draw.rectangle(pixels, outline=(*color, 255), fill=(*color, 22) if tint else None, width=2)
        return pixels

    def label(text: str, pixels: tuple[float, ...], color: tuple[int, int, int]) -> None:
        width = draw.textlength(text, font=font)
        x = max(0, pixels[0] - width - 5)
        y = max(0, pixels[1] - 2)
        bounds = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bounds, fill=(255, 255, 255, 240))
        draw.text((x, y), text, font=font, fill=(*color, 255))

    for block in iter_nodes(page["blocks"]):
        caption = block["type"].endswith("_caption")
        if caption:
            counts["captions"] += 1
        for line in block["lines"]:
            number += 1
            color = color_for(number)
            counts["lines"] += 1
            if line.get("bbox") is not None:
                pixels = rectangle(line["bbox"], color, tint=True)
                label(f"caption L{number}" if caption else f"L{number}", pixels, color)
            for element in line["elements"]:
                parts = element.get("source_parts", [])
                boxes = [part["bbox"] for part in parts if part["source_span"]["line"] == line["line_id"]]
                if not boxes and element.get("bbox") is not None:
                    boxes = [element["bbox"]]
                for box in boxes:
                    pixels = rectangle(box, color)
                    if element["level"] not in {"word", "formula"}:
                        label(element["level"], pixels, color)
                counts[element["level"]] += 1
        # These mappings have no verified line ownership. Do not assign a line.
        for element in block["elements"]:
            if element.get("bbox") is None:
                continue
            number += 1
            color = color_for(number)
            kind = block["type"] if element["level"] in {"block", "group", "formula"} else element["level"]
            for box in element.get("bboxes", [element["bbox"]]):
                pixels = rectangle(box, color, tint=True)
                label(f"{kind} {number} [{element['level']}]", pixels, color)
            counts[f"unassigned_{kind}"] += 1
    combined = Image.alpha_composite(canvas, overlay).convert("RGB")
    overlay.close()
    canvas.close()
    return combined, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--map", type=Path, help="Existing version-2 map; otherwise parse the PDF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--tier", choices=["basic", "standard", "advanced"], default="standard")
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("scale must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.map:
        mapping = json.loads(args.map.read_text())
    else:
        from mineru.parser import parse

        result = parse(args.pdf, tier=args.tier, positions=True)
        markdown, mapping = result.markdown_with_map()
        assert markdown == result.markdown()
        assert mapping["markdown_sha256"] == hashlib.sha256(markdown.encode()).hexdigest()
        (args.output / "markdown.md").write_bytes(markdown.encode())
    if mapping["version"] != 2:
        raise ValueError("Expected a version-2 hierarchical map")
    (args.output / "markdown_map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    records = []
    gallery = []
    with PDFDocument(str(args.pdf)) as document:
        for page in mapping["pages"]:
            index = page["page_idx"]
            artifact = document.render_image(index, image_format="png", scale=args.scale)
            with Image.open(io.BytesIO(artifact.data)) as original:
                image, counts = render_page(original, page)
            name = f"page_{index + 1:03d}.png"
            image.save(args.output / name)
            image.close()
            records.append({"page_idx": index, "image": name, "counts": counts})
            gallery.append(f'<h2>Page {index + 1}</h2><a href="{name}"><img src="{name}" loading="lazy"></a>')
            print(f"Rendered page {index + 1}: {counts}", flush=True)
    manifest = {"pdf": str(args.pdf), "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(), "pages": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Attention: PDF layout map</title>'
        "<style>body{font-family:sans-serif;max-width:1500px;margin:2em auto}img{width:100%}</style>"
        "<h1>PDF line and word provenance</h1><p>Each L-number identifies a producer PDF line, in reading order. "
        "Word outlines share the line color. Formula regions remain atomic. "
        "Span, line, block and group labels explicitly identify coarse precision; these are not word boxes. "
        "Tables, images and their captions have separate labeled outlines. Numbers restart on each page.</p>"
        f"<p>Source: {html.escape(args.pdf.name)}</p>" + "\n".join(gallery),
        encoding="utf-8",
    )
    print(f"Gallery: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
