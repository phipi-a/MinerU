"""Window-bounded OCR text candidates, independent of the PDF's text objects."""

from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from loguru import logger
from PIL.Image import Image

from .selection import TextCandidate, has_inline_math

__all__ = ["TesseractWord", "check_tesseract", "recognize_pages", "fill_tesseract_candidates", "parse_tsv"]

TESSERACT_DPI = 300
TESSERACT_WORKERS = 8
TESSERACT_TIMEOUT = 180


@dataclass(frozen=True, slots=True)
class TesseractWord:
    """An OCR word and its own normalized, top-left page geometry."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    line: tuple[int, int, int]


def check_tesseract() -> str:
    """Fail before loading models if the optional executable or English data is missing."""
    executable = shutil.which("tesseract")
    if executable is None:
        raise RuntimeError("text_source='tesseract' requires Tesseract and its English language data on PATH")
    result = subprocess.run([executable, "--list-langs"], capture_output=True, text=True, check=True, timeout=10)
    if "eng" not in (result.stdout + result.stderr).split():
        raise RuntimeError("text_source='tesseract' requires the Tesseract English language data (eng)")
    return executable


def parse_tsv(tsv: str, size: tuple[int, int]) -> list[TesseractWord]:
    """Read only valid word records; never invent finer boxes than Tesseract reports."""
    width, height = size
    words = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        try:
            left, top, w, h = (int(row[key]) for key in ("left", "top", "width", "height"))
            confidence = float(row["conf"])
            line = tuple(int(row[key]) for key in ("block_num", "par_num", "line_num"))
        except (KeyError, ValueError, TypeError):
            continue
        if not math.isfinite(confidence) or confidence < 0 or w <= 0 or h <= 0:
            continue
        if left < 0 or top < 0 or left + w > width or top + h > height:
            continue
        words.append(
            TesseractWord(
                row["text"].strip(), (left / width, top / height, (left + w) / width, (top + h) / height), confidence, line
            )
        )
    return words


def recognize_pages(images: list[Image], *, page_start: int = 0) -> list[list[TesseractWord]]:
    """Run at most eight single-threaded subprocesses; no PDF handles cross threads."""
    executable = check_tesseract()
    environment = dict(os.environ, OMP_THREAD_LIMIT="1")

    def recognize(item: tuple[int, Image]) -> list[TesseractWord]:
        index, image = item
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            try:
                result = subprocess.run(
                    [executable, "stdin", "stdout", "-l", "eng", "--dpi", str(TESSERACT_DPI), "tsv"],
                    input=buffer.getvalue(),
                    capture_output=True,
                    env=environment,
                    timeout=TESSERACT_TIMEOUT,
                    check=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Tesseract timed out on PDF page {page_start + index + 1}") from exc
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.decode("utf-8", errors="replace")[-1000:]
                raise RuntimeError(f"Tesseract failed on PDF page {page_start + index + 1}: {detail}") from exc
        words = parse_tsv(result.stdout.decode("utf-8"), image.size)
        logger.info("Tesseract page {} finished: {} words", page_start + index + 1, len(words))
        return words

    with ThreadPoolExecutor(max_workers=TESSERACT_WORKERS) as executor:
        return list(executor.map(recognize, enumerate(images)))


def _coverage(word: TesseractWord, box: list[float]) -> float:
    x0, y0, x1, y1 = word.bbox
    overlap = max(0.0, min(x1, box[2]) - max(x0, box[0])) * max(0.0, min(y1, box[3]) - max(y0, box[1]))
    return overlap / ((x1 - x0) * (y1 - y0))


def fill_tesseract_candidates(candidates: list[TextCandidate], words: list[TesseractWord]) -> None:
    """Assign each word to one layout block; keep inline mathematics VLM-owned."""
    assignments: list[list[TesseractWord]] = [[] for _ in candidates]
    for word in words:
        scores = [_coverage(word, candidate.block["bbox"]) for candidate in candidates]
        if scores and max(scores) >= 0.5:
            assignments[max(range(len(scores)), key=scores.__getitem__)].append(word)
    for candidate, block_words in zip(candidates, assignments, strict=True):
        if not block_words or has_inline_math(candidate.vlm_block.get("content")):
            continue
        block = candidate.block
        text = ""
        tokens: list[dict[str, Any]] = []
        lines = []
        previous_line = None
        for word in block_words:
            if text:
                text += " " if word.line == previous_line else "\n"
            if word.line != previous_line:
                lines.append({"line_id": len(lines), "bbox": list(word.bbox)})
            else:
                box = lines[-1]["bbox"]
                lines[-1]["bbox"] = [
                    min(box[0], word.bbox[0]),
                    min(box[1], word.bbox[1]),
                    max(box[2], word.bbox[2]),
                    max(box[3], word.bbox[3]),
                ]
            start = len(text)
            text += word.text
            tokens.append(
                {
                    "type": "word",
                    "level": "word",
                    "text": word.text,
                    "bbox": list(word.bbox),
                    "start": start,
                    "end": len(text),
                    "confidence": word.confidence,
                }
            )
            previous_line = word.line
        # VLM styles/line boxes belong to the other producer, not these OCR words.
        for key in ("style", "lines", "_position_tokens", "_position_content", "_position_lines"):
            block.pop(key, None)
        block["content"] = text
        block["_position_content"] = text
        block["_position_tokens"] = tokens
        block["_position_lines"] = lines
