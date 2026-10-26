"""Compare native text with independently recognized VLM text per layout block."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from ..constants import NOT_EXTRACT_TYPES

__all__ = [
    "SELECTION_FIELD",
    "TextCandidate",
    "prepare_text_candidates",
    "select_text_candidates",
    "select_best_text_candidates",
    "has_inline_math",
]

MIN_TEXT_SIMILARITY = 0.90
SELECTION_FIELD = "_mineru_text_selection"
_INLINE_MATH = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)


@dataclass
class TextCandidate:
    """Keep the original VLM block separate from its mutable native candidate."""

    block: dict[str, Any]
    vlm_block: dict[str, Any]


def prepare_text_candidates(pages: list[list[dict[str, Any]]]) -> list[TextCandidate]:
    """Clear eligible text only; formulas, tables, and layout remain VLM-owned."""
    candidates = []
    for page in pages:
        for block in page:
            if block.get("type") not in NOT_EXTRACT_TYPES:
                continue
            candidates.append(TextCandidate(block, deepcopy(block)))
            block["content"] = ""
    return candidates


def _prose(content: Any) -> str:
    if isinstance(content, str):
        return _INLINE_MATH.sub(" ", content)
    if isinstance(content, list):
        return "".join(
            _prose(span.get("content", "")) if span.get("type") != "equation_inline" else " "
            for span in content
            if isinstance(span, dict)
        )
    return ""


def has_inline_math(content: Any) -> bool:
    """Keep formula-bearing blocks on VLM until OCR has an independent math owner."""
    if isinstance(content, str):
        return _INLINE_MATH.search(content) is not None
    if isinstance(content, list):
        return any(span.get("type") == "equation_inline" or has_inline_math(span.get("content")) for span in content)
    return False


def _normalized(content: Any) -> str:
    text = re.sub(r"<[^>]*>", "", _prose(content))
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\w)[-\u00ad]\s*\n\s*(?=\w)", "", text)
    # Compare the visible prose, independently of styles, punctuation, or LaTeX spelling.
    return " ".join(re.findall(r"[^\W_]+", text))


@dataclass(frozen=True)
class _Assessment:
    similarity: float
    threshold: float
    reason: str

    def diagnostic(self) -> dict[str, Any]:
        return {"similarity": round(self.similarity, 6), "threshold": self.threshold, "reason": self.reason}


def _assess(candidate: TextCandidate, text_source: Literal["pdf", "tesseract"]) -> _Assessment:
    block, original = candidate.block, candidate.vlm_block
    native = _normalized(block.get("content"))
    vlm = _normalized(original.get("content"))
    similarity = SequenceMatcher(None, native, vlm, autojunk=False).ratio() if native and vlm else 0.0
    native_prose = _prose(block.get("content"))
    invalid_text = any(unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in native_prose)
    tokens = block.get("_position_tokens", [])
    has_words = any(token.get("level") == "word" for token in tokens)
    # Short texts must agree exactly: one changed digit/name can be the whole block.
    threshold = 1.0 if min(len(native), len(vlm)) < 20 else MIN_TEXT_SIMILARITY
    if text_source == "tesseract" and has_inline_math(original.get("content")):
        reason = "inline_math_preserved"
    elif invalid_text:
        reason = "invalid_native_characters"
    elif not native or not vlm:
        reason = "missing_comparable_text"
    elif similarity < threshold:
        reason = "text_differs"
    elif not has_words:
        reason = "missing_native_word_positions"
    else:
        reason = "text_agrees"
    return _Assessment(similarity, threshold, reason)


def select_text_candidates(candidates: list[TextCandidate], *, text_source: Literal["pdf", "tesseract"] = "pdf") -> None:
    """Keep native text only when supported word positions and prose agree.

    The fixed threshold is a conservative heuristic, not a correctness score.
    Never align VLM words to unrelated native boxes after the decision.
    """
    for candidate in candidates:
        block, original = candidate.block, candidate.vlm_block
        assessment = _assess(candidate, text_source)
        use_native = assessment.reason == "text_agrees"
        if not use_native:
            # Restore all VLM text attributes, removing every native sidecar/style.
            block.clear()
            block.update(deepcopy(original))
        block[SELECTION_FIELD] = {
            "source": ("tesseract" if text_source == "tesseract" else "txt") if use_native else "vlm",
            **assessment.diagnostic(),
        }


def select_best_text_candidates(pdf_candidates: list[TextCandidate], tesseract_candidates: list[TextCandidate]) -> None:
    """Choose the closest eligible producer; PDF wins ties, and boxes never cross producers."""
    if len(pdf_candidates) != len(tesseract_candidates):
        raise ValueError("PDF/Tesseract candidate count mismatch")
    for pdf, tesseract in zip(pdf_candidates, tesseract_candidates, strict=True):
        pdf_assessment = _assess(pdf, "pdf")
        tesseract_assessment = _assess(tesseract, "tesseract")
        options = [(pdf, pdf_assessment, "txt"), (tesseract, tesseract_assessment, "tesseract")]
        eligible = [option for option in options if option[1].reason == "text_agrees"]
        winner, assessment, source = max(eligible or options, key=lambda option: option[1].similarity)
        chosen = winner.block if eligible else pdf.vlm_block
        # Copy before clearing: the PDF candidate is the live output block.
        selected = deepcopy(chosen)
        pdf.block.clear()
        pdf.block.update(selected)
        pdf.block[SELECTION_FIELD] = {
            "source": source if eligible else "vlm",
            **assessment.diagnostic(),
            "candidates": {"pdf": pdf_assessment.diagnostic(), "tesseract": tesseract_assessment.diagnostic()},
        }
