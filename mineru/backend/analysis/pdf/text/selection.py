"""Compare native text with independently recognized VLM text per layout block."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from ..constants import NOT_EXTRACT_TYPES

__all__ = ["SELECTION_FIELD", "TextCandidate", "prepare_text_candidates", "select_text_candidates"]

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


def _normalized(content: Any) -> str:
    text = re.sub(r"<[^>]*>", "", _prose(content))
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\w)[-\u00ad]\s*\n\s*(?=\w)", "", text)
    # Compare the visible prose, independently of styles, punctuation, or LaTeX spelling.
    return " ".join(re.findall(r"[^\W_]+", text))


def select_text_candidates(candidates: list[TextCandidate]) -> None:
    """Keep native text only when supported word positions and prose agree.

    The fixed threshold is a conservative heuristic, not a correctness score.
    Never align VLM words to unrelated native boxes after the decision.
    """
    for candidate in candidates:
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
        if invalid_text:
            reason = "invalid_native_characters"
        elif not native or not vlm:
            reason = "missing_comparable_text"
        elif similarity < threshold:
            reason = "text_differs"
        elif not has_words:
            reason = "missing_native_word_positions"
        else:
            reason = "text_agrees"
        use_native = reason == "text_agrees"
        if not use_native:
            # Restore all VLM text attributes, removing every native sidecar/style.
            block.clear()
            block.update(deepcopy(original))
        block[SELECTION_FIELD] = {
            "source": "txt" if use_native else "vlm",
            "reason": reason,
            "similarity": round(similarity, 6),
            "threshold": threshold,
        }
