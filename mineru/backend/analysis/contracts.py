# Copyright (c) Opendatalab. All rights reserved.
"""文档分析分支汇合前使用的内部结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

AnalyzeEffort: TypeAlias = Literal["flash", "medium", "high", "xhigh"]
# Auto resolves to txt/ocr; all explicitly compares native and VLM text per block.
ParseMode: TypeAlias = Literal["auto", "txt", "ocr", "all"]
ResolvedParseMode: TypeAlias = Literal["txt", "ocr", "all"]
OfficeSuffix: TypeAlias = Literal["doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf", "odt", "ods", "odp"]


@dataclass(slots=True)
class AnalysisResult:
    """封装 model-list 及分析分支最终采用的元数据和计时结果。"""

    model_list: list[list[dict[str, Any]]]
    effort: AnalyzeEffort
    parse_mode: ResolvedParseMode
    elapsed: float
    layout_geometry: dict[str, Any] | None = None
