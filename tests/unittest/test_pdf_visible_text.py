"""Keep clipped Form labels out of native text and word provenance."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from docvortex.analyzers.pdf import prepare_text_evidence
from docvortex.document.pdf import PDFDocument
from mineru.backend.analysis.pdf.snapshots import get_page_snapshot_entry
from mineru.backend.analysis.pdf.text.content import _fill_native_pdf_text_spans
from mineru.backend.analysis.pdf.text.models import _AnalyzeSpan
from mineru.backend.analysis.pdf.text.native import txt_spans_extract


def _nested_clipped_heading() -> bytes:
    """Place clipped figure text over a visible heading, like Attention page 13."""
    output = BytesIO()
    painter = Canvas(output, pagesize=(612, 792))
    painter.setFont("Times-Bold", 12)
    painter.drawString(108, 710, "Attention Visualizations")
    painter.beginForm("inner", 0, 0, 382, 230)
    painter.setFont("Helvetica-Bold", 16)
    painter.drawString(0, 217, "Input-Input Layer5")
    painter.drawString(20, 100, "VISIBLE FIGURE TEXT")
    painter.endForm()
    painter.beginForm("outer", 0, 0, 382, 194)
    painter.doForm("inner")
    painter.endForm()
    painter.saveState()
    painter.translate(108, 489.94)
    painter.scale(1.03578, 1.03578)
    painter.doForm("outer")
    painter.restoreState()
    painter.save()
    return output.getvalue()


@pytest.mark.parametrize("positions", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_hybrid_text_and_boxes_exclude_clipped_form_labels(positions: bool, owned: bool) -> None:
    with PDFDocument(_nested_clipped_heading()) as document, Image.new("RGB", (612, 792), "white") as image:
        page = document[0]
        raw = "".join(char["char"] for char in page.get_chars())
        assert "Input-Input Layer5" in raw
        assert "VISIBLE FIGURE TEXT" in raw
        span = _AnalyzeSpan("text", (107, 72, 233, 85), metadata={"_collect_positions": positions})
        entry = get_page_snapshot_entry([None], 0, page) if owned else None
        if owned and entry is None:
            pytest.skip("owned snapshot unavailable in reference backend")
        geometry = (
            prepare_text_evidence(page, snapshot=entry.get_full_snapshot(page)).geometry
            if entry is not None
            else page.get_chars_with_geometry(visible_only=True)
        )
        visible = "".join(char["char"] for char in geometry.chars)
        assert "Input-Input Layer5" not in visible
        assert "VISIBLE FIGURE TEXT" in visible
        _fill_native_pdf_text_spans(
            page,
            [span],
            image,
            1.0,
            page.size,
            page_text_geometry=geometry,
            detect_scripts=False,
            _native_text=entry.text_owner if entry is not None else None,
        )
        assert span.content == "Attention Visualizations"
        if positions:
            words = span.metadata["_position_words"]["words"]
            assert [word["text"] for word in words] == ["Attention", "Visualizations"]
            assert words[1]["bbox"][0] > words[0]["bbox"][2]
        if entry is not None:
            entry.close()


def test_direct_span_fill_requests_visible_geometry() -> None:
    """The uncached low-level entry must not fall back to raw PDF characters."""
    with PDFDocument(_nested_clipped_heading()) as document:
        page = document[0]
        span = _AnalyzeSpan("text", (107, 72, 233, 85), metadata={"_collect_positions": True})
        txt_spans_extract(
            page,
            [span],
            None,
            1.0,
            [(0, 0, 612, 792, None, None, None, "text")],
            [],
            detect_scripts=False,
        )
        assert span.content == "Attention Visualizations"
        assert len(span.metadata["_position_words"]["words"]) == 2
