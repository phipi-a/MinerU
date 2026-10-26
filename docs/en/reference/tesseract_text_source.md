# Tesseract text candidates in `all` mode

`all` independently recognizes PDF text with the VLM and compares a second text
candidate per layout block. By default that candidate comes from the PDF's text
objects. Set `text_source="tesseract"` to replace that candidate with Tesseract OCR.
Set `text_source="both"` to compare both PDF text and Tesseract against the same
original VLM result, independently for each block. The eligible candidate with
the higher similarity wins; PDF text wins ties. If neither candidate qualifies,
the complete VLM block is retained.
The VLM still runs, and the source PDF is never modified or rewritten.

This option supports `standard` and `advanced` with `ocr_mode="all"`. Other mode
and tier combinations are rejected. Image inputs converted to PDF also work.
The default `text_source="pdf"` retains the existing behavior.

## Python

```python
from mineru.parser import parse

result = parse(
    "book.pdf",
    tier="advanced",
    ocr_mode="all",
    text_source="both",
    page_range="1-10",
)
```

The same option is available on `parse_async`, `MinerUParser`, and
`MinerUApiParser`. The remote parser can omit `ocr_mode` when its server defaults
to `all` for the selected tier.

## CLI

```bash
mineru-kit parse book.pdf -o output/book.md \
  --tier advanced --ocr-mode all --text-source both
```

Add `--remote-url http://localhost:8000` to submit to a server that implements this
option. The `mineru` Doclib CLI and Gradio do not expose this option yet.

## HTTP

For `POST /v1/parse/jobs`, add the following fields to the normal request:

```json
{
  "tier": "advanced",
  "ocr_mode": "all",
  "text_source": "both"
}
```

The `/v1/tasks` and `/v1/file_parse` upload endpoints accept the same
`text_source=both` query parameter. `pdf` and `tesseract` remain available for
single-candidate comparisons.

## Execution and selection

- Tesseract must be on the server's `PATH`, with English (`eng`) language data.
  The REST Dockerfile installs `tesseract-ocr` and `tesseract-ocr-eng`. Local
  installations must provide these system dependencies separately.
- Pages render at 300 dpi within the existing processing windows. At most eight
  OCR subprocesses run concurrently, with `OMP_THREAD_LIMIT=1` each. No raster PDF
  or page images are persisted by this feature.
- Each page has a 180-second OCR timeout. Missing dependencies and subprocess
  failures fail the parse explicitly. A page with no words supplies an empty
  candidate, allowing the normal VLM selection for its blocks.
- TSV words retain their own normalized top-left bounding boxes and line order.
  Each word is assigned to at most one text block using geometric overlap.
- The existing normalized prose comparison applies: similarity must reach 0.90;
  short text must match exactly; word positions must exist. Rejected candidates
  restore the complete VLM block, including its own positions.
- Tables and display formulas stay with the existing VLM/formula pipeline.
  Tesseract candidates for blocks containing inline formulas are rejected;
  Tesseract words are not matched to formula geometry. In `both` mode a valid PDF
  candidate can still win on those blocks; otherwise the VLM result is retained.
- `extensions["mineru_text_selection"]` records `text_source` and each block's
  selected `source` (`txt`, `tesseract`, or `vlm`), reason, and similarity. The
  `both` mode additionally records each candidate's similarity, threshold, and
  acceptance/rejection reason under `candidates.pdf` and `candidates.tesseract`.
  Text and positions are always selected together from the same producer. The
  strict `extensions["mineru"]` tier/parse-mode contract is unchanged.

The extra OCR adds CPU work; it does not remove VLM inference. This option is
intended to provide independent text and word positions when PDF extraction is
unreliable, not to guarantee a faster parse or better recognition on every block.
