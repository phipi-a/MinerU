# Visible native PDF text in Hybrid parsing

Hybrid native text filling now requests DocVortex's `visible_only=True` evidence.
This prevents clipped text inside embedded Forms from leaking into unrelated
visible paragraphs and their word boxes. Text, span ownership, character geometry,
inline evidence, and native-table inputs use the same policy. Existing explicit
geometry inputs remain caller-owned and are not silently replaced.

The raw PDF character-count limit is unchanged. Geometry is still read lazily;
no model or page rendering is needed to apply the shared visibility filter.
The public raw DocVortex API remains available for consumers needing hidden OCR
text. Hybrid's existing OCR fallback handles spans without usable native text.
The change does not alter the layout model's block classification.

## Pinned fork dependency

The public visibility options are provided by the public
[DocVortex fork](https://github.com/phipi-a/DocVortex), including upstream changes
through commit `be20158` (package version `0.5.16`).
MinerU's `[tool.uv.sources]` pins commit
`5fa9434f4b92cc9513a49d5c183e25a60d54ebd7` from branch
`fix/visible-hybrid-text`. The full commit pin makes installation reproducible
without a sibling checkout or relying on the branch remaining unchanged:

```bash
uv sync --extra dev --extra test
```

The source override applies to uv project workflows. Other installers do not
honor `[tool.uv.sources]`; they must explicitly install the pinned Git dependency.
Before publishing MinerU with this requirement, release the DocVortex API
additions and replace the source override with a minimum dependency version
containing them. Do not patch installed site-packages or import private
DocVortex visibility helpers from MinerU.

## Regression coverage

- Nested Form text that overlaps a visible heading but lies outside an ancestor
  clip is excluded before native span filling and word-box capture.
- Visible figure text remains available.
- Partially clipped characters survive with clipped geometry under page and Form
  rotation (DocVortex's visibility tests).
- Raw/visible snapshot caches stay separate, and snapshots remain usable after
  closing the PDF.
- Both the native and Python reference paths are exercised without inference
  models by `tests/unittest/test_pdf_visible_text.py` and the DocVortex tests.
