# Markdown box source map

The V1 API can return a separate, lightweight `markdown_map` artifact:

```json
{"output_formats": ["markdown", "markdown_map"]}
```

Download `output_files.markdown` and `output_files.markdown_map` through the normal
`/v1/files/{file_id}/content` endpoint. The map does not contain Markdown,
MiddleJson, semantic text copies, or the raw producer provenance extension.
Requesting the map automatically enables producer position collection.
The map is registered as a job output and follows the same retention policy as
Markdown and other artifacts. Its retention period starts after job cleanup;
download the paired files before they expire.

For an offline download with local images, request both formats:

```json
{"output_formats": ["zip", "markdown_map"]}
```

Download `output_files.zip` and extract it. The archive contains `markdown.md`,
`markdown_map.json`, and local image assets under `images/`, alongside the usual
structured result files. The bundled map is generated after images are
materialized, so its hash and offsets match the exact `markdown.md` inside the
archive. Keep the extracted directory structure to preserve relative image links.
The separately returned `output_files.markdown_map` still maps the standalone
Markdown output; image paths can differ, so use the map bundled with the ZIP for
the bundled Markdown. Requesting only `zip` does not add a map.

To select HTML-free Markdown for both standalone outputs and the ZIP, add the
`pure_markdown` option to the job request:

```json
{
  "files": [{"source": {"type": "file_id", "file_id": "file_example"}}],
  "output_formats": ["markdown", "markdown_map", "zip"],
  "pure_markdown": true
}
```

Download `output_files.zip` through `/v1/files/{file_id}/content`. Its
`markdown.md`, `markdown_map.json`, and local `images/` paths form an offline
bundle. The option also works on `/v1/tasks` and `/v1/file_parse`; multipart/raw
uploads accept `pure_markdown=true` as a query parameter and repeated
`output_formats=zip&output_formats=markdown_map` parameters.

The HTML-free variant uses GFM tables and strikethrough. Unsupported visual
styles become plain text, collapsible sections are expanded, and complex tables
lose merged cell spans. Code and formula literals retain their contents, including
literal angle brackets. Custom HTML anchor targets are removed. The document's
semantic representation and other output formats are unchanged. Remote image
URLs are not downloaded automatically.

The map is translated through DocVortex's explicit edit trace and hashed against
the final Markdown bytes. Table/code structure replacements use conservative
block ranges instead of retaining unsupported fine-grained positions. The default
`pure_markdown=false` preserves the existing HTML-enhanced Markdown behavior.

The pinned DocVortex dependency includes the shared
`docvortex.render.pure_markdown` API. For development against a local checkout,
install it with `uv pip install --python .venv/bin/python --no-deps -e ../DocVortex`.

```json
{
  "version": 2,
  "markdown_sha256": "…",
  "offset_units": "unicode_codepoints",
  "coordinate_system": "top_left_xyxy",
  "bbox_units": "normalized",
  "pages": [
    {
      "page_idx": 0,
      "blocks": [
        {
          "element": "/pages/0/blocks/0",
          "type": "text",
          "bbox": [0.1, 0.2, 0.8, 0.4],
          "level": "block",
          "markdown_char_start": 42,
          "markdown_char_end": 66,
          "elements": [],
          "lines": [
            {
              "line_id": 0,
              "bbox": [0.1, 0.2, 0.8, 0.25],
              "level": "line",
              "markdown_char_start": 42,
              "markdown_char_end": 66,
              "elements": [
                {
                  "type": "word",
                  "bbox": [0.304, 0.2, 0.393, 0.25],
                  "level": "word",
                  "token_id": "/pages/0/blocks/0/tokens/42:66",
                  "markdown_char_start": 42,
                  "markdown_char_end": 66
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

Version 2 replaces the flat version-1 region array with physical blocks containing
`lines` and `elements`. Blocks carry their semantic `type`; display formulas,
tables, and images remain separate blocks. Each line's block-local `line_id` and
normalized geometry come directly from analysis, not text search or geometric
inference. Block `elements` hold mappings without verified line ownership,
including explicit span, block, or group fallbacks. No synthetic lines are added
for older results without line provenance. List leaves retain their own `element`
JSON pointers rather than being assigned invented container lines.

Visual blocks also expose a `children` array for their captions (`image_caption`,
`table_caption`, `chart_caption`). Each caption has its own `element` pointer,
`bbox`, global Markdown range, `lines`, `elements`, and `children`. Captions use
producer word/line provenance when available, independently of the visual body.
Without fine provenance, a caption keeps its own box and exact range at `block`
precision; missing geometry stays `null` with `unavailable` precision. If its
serialization cannot be separated safely, `group` precision explicitly shares
the parent's range instead. Caption positions are never inferred from the image
box, surrounding text, or repeated caption strings. Consumers should traverse
`children` recursively, not only the page's top-level `blocks`.

Offsets are global to the exact paired default Markdown output, start-inclusive
and end-exclusive. Page indices retain original zero-based document numbering.
Coordinates are normalized top-left `xyxy` boxes. Line ranges enclose their
mapped elements; they need not be disjoint or cover every structural character.
A word spanning lines appears under each owning line with the same `token_id`
and complete Markdown range. Each occurrence's `bbox` covers only its local
source parts. `source_parts` and `bboxes` retain all component regions; do not
interpret these occurrences as separate words. Inline and display formulas stay
atomic. Missing line geometry is represented as `bbox: null`, not guessed.

In Python:

```python
import hashlib
import json
from pathlib import Path

raw = Path("markdown.md").read_bytes()
source_map = json.loads(Path("markdown_map.json").read_bytes())
assert hashlib.sha256(raw).hexdigest() == source_map["markdown_sha256"]
markdown = raw.decode("utf-8")
region = source_map["pages"][0]["blocks"][0]["lines"][0]["elements"][0]
fragment = markdown[region["markdown_char_start"]:region["markdown_char_end"]]
```

For example, an author-name box can refer to the entire exact fragment
`Gugler<sup>1,2†</sup>,`, without splitting that box at the superscript boundary.

## Hash and Unicode

SHA-256 covers the exact UTF-8 Markdown bytes, with no added newline or Unicode
normalization. Preserve the downloaded bytes: changing line endings or saving
with a BOM produces a different hash. The hash identifies the paired output, not
correctness of visual recognition or authenticity against a malicious sender.

JavaScript string indexing uses UTF-16. Use `Array.from(markdown).slice(start,
end).join("")` or an explicit codepoint-to-UTF-16 conversion when consuming these
codepoint offsets. Unicode such as emoji is covered by regression tests.

## Precision and limits

The implementation composes public DocVortex render operations and records
emission offsets as those verified pieces are appended. It never searches for a
word occurrence in completed Markdown, nor imports private renderer functions.
The final assembled output must equal the unmodified default Markdown renderer.
Fine inline boundaries are accepted only when the public serialization composes
exactly across those owned boundaries, including escapes and styles.

- `word`, `formula`, `span`, `line`: verified fine provenance/serialization.
- `block`: existing source block geometry and its complete Markdown fragment.
- `group`: an explicitly coarser **shared logical render-group range**. Physical
  blocks in a continuation group share `group_id` and the group's Markdown range.
  A group may cross pages; its individual box does **not** cover every character
  in that group. Consumers requiring exact word correspondence must not treat a
  `group` entry as one. Paragraph continuations can retain physical word and line
  ownership across pages and intervening images: controlled boundary markers are
  rendered through the public document renderer, and accepted only if removing
  them exactly reproduces the original Markdown. No source-text search is used.
  Intervening images retain their own independent block ranges. Physical block
  order is preserved even when Markdown merges a later fragment into an earlier
  paragraph. Flat reference-list continuations can also be refined. Boundary
  rewrites that cannot be traced reversibly (such as some dehyphenation cases),
  unsupported styles and table continuation merges still use explicit fallbacks.

Context-sensitive escaping, non-flat lists, complex visual contents, and global
anchor interactions may force coarser mappings. The installed DocVortex version
has no public per-character tracing callback; unsupported operations are not
assigned guessed word offsets. The Markdown itself is never rewritten to make a
mapping pass. There is no promise that every Markdown character has a box;
structural separators and filtered page auxiliaries have no independent mapping.
Individual slices can include only an opening/closing style wrapper when that
wrapper spans multiple words; slices need not be standalone valid Markdown.

## SDK

```python
result = parser.parse(path, positions=True)
markdown, source_map = result.markdown_with_map()
# Or, when only the separate dictionary is wanted:
source_map = result.markdown_map()

# Save Markdown, local images, and their paired map to a directory:
from mineru.parser.writer import FileBasedDataWriter
result.save(FileBasedDataWriter("output"), include_markdown_map=True)

# Select the HTML-free variant, including the paired map:
markdown, source_map = result.markdown_with_map(pure_markdown=True)
result.save(FileBasedDataWriter("output"), include_markdown_map=True, pure_markdown=True)
```

Without `positions=True`, existing block boxes remain available but no word
geometry is invented. These methods map the default Markdown rendering, not
`FULL`/custom-marker or custom-image-renderer variants. This new artifact is not a
new shared `RenderFormat` and is not automatically exposed by every CLI.

## Headless REST demonstration

```bash
.venv/bin/python scripts/test_markdown_map_api_example.py \
  output_api/positioned_source/2606.30778v1.pdf \
  --output output_api/2606.30778v1_markdown_map
```

The script submits a real Advanced job, downloads both files, verifies the exact
byte hash, checks every range/box, and saves request/response, statistics, and
formatted token examples. It refuses to overwrite an existing directory.
