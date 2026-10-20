"""Headless REST demonstration: save exact Markdown and its separate box map."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    request = {
        "files": [{"source": {"type": "local", "path": str(args.pdf.resolve())}}],
        "tier": "advanced",
        "ocr_mode": "auto",
        "output_formats": ["markdown", "markdown_map"],
    }
    save_json(args.output / "request.json", request)
    started_at = time.monotonic()
    with httpx.Client(base_url=args.api_url, timeout=60, trust_env=False) as client:
        health = client.get("/v1/health")
        health.raise_for_status()
        save_json(args.output / "health.json", health.json())
        assert "markdown_map" in health.json()["features"]["output_formats"]
        submitted = client.post("/v1/parse/jobs", json=request)
        submitted.raise_for_status()
        job = submitted.json()
        save_json(args.output / "submit_response.json", job)
        job_id = job["job_id"]
        print(f"Submitted {job_id}", flush=True)
        previous = None
        while time.monotonic() - started_at < args.timeout:
            response = client.get(f"/v1/parse/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            save_json(args.output / "response.json", job)
            if job["status"] != previous:
                print(f"Status: {job['status']}", flush=True)
                previous = job["status"]
            if job["status"] in {"completed", "partial", "failed", "canceled"}:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Job {job_id} did not finish; its saved response can be used to resume")
        assert job["status"] == "completed", job
        downloads = {}
        for format_name, filename in [("markdown", "markdown.md"), ("markdown_map", "markdown_map.json")]:
            reference = job["files"][0]["output_files"][format_name]
            response = client.get(f"/v1/files/{reference['file_id']}/content")
            response.raise_for_status()
            assert len(response.content) == reference["bytes"]
            (args.output / filename).write_bytes(response.content)
            downloads[filename] = {"file_id": reference["file_id"], "bytes": len(response.content)}
    markdown_bytes = (args.output / "markdown.md").read_bytes()
    markdown = markdown_bytes.decode("utf-8")
    mapping = json.loads((args.output / "markdown_map.json").read_bytes())
    assert mapping["markdown_sha256"] == hashlib.sha256(markdown_bytes).hexdigest()
    assert mapping["offset_units"] == "unicode_codepoints"
    assert "markdown" not in mapping and "extensions" not in mapping
    levels: Counter[str] = Counter()
    samples = {}
    for page in mapping["pages"]:
        nodes = list(page["blocks"])
        for node in nodes:
            nodes.extend(node.get("children", []))
        regions = [
            element
            for block in nodes
            for element in block["elements"] + [e for line in block["lines"] for e in line["elements"]]
        ]
        for block in regions:
            start, end = block["markdown_char_start"], block["markdown_char_end"]
            assert 0 <= start < end <= len(markdown)
            levels[block["level"]] += 1
            for box in block.get("bboxes", [block["bbox"]]):
                assert len(box) == 4 and all(0 <= coordinate <= 1 for coordinate in box)
                assert box[0] < box[2] and box[1] < box[3]
            fragment = markdown[start:end]
            if "<sup>" in fragment and block["level"] == "word":
                samples.setdefault("superscript_word", {"page_idx": page["page_idx"], **block, "fragment": fragment})
            if block["level"] == "word" and "bboxes" in block:
                samples.setdefault("multipart_word", {"page_idx": page["page_idx"], **block, "fragment": fragment})
            if block["level"] == "formula":
                samples.setdefault("formula", {"page_idx": page["page_idx"], **block, "fragment": fragment})
    assert levels["word"] > 0 and "superscript_word" in samples
    summary = {
        "job_id": job_id,
        "status": job["status"],
        "tier": job["tier"],
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
        "source_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "markdown_sha256": mapping["markdown_sha256"],
        "pages": len(mapping["pages"]),
        "markdown_codepoints": len(markdown),
        "markdown_utf8_bytes": len(markdown_bytes),
        "mapped_regions_by_level": dict(levels),
        "hash_matches_downloaded_file": True,
        "all_ranges_valid": True,
        "all_boxes_valid": True,
        "downloads": downloads,
    }
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "samples.json", samples)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
