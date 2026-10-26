"""Measure sampled PDF pages with Tesseract and estimate whole-document OCR time."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.dpi < 1 or any(n < 1 for n in args.workers):
        parser.error("samples must be at least 2; dpi and workers must be positive")
    binary = shutil.which("tesseract")
    if binary is None:
        parser.error("tesseract must be available on PATH")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    images: list[Path] = []
    render_started = time.perf_counter()
    with pdfium.PdfDocument(args.pdf) as document:
        total_pages = len(document)
        count = min(args.samples, total_pages)
        indices = sorted({round(i * (total_pages - 1) / (count - 1)) for i in range(count)}) if count > 1 else [0]
        for index in indices:
            path = args.output_dir / f"page-{index + 1:04d}.png"
            page = document[index]
            try:
                bitmap = page.render(scale=args.dpi / 72)
                try:
                    bitmap.to_pil().save(path)
                finally:
                    bitmap.close()
            finally:
                page.close()
            images.append(path)
    render_seconds = time.perf_counter() - render_started
    report = {
        "input": str(args.pdf.resolve()),
        "tesseract": version,
        "total_pages": total_pages,
        "sample_pages": [i + 1 for i in indices],
        "dpi": args.dpi,
        "language": "eng",
        "omp_thread_limit": 1,
        "render_seconds": render_seconds,
        "runs": [],
    }
    print(f"Rendered {len(images)} / {total_pages} pages at {args.dpi} dpi in {render_seconds:.2f}s", flush=True)
    environment = dict(os.environ, OMP_THREAD_LIMIT="1")
    for workers in args.workers:
        run_dir = args.output_dir / f"workers-{workers}"
        run_dir.mkdir(exist_ok=True)

        def recognize(image: Path) -> dict[str, float | int]:
            started = time.perf_counter()
            output = run_dir / image.stem
            subprocess.run(
                [binary, str(image), str(output), "-l", "eng", "--dpi", str(args.dpi), "pdf", "txt"],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
                timeout=180,
            )
            elapsed = time.perf_counter() - started
            print(f"workers={workers} {image.stem}: {elapsed:.2f}s", flush=True)
            return {
                "page": int(image.stem.split("-")[1]),
                "seconds": elapsed,
                "text_bytes": output.with_suffix(".txt").stat().st_size,
            }

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pages = list(executor.map(recognize, images))
        ocr_seconds = time.perf_counter() - started
        merge_started = time.perf_counter()
        with pdfium.PdfDocument.new() as merged:
            for image in images:
                with pdfium.PdfDocument(run_dir / f"{image.stem}.pdf") as page_pdf:
                    merged.import_pages(page_pdf)
            merged.save(run_dir / "sample-searchable.pdf")
        merge_seconds = time.perf_counter() - merge_started
        estimated_seconds = (render_seconds + ocr_seconds + merge_seconds) * total_pages / len(images)
        run = {
            "workers": workers,
            "ocr_wall_seconds": ocr_seconds,
            "merge_seconds": merge_seconds,
            "estimated_full_document_minutes": estimated_seconds / 60,
            "pages": pages,
        }
        report["runs"].append(run)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"workers={workers}: OCR {ocr_seconds:.2f}s; estimated full PDF {estimated_seconds / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
