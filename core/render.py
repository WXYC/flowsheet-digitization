"""PDF → PNG rendering via the poppler `pdftoppm` and `pdfinfo` CLIs.

We shell out instead of pulling pypdfium2/pdf2image because:
  * poppler is already standard on macOS via Homebrew and on Ubuntu via apt,
  * it's the same toolchain we used to generate the sample images,
  * it gives us the most predictable file naming.

Renders are idempotent — if the target PNG already exists we skip unless
`force=True`. Page numbers are 1-based.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class RenderError(RuntimeError):
    """Raised when pdftoppm or pdfinfo cannot process a PDF."""


def _require_poppler() -> None:
    if shutil.which("pdftoppm") is None or shutil.which("pdfinfo") is None:
        raise RenderError(
            "poppler not found on PATH (need pdftoppm and pdfinfo). "
            "Install with `brew install poppler` (macOS) or `apt-get install poppler-utils`."
        )


def image_path_for(out_dir: Path, page_number: int) -> Path:
    """Return the canonical path for a rendered page image.

    Two-digit zero-padding is the default; longer is allowed for PDFs > 99 pages.
    """
    width = max(2, len(str(page_number)))
    return out_dir / f"page-{page_number:0{width}d}.png"


def count_pages(pdf_path: Path) -> int:
    """Return the number of pages in `pdf_path`. Raises RenderError on failure."""
    _require_poppler()
    if not pdf_path.is_file():
        raise RenderError(f"PDF not found: {pdf_path}")
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RenderError(f"pdfinfo failed for {pdf_path}: {result.stderr.strip()}")
    match = re.search(r"^Pages:\s+(\d+)", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RenderError(f"pdfinfo output for {pdf_path} did not include a Pages line")
    return int(match.group(1))


def render_page(
    pdf_path: Path,
    page_number: int,
    out_dir: Path,
    dpi: int = 300,
    *,
    force: bool = False,
) -> Path:
    """Render a single page of `pdf_path` to a PNG in `out_dir`.

    Returns the output path. Idempotent: skips if the file already exists,
    unless `force=True`. Raises RenderError on failure or out-of-range pages.
    """
    _require_poppler()
    if not pdf_path.is_file():
        raise RenderError(f"PDF not found: {pdf_path}")
    if page_number < 1:
        raise RenderError(f"page_number must be >= 1, got {page_number}")

    target = image_path_for(out_dir, page_number)
    if target.exists() and not force:
        return target

    out_dir.mkdir(parents=True, exist_ok=True)

    # pdftoppm names output as <root>-<padded-page>.png, where the padding
    # width depends on total page count (not the requested page number). To
    # avoid colliding with any pre-existing output in `out_dir`, we render
    # into a per-call tempdir and move the single produced file to `target`.
    with tempfile.TemporaryDirectory(prefix="flowsheet_render_", dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        root = tmp_dir / "page"
        cmd = [
            "pdftoppm",
            "-r",
            str(dpi),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-png",
            str(pdf_path),
            str(root),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RenderError(
                f"pdftoppm failed for {pdf_path} page {page_number}: {result.stderr.strip()}"
            )

        produced: Path | None = None
        for cand in sorted(tmp_dir.iterdir()):
            match = re.match(r"^page-(\d+)\.png$", cand.name)
            if match and int(match.group(1)) == page_number:
                produced = cand
                break
        if produced is None:
            raise RenderError(
                f"pdftoppm reported success but produced no file for page {page_number}; "
                f"saw {[c.name for c in tmp_dir.iterdir()]}"
            )

        if target.exists():
            target.unlink()
        shutil.move(str(produced), str(target))
    return target
