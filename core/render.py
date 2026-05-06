"""Page-image extraction via the poppler `pdfimages` and `pdfinfo` CLIs.

The WXYC scan PDFs each embed exactly one CCITT Group 4 (lossless 1-bit
grayscale) image per page, at the page's native 300 PPI. We extract those
embedded images directly with `pdfimages -png` rather than rasterizing the
PDF page with `pdftoppm`. That:

  * avoids the rendering pipeline's anti-aliasing and color-space steps,
    so the output is bit-for-bit identical to the source CCITT bitmap,
  * gives us native resolution automatically (no DPI to choose),
  * is faster and produces smaller PNGs.

After extraction we **rotate 180°** before saving. The WXYC PDFs (produced
by EPSON Scan) draw their embedded bitmap with a content-stream
transformation matrix that flips the image right-side up at render time;
`pdftoppm` honors that transform, `pdfimages` does not — it dumps the raw
stored bitmap. Hardcoded 180° here is correct for this corpus (519 PDFs,
all from the same scanner workflow). If we ever ingest non-WXYC PDFs,
this becomes a flag.

Public surface keeps the function name `render_page` because callers care
about "give me a PNG of this page" — the implementation behind it is
strictly an implementation detail. Renders are idempotent: if the target
PNG already exists we skip unless `force=True`. Page numbers are 1-based.

Assumes each PDF page contains exactly one embedded image. The audit
documented in the README confirmed this across the entire WXYC corpus.
If a future PDF has 0 or 2+ images on a page, `render_page` raises
`RenderError` and the pipeline marks that page failed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


class RenderError(RuntimeError):
    """Raised when pdfimages or pdfinfo cannot process a PDF."""


def _require_poppler() -> None:
    if shutil.which("pdfimages") is None or shutil.which("pdfinfo") is None:
        raise RenderError(
            "poppler not found on PATH (need pdfimages and pdfinfo). "
            "Install with `brew install poppler` (macOS) or `apt-get install poppler-utils`."
        )


def image_path_for(out_dir: Path, page_number: int) -> Path:
    """Return the canonical path for an extracted page image.

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
    *,
    force: bool = False,
) -> Path:
    """Extract page `page_number` of `pdf_path` to a PNG in `out_dir`.

    Returns the output path. Idempotent: skips if the file already exists,
    unless `force=True`. Raises RenderError on failure, out-of-range pages,
    or pages whose embedded-image count is not exactly 1.
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

    # `pdfimages -f N -l N -png in.pdf <root>` writes one file per embedded
    # image on page N, named `<root>-NNN.png` where NNN is a 0-based serial
    # number reset to 0 for each invocation. We extract into a per-call
    # tempdir so the produced file is unambiguous, then move it to `target`.
    with tempfile.TemporaryDirectory(prefix="flowsheet_extract_", dir=out_dir) as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            "pdfimages",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-png",
            str(pdf_path),
            str(tmp_dir / "page"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RenderError(
                f"pdfimages failed for {pdf_path} page {page_number}: {result.stderr.strip()}"
            )

        produced = sorted(tmp_dir.glob("page-*.png"))
        if len(produced) == 0:
            raise RenderError(
                f"page {page_number} of {pdf_path} has no embedded image; "
                "phase-1 assumes exactly one image per page"
            )
        if len(produced) > 1:
            raise RenderError(
                f"page {page_number} of {pdf_path} has {len(produced)} embedded images; "
                "phase-1 assumes exactly one image per page"
            )

        # Rotate 180° before writing to `target`. See module docstring.
        with Image.open(produced[0]) as img:
            img.load()
            rotated = img.rotate(180, expand=False)
        if target.exists():
            target.unlink()
        rotated.save(target, format="PNG", optimize=True)
    return target
