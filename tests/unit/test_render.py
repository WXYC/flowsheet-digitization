"""Tests for PDF→PNG rendering via pdftoppm."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from core.render import RenderError, count_pages, image_path_for, render_page

POPPLER_AVAILABLE = shutil.which("pdftoppm") is not None and shutil.which("pdfinfo") is not None


def _make_blank_pdf(path: Path, n_pages: int = 2) -> None:
    """Generate a tiny multi-page PDF using pdf_compose-free tooling.

    We write a minimal hand-rolled PDF rather than depend on reportlab. The
    PDF only needs to be parseable by pdfinfo + rasterizable by pdftoppm.
    """
    # Minimal PDF: header, n blank Page objects, xref, trailer.
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    objects: list[bytes] = []

    def add(obj_id: int, body: bytes) -> None:
        objects.append(f"{obj_id} 0 obj\n".encode() + body + b"\nendobj\n")

    # Object 1: Catalog
    add(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    # Object 2: Pages tree
    kids = " ".join(f"{3 + i} 0 R" for i in range(n_pages)).encode()
    add(2, b"<< /Type /Pages /Count " + str(n_pages).encode() + b" /Kids [ " + kids + b" ] >>")
    # Pages
    for i in range(n_pages):
        add(
            3 + i,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << >> /Contents [] >>",
        )

    # Assemble with xref
    out = bytearray(header)
    offsets = [0]
    for obj_bytes in objects:
        offsets.append(len(out))
        out.extend(obj_bytes)
    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(out))


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "sample.pdf"
    _make_blank_pdf(pdf, n_pages=3)
    # Sanity: pdfinfo can read it.
    if POPPLER_AVAILABLE:
        result = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
    return pdf


def test_image_path_for_uses_zero_padded_page_number(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    p = image_path_for(out_dir, page_number=4)
    assert p == out_dir / "page-04.png"


def test_image_path_for_pads_three_digits(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    p = image_path_for(out_dir, page_number=128)
    assert p == out_dir / "page-128.png"


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
def test_count_pages(sample_pdf: Path) -> None:
    assert count_pages(sample_pdf) == 3


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
def test_render_page_writes_file(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    result = render_page(sample_pdf, page_number=1, out_dir=out, dpi=72)
    assert result == out / "page-01.png"
    assert result.exists()
    assert result.stat().st_size > 0


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
def test_render_page_is_idempotent(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    first = render_page(sample_pdf, page_number=1, out_dir=out, dpi=72)
    mtime1 = first.stat().st_mtime_ns
    second = render_page(sample_pdf, page_number=1, out_dir=out, dpi=72)
    assert second == first
    # Skipped re-render: mtime unchanged.
    assert second.stat().st_mtime_ns == mtime1


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
def test_render_page_force_overwrites(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    first = render_page(sample_pdf, page_number=1, out_dir=out, dpi=72)
    mtime1 = first.stat().st_mtime_ns
    # Tiny sleep substitute: bump the file's mtime backward so any rewrite is detectable.
    import os

    os.utime(first, ns=(mtime1 - 1_000_000_000, mtime1 - 1_000_000_000))
    second = render_page(sample_pdf, page_number=1, out_dir=out, dpi=72, force=True)
    assert second.stat().st_mtime_ns > mtime1 - 1_000_000_000


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdftoppm/pdfinfo not installed")
def test_render_page_rejects_out_of_range(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    with pytest.raises(RenderError):
        render_page(sample_pdf, page_number=99, out_dir=out, dpi=72)


def test_render_page_rejects_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(RenderError):
        render_page(tmp_path / "nope.pdf", page_number=1, out_dir=tmp_path / "out", dpi=72)
