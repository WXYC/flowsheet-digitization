"""Tests for the page-image extraction module.

The implementation uses `pdfimages -png` to pull the embedded raster image
out of each PDF page directly. The fixture PDF
`tests/fixtures/three_pages_with_images.pdf` is a 3-page PDF where each
page wraps a single 50x50 CCITT-G4 image — same encoding as the real WXYC
corpus.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.render import RenderError, count_pages, image_path_for, render_page

POPPLER_AVAILABLE = shutil.which("pdfimages") is not None and shutil.which("pdfinfo") is not None

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE_PDF = FIXTURES / "three_pages_with_images.pdf"
SAMPLE_PDF_PAGES = 3


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Return a writable copy of the bundled fixture PDF."""
    dst = tmp_path / "sample.pdf"
    dst.write_bytes(SAMPLE_PDF.read_bytes())
    return dst


def test_image_path_for_uses_zero_padded_page_number(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    p = image_path_for(out_dir, page_number=4)
    assert p == out_dir / "page-04.png"


def test_image_path_for_pads_three_digits(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    p = image_path_for(out_dir, page_number=128)
    assert p == out_dir / "page-128.png"


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_count_pages(sample_pdf: Path) -> None:
    assert count_pages(sample_pdf) == SAMPLE_PDF_PAGES


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_render_page_writes_file(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    result = render_page(sample_pdf, page_number=1, out_dir=out)
    assert result == out / "page-01.png"
    assert result.exists()
    assert result.stat().st_size > 0


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_render_page_extracts_each_page_independently(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    for page in (1, 2, 3):
        result = render_page(sample_pdf, page_number=page, out_dir=out)
        assert result == out / f"page-0{page}.png"
        assert result.exists()


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_render_page_is_idempotent(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    first = render_page(sample_pdf, page_number=1, out_dir=out)
    mtime1 = first.stat().st_mtime_ns
    second = render_page(sample_pdf, page_number=1, out_dir=out)
    assert second == first
    # Skipped re-extract: mtime unchanged.
    assert second.stat().st_mtime_ns == mtime1


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_render_page_force_overwrites(sample_pdf: Path, tmp_path: Path) -> None:
    import os

    out = tmp_path / "rendered"
    first = render_page(sample_pdf, page_number=1, out_dir=out)
    mtime1 = first.stat().st_mtime_ns
    os.utime(first, ns=(mtime1 - 1_000_000_000, mtime1 - 1_000_000_000))
    second = render_page(sample_pdf, page_number=1, out_dir=out, force=True)
    assert second.stat().st_mtime_ns > mtime1 - 1_000_000_000


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_render_page_rejects_out_of_range(sample_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "rendered"
    with pytest.raises(RenderError):
        render_page(sample_pdf, page_number=99, out_dir=out)


def test_render_page_rejects_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(RenderError):
        render_page(tmp_path / "nope.pdf", page_number=1, out_dir=tmp_path / "out")
