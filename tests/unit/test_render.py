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


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_render_page_rotates_180_after_extraction(
    sample_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WXYC PDFs draw their embedded bitmap with a flipping transform.

    `pdfimages` emits the raw stored bitmap (upside-down) while `pdftoppm`
    honors the page's content-stream transform. Since the pipeline uses
    `pdfimages` for losslessness, we rotate 180° after extraction so the
    saved PNG matches the page's intended orientation.
    """
    from PIL import Image

    rotation_angles: list[float] = []
    real_rotate = Image.Image.rotate

    def spy_rotate(self: Image.Image, angle: float, *args: object, **kwargs: object) -> Image.Image:
        rotation_angles.append(angle)
        return real_rotate(self, angle, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "rotate", spy_rotate)

    out = tmp_path / "rendered"
    render_page(sample_pdf, page_number=1, out_dir=out)

    assert 180 in rotation_angles, (
        f"render_page must rotate 180° after pdfimages extraction; "
        f"observed rotation angles: {rotation_angles}"
    )


@pytest.mark.skipif(not POPPLER_AVAILABLE, reason="pdfimages/pdfinfo not installed")
def test_render_page_output_dimensions_preserved(sample_pdf: Path, tmp_path: Path) -> None:
    """180° rotation preserves image dimensions (width/height swap is for 90°)."""
    from PIL import Image

    out = tmp_path / "rendered"
    result = render_page(sample_pdf, page_number=1, out_dir=out)

    with Image.open(result) as img:
        assert img.size == (50, 50)  # the test fixture is 50x50
