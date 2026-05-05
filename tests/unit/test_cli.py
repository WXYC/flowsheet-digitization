"""Tests for the Typer CLI.

Each command is tested by mocking out the underlying pipeline function it
delegates to. This keeps CLI tests focused on argument parsing, exit codes,
and output rendering.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

import cli
from core.jobs import JobStatus

runner = CliRunner()


@pytest.fixture
def stub_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI at a tmp dir for SCANS_ROOT/DATA_ROOT and a stub API key."""
    scans = tmp_path / "scans"
    data = tmp_path / "data"
    scans.mkdir()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("SCANS_ROOT", str(scans))
    monkeypatch.setenv("DATA_ROOT", str(data))
    return tmp_path


def test_status_prints_counts_by_status(stub_env: Path) -> None:
    fake_counts = AsyncMock(return_value={JobStatus.PENDING: 4, JobStatus.COMPLETED: 1})
    with (
        patch.object(
            cli, "_init_store", new=AsyncMock(return_value=_MockStore(counts=fake_counts))
        ),
    ):
        result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0
    assert "pending" in result.stdout
    assert "4" in result.stdout
    assert "completed" in result.stdout
    assert "1" in result.stdout


def test_discover_invokes_pipeline_and_reports_count(stub_env: Path) -> None:
    discover_mock = AsyncMock(return_value=12)
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "discover_pdfs", new=discover_mock),
    ):
        result = runner.invoke(cli.app, ["discover"])
    assert result.exit_code == 0
    assert "12" in result.stdout
    discover_mock.assert_awaited_once()


def test_render_invokes_pipeline_and_reports_count(stub_env: Path) -> None:
    render_mock = AsyncMock(return_value=7)
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "render_pending", new=render_mock),
    ):
        result = runner.invoke(cli.app, ["render", "--limit", "100"])
    assert result.exit_code == 0
    assert "7" in result.stdout
    kwargs = render_mock.await_args.kwargs
    assert kwargs["limit"] == 100


def test_process_invokes_pipeline_and_reports_count(stub_env: Path) -> None:
    process_mock = AsyncMock(return_value=3)
    # Bypass real genai.Client construction.
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "process_pending", new=process_mock),
        patch.object(cli, "_build_gemini_client", return_value=object()),
    ):
        result = runner.invoke(cli.app, ["process", "--limit", "50"])
    assert result.exit_code == 0
    assert "3" in result.stdout
    kwargs = process_mock.await_args.kwargs
    assert kwargs["limit"] == 50


def test_retry_page_calls_store_retry(stub_env: Path) -> None:
    store = _MockStore()
    with patch.object(cli, "_init_store", new=AsyncMock(return_value=store)):
        result = runner.invoke(cli.app, ["retry-page", "1990/January 1990/foo.pdf", "5"])
    assert result.exit_code == 0
    assert store.retry_called_with == ("1990/January 1990/foo.pdf", 5)


def test_retry_page_unknown_exits_nonzero(stub_env: Path) -> None:
    store = _MockStore(retry_raises=True)
    with patch.object(cli, "_init_store", new=AsyncMock(return_value=store)):
        result = runner.invoke(cli.app, ["retry-page", "scans/nope.pdf", "1"])
    assert result.exit_code != 0


# --- Test helpers -----------------------------------------------------------


class _MockStore:
    """Minimal JobStore stand-in used by CLI tests.

    Only the methods CLI commands call are implemented. Anything else is left
    as an attribute the test can set up explicitly.
    """

    def __init__(
        self,
        *,
        counts: AsyncMock | None = None,
        retry_raises: bool = False,
    ) -> None:
        self._counts = counts or AsyncMock(return_value={})
        self._retry_raises = retry_raises
        self.retry_called_with: tuple[str, int] | None = None

    async def counts_by_status(self) -> dict:
        return await self._counts()

    async def retry(self, pdf_path: str, page_number: int) -> None:
        from core.jobs import JobError

        if self._retry_raises:
            raise JobError("nope")
        self.retry_called_with = (pdf_path, page_number)
