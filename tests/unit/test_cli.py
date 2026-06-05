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


def test_render_passes_concurrency_flag_to_pipeline(stub_env: Path) -> None:
    render_mock = AsyncMock(return_value=2)
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "render_pending", new=render_mock),
    ):
        result = runner.invoke(cli.app, ["render", "--concurrency", "8"])
    assert result.exit_code == 0
    kwargs = render_mock.await_args.kwargs
    assert kwargs["concurrency"] == 8


def test_render_concurrency_defaults_from_env(
    stub_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RENDER_CONCURRENCY", "6")
    render_mock = AsyncMock(return_value=0)
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "render_pending", new=render_mock),
    ):
        result = runner.invoke(cli.app, ["render"])
    assert result.exit_code == 0
    kwargs = render_mock.await_args.kwargs
    assert kwargs["concurrency"] == 6


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


def test_process_passes_concurrency_flag_to_pipeline(stub_env: Path) -> None:
    process_mock = AsyncMock(return_value=2)
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "process_pending", new=process_mock),
        patch.object(cli, "_build_gemini_client", return_value=object()),
    ):
        result = runner.invoke(cli.app, ["process", "--concurrency", "8"])
    assert result.exit_code == 0
    kwargs = process_mock.await_args.kwargs
    assert kwargs["concurrency"] == 8


def test_process_concurrency_defaults_from_env(
    stub_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROCESS_CONCURRENCY", "5")
    process_mock = AsyncMock(return_value=0)
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "process_pending", new=process_mock),
        patch.object(cli, "_build_gemini_client", return_value=object()),
    ):
        result = runner.invoke(cli.app, ["process"])
    assert result.exit_code == 0
    kwargs = process_mock.await_args.kwargs
    assert kwargs["concurrency"] == 5


def test_process_passes_on_complete_callback(stub_env: Path) -> None:
    """The CLI should pass an on_complete callback so progress is shown."""
    process_mock = AsyncMock(return_value=0)
    with (
        patch.object(cli, "_init_store", new=AsyncMock(return_value=_MockStore())),
        patch.object(cli, "process_pending", new=process_mock),
        patch.object(cli, "_build_gemini_client", return_value=object()),
    ):
        result = runner.invoke(cli.app, ["process", "--limit", "1"])
    assert result.exit_code == 0
    kwargs = process_mock.await_args.kwargs
    assert kwargs.get("on_complete") is not None
    assert callable(kwargs["on_complete"])


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


# --- reconcile-page --------------------------------------------------------


def _write_minimal_page_result(path: Path) -> None:
    """Write a minimal PageResult JSON to `path` for CLI roundtrip tests."""
    import json

    path.write_text(
        json.dumps(
            {
                "page_date_raw": "Sun 4/1/90",
                "comments_raw": None,
                "oddities": [],
                "quadrants": [
                    {
                        "position": "top_left",
                        "hour_raw": "8AM",
                        "jock_raw": "DJ",
                        "entries": [
                            {
                                "row_index": 0,
                                "raw_text": "STEREOLAB - Cybeles",
                                "type_raw": None,
                                "confidence": "high",
                                "notes": None,
                                "oddities": [],
                            }
                        ],
                        "oddities": [],
                    },
                    {
                        "position": "top_right",
                        "hour_raw": None,
                        "jock_raw": None,
                        "entries": [],
                        "oddities": [],
                    },
                    {
                        "position": "bottom_left",
                        "hour_raw": None,
                        "jock_raw": None,
                        "entries": [],
                        "oddities": [],
                    },
                    {
                        "position": "bottom_right",
                        "hour_raw": None,
                        "jock_raw": None,
                        "entries": [],
                        "oddities": [],
                    },
                ],
                "model_version": "gemini-3.1-pro-preview",
                "extracted_at": "2026-05-06T04:11:12.540095+00:00",
            }
        )
    )


def test_reconcile_page_writes_corrected_result(stub_env: Path, tmp_path: Path) -> None:
    """`flowsheets reconcile-page` reads a PageResult JSON, calls
    `reconcile()` with an LML client built from env, and writes the
    corrected JSON to disk."""
    import json

    src = tmp_path / "page.json"
    out = tmp_path / "page.reconciled.json"
    _write_minimal_page_result(src)

    async def stub_reconcile(page, *, lml, threshold):  # type: ignore[no-untyped-def]

        # Rewrite the artist on the single row so we can assert the write happened.
        q = page.quadrants[0]
        new_entry = q.entries[0].model_copy(update={"raw_text": "Stereolab - Cybeles"})
        new_quadrants = [q.model_copy(update={"entries": [new_entry]})] + page.quadrants[1:]
        return page.model_copy(update={"quadrants": new_quadrants}), []

    with (
        patch.object(cli, "_build_lml_client", return_value=_FakeLMLContext()),
        patch.object(cli, "reconcile", new=stub_reconcile),
    ):
        result = runner.invoke(
            cli.app,
            ["reconcile-page", str(src), "--out", str(out), "--threshold", "85"],
        )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text())
    assert payload["quadrants"][0]["entries"][0]["raw_text"] == "Stereolab - Cybeles"
    # Page metadata round-trips.
    assert payload["page_date_raw"] == "Sun 4/1/90"
    assert payload["model_version"] == "gemini-3.1-pro-preview"


def test_reconcile_page_writes_flagged_rows_alongside(stub_env: Path, tmp_path: Path) -> None:
    """Below-threshold rows are written to a sibling `.flagged.json` so a
    human can review without parsing the corrected result."""
    import json

    src = tmp_path / "page.json"
    out = tmp_path / "page.reconciled.json"
    _write_minimal_page_result(src)

    from core.reconciliation import FlaggedRow

    async def stub_reconcile(page, *, lml, threshold):  # type: ignore[no-untyped-def]
        return page, [
            FlaggedRow(
                quadrant="top_left",
                row_index=0,
                original_artist="STEREOLAB",
                suggested_artist="Stereolob",
                score=70,
                raw_text="STEREOLAB - Cybeles",
            )
        ]

    with (
        patch.object(cli, "_build_lml_client", return_value=_FakeLMLContext()),
        patch.object(cli, "reconcile", new=stub_reconcile),
    ):
        result = runner.invoke(
            cli.app,
            ["reconcile-page", str(src), "--out", str(out)],
        )
    assert result.exit_code == 0, result.stdout
    flagged_path = out.with_suffix(".flagged.json")
    assert flagged_path.exists()
    flagged = json.loads(flagged_path.read_text())
    assert flagged == [
        {
            "quadrant": "top_left",
            "row_index": 0,
            "original_artist": "STEREOLAB",
            "suggested_artist": "Stereolob",
            "score": 70,
            "raw_text": "STEREOLAB - Cybeles",
        }
    ]


def test_reconcile_page_default_out_is_input_with_reconciled_suffix(
    stub_env: Path, tmp_path: Path
) -> None:
    """When `--out` is omitted, the corrected JSON is written next to
    the input as `<stem>.reconciled.json`."""
    src = tmp_path / "page.json"
    _write_minimal_page_result(src)

    async def stub_reconcile(page, *, lml, threshold):  # type: ignore[no-untyped-def]
        return page, []

    with (
        patch.object(cli, "_build_lml_client", return_value=_FakeLMLContext()),
        patch.object(cli, "reconcile", new=stub_reconcile),
    ):
        result = runner.invoke(cli.app, ["reconcile-page", str(src)])
    assert result.exit_code == 0
    expected = src.with_suffix(".reconciled.json")
    assert expected.exists()


# --- Test helpers -----------------------------------------------------------


class _FakeLMLContext:
    """Stand-in for the LMLClient context manager. The CLI uses
    `async with _build_lml_client() as lml:` so we need an async CM."""

    async def __aenter__(self):  # type: ignore[no-untyped-def]
        return self

    async def __aexit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return None

    async def bulk_lookup(self, items):  # type: ignore[no-untyped-def]
        return []


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
