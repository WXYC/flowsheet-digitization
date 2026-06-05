"""Typer CLI for the flowsheet digitization pipeline.

Subcommands map 1:1 to phases of the ETL:

    flowsheets discover       → walk SCANS_ROOT, register one job per page
    flowsheets render         → render pending pages to PNGs
    flowsheets process        → send rendered pages to Gemini
    flowsheets status         → counts by status
    flowsheets retry-page     → explicit retry of a single page

Each command builds its dependencies from environment variables (see
.env.example) and calls into core.pipeline / core.jobs.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from core.gemini import GeminiClient, MediaResolution
from core.jobs import JobError, JobStatus, JobStore
from core.lml_client import DEFAULT_LML_URL, LMLClient
from core.pipeline import discover_pdfs, process_pending, render_pending
from core.reconciliation import FlaggedRow, reconcile
from core.schema import PageResult

# Tuned offline against the 19-page verified corpus by
# `scripts/tune_reconciliation_threshold.py`. The smallest T where
# (Gemini->LML correction) achieves precision >=95% against Alex's
# verified.json. See the PR body for the precision/recall curve.
DEFAULT_RECONCILIATION_THRESHOLD = 90

app = typer.Typer(
    add_completion=False,
    help="WXYC flowsheet OCR pipeline (Gemini 3).",
    no_args_is_help=True,
)
console = Console()


# -- Env / dependency construction -----------------------------------------


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def _scans_root() -> Path:
    return _env_path("SCANS_ROOT", "./scans")


def _data_root() -> Path:
    return _env_path("DATA_ROOT", "./data")


def _db_path() -> Path:
    return _data_root() / "jobs.db"


async def _init_store() -> JobStore:
    store = JobStore(_db_path())
    await store.init()
    return store


def _build_lml_client() -> LMLClient:
    """Construct an LMLClient from env. Patched out by unit tests.

    Reads:
      * `LML_URL` — base URL. Defaults to the deployed production service.
      * `LML_API_KEY` — bearer token. Optional; LML enforces auth only
        when `LML_REQUIRE_AUTH=true` on the server.
    """
    import httpx

    base_url = os.environ.get("LML_URL", DEFAULT_LML_URL)
    api_key = os.environ.get("LML_API_KEY") or None
    http = httpx.AsyncClient(base_url=base_url, timeout=60)
    return LMLClient(http=http, api_key=api_key)


def _build_gemini_client() -> GeminiClient:
    """Construct a real Gemini client from env. Patched out by unit tests."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise typer.BadParameter(
            "GEMINI_API_KEY is not set. Add it to .env or your shell environment."
        )
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
    media_resolution = MediaResolution.from_string(
        os.environ.get("GEMINI_MEDIA_RESOLUTION", "high")
    )
    # Imported lazily so tests that don't exercise this path don't pay the
    # cost of the SDK's network-capable Client class.
    from google import genai

    return GeminiClient(
        sdk=genai.Client(api_key=api_key),
        model=model,
        media_resolution=media_resolution,
    )


# -- Commands --------------------------------------------------------------


@app.callback()
def _root() -> None:
    """Load .env once for the entire CLI invocation."""
    load_dotenv(override=False)


@app.command()
def discover() -> None:
    """Walk SCANS_ROOT and register one pending job per PDF page."""
    n = asyncio.run(_discover())
    console.print(f"Registered [bold]{n}[/bold] new pages.")


async def _discover() -> int:
    store = await _init_store()
    return await discover_pdfs(store, scans_root=_scans_root())


@app.command()
def render(
    limit: Annotated[int, typer.Option(help="Max pages to render this run.")] = 200,
    concurrency: Annotated[
        int | None,
        typer.Option(
            help="Parallel render workers. Defaults to RENDER_CONCURRENCY env var, then 4.",
        ),
    ] = None,
) -> None:
    """Render pending pages to PNGs under DATA_ROOT/pages."""
    effective = (
        concurrency if concurrency is not None else int(os.environ.get("RENDER_CONCURRENCY", "4"))
    )
    n = asyncio.run(_render(limit=limit, concurrency=effective))
    console.print(f"Rendered [bold]{n}[/bold] pages (concurrency={effective}).")


async def _render(limit: int, concurrency: int) -> int:
    store = await _init_store()
    return await render_pending(
        store,
        scans_root=_scans_root(),
        data_root=_data_root(),
        limit=limit,
        concurrency=concurrency,
    )


@app.command()
def process(
    limit: Annotated[int, typer.Option(help="Max pages to process this run.")] = 50,
    concurrency: Annotated[
        int | None,
        typer.Option(
            help="Parallel Gemini calls. Defaults to PROCESS_CONCURRENCY env var, then 4. "
            "Tune to fit your per-minute rate limit.",
        ),
    ] = None,
) -> None:
    """Send rendered pages to Gemini and store JSON results."""
    effective = (
        concurrency if concurrency is not None else int(os.environ.get("PROCESS_CONCURRENCY", "4"))
    )
    n = asyncio.run(_process(limit=limit, concurrency=effective))
    console.print(f"Processed [bold]{n}[/bold] pages (concurrency={effective}).")


async def _process(limit: int, concurrency: int) -> int:
    store = await _init_store()
    client = _build_gemini_client()
    max_attempts = int(os.environ.get("MAX_ATTEMPTS", "3"))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        task_id = progress.add_task("Processing", total=limit)

        def on_complete(pdf_path: str, page_number: int, success: bool) -> None:
            mark = "ok" if success else "FAIL"
            progress.console.log(f"[{mark}] {pdf_path} page {page_number}")
            progress.advance(task_id)

        n = await process_pending(
            store,
            client=client,
            data_root=_data_root(),
            limit=limit,
            max_attempts=max_attempts,
            concurrency=concurrency,
            on_complete=on_complete,
        )
    return n


@app.command()
def status() -> None:
    """Show job counts by status."""
    counts = asyncio.run(_status())
    if not counts:
        console.print("[dim]No jobs yet. Run `flowsheets discover` to get started.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for s, n in sorted(counts.items()):
        table.add_row(str(s), str(n))
    console.print(table)


async def _status() -> dict[JobStatus, int]:
    store = await _init_store()
    return await store.counts_by_status()


@app.command("retry-page")
def retry_page(
    pdf_path: Annotated[str, typer.Argument(help="PDF path relative to SCANS_ROOT.")],
    page_number: Annotated[int, typer.Argument(help="1-based page number.")],
) -> None:
    """Reset a completed/low-confidence page to `rendered` so it gets reprocessed.

    Requires that the page was previously rendered (image_path on record).
    Use this when you know a specific page's extraction was wrong and you
    want to try again — never use it as a bulk reset.
    """
    try:
        asyncio.run(_retry_page(pdf_path, page_number))
    except JobError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Reset {pdf_path} page {page_number} for reprocessing.")


async def _retry_page(pdf_path: str, page_number: int) -> None:
    store = await _init_store()
    await store.retry(pdf_path, page_number)


@app.command("reconcile-page")
def reconcile_page(
    result_path: Annotated[
        Path,
        typer.Argument(
            help="Path to a PageResult JSON (Gemini's on-disk output).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Where to write the corrected PageResult. Defaults to "
                "`<input>.reconciled.json` next to the source."
            ),
        ),
    ] = None,
    threshold: Annotated[
        int,
        typer.Option(
            "--threshold",
            help=(
                "Minimum token_set_ratio (0-100) between LML's "
                "corrected_artist and the original to auto-accept the "
                "correction. Tune via "
                "`scripts/tune_reconciliation_threshold.py`."
            ),
            min=0,
            max=100,
        ),
    ] = DEFAULT_RECONCILIATION_THRESHOLD,
) -> None:
    """Reconcile a single page's transcribed artists against the WXYC library.

    Reads the on-disk PageResult JSON, calls LML's bulk lookup for each
    row that parses to an artist, applies LML's `corrected_artist`
    when its similarity to the Gemini-emitted artist is at or above
    `threshold`, and writes the corrected page to `--out` (or
    `<input>.reconciled.json`). Below-threshold suggestions are
    written to a sibling `<out-stem>.flagged.json` for human review.

    Track text and every non-`raw_text` Entry field is preserved
    byte-identical. Page metadata (`page_date_raw`, `comments_raw`,
    page-level `oddities`, `model_version`, `extracted_at`) survives
    unchanged.
    """
    out_path = out if out is not None else result_path.with_suffix(".reconciled.json")
    n_corrections, n_flagged = asyncio.run(
        _reconcile_page(result_path, out_path=out_path, threshold=threshold)
    )
    console.print(
        f"Reconciled [bold]{result_path.name}[/bold]: "
        f"{n_corrections} corrections applied, {n_flagged} flagged for review."
    )
    console.print(f"  corrected -> {out_path}")
    if n_flagged:
        console.print(f"  flagged   -> {out_path.with_suffix('.flagged.json')}")


async def _reconcile_page(result_path: Path, *, out_path: Path, threshold: int) -> tuple[int, int]:
    page = PageResult.model_validate_json(result_path.read_text())
    async with _build_lml_client() as lml:
        corrected, flagged = await reconcile(page, lml=lml, threshold=threshold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corrected.model_dump_json(indent=2))
    if flagged:
        flagged_path = out_path.with_suffix(".flagged.json")
        flagged_path.write_text(
            json.dumps(
                [_flagged_to_dict(f) for f in flagged],
                indent=2,
            )
        )
    return _diff_count(page, corrected), len(flagged)


def _flagged_to_dict(flag: FlaggedRow) -> dict[str, object]:
    """Convert a FlaggedRow dataclass to a stable JSON-friendly dict.

    Explicit projection (instead of `dataclasses.asdict`) so the on-disk
    JSON contract is reviewable here when `FlaggedRow` grows fields.
    """
    return {
        "quadrant": flag.quadrant,
        "row_index": flag.row_index,
        "original_artist": flag.original_artist,
        "suggested_artist": flag.suggested_artist,
        "score": flag.score,
        "raw_text": flag.raw_text,
    }


def _diff_count(before: PageResult, after: PageResult) -> int:
    n = 0
    for b, a in zip(before.quadrants, after.quadrants, strict=True):
        for be, ae in zip(b.entries, a.entries, strict=True):
            if be.raw_text != ae.raw_text:
                n += 1
    return n


if __name__ == "__main__":
    app()
