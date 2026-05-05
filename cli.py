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
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from core.gemini import GeminiClient, MediaResolution
from core.jobs import JobError, JobStatus, JobStore
from core.pipeline import discover_pdfs, process_pending, render_pending

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
    dpi = int(os.environ.get("RENDER_DPI", "300"))
    return await render_pending(
        store,
        scans_root=_scans_root(),
        data_root=_data_root(),
        dpi=dpi,
        limit=limit,
        concurrency=concurrency,
    )


@app.command()
def process(
    limit: Annotated[int, typer.Option(help="Max pages to process this run.")] = 50,
) -> None:
    """Send rendered pages to Gemini and store JSON results."""
    n = asyncio.run(_process(limit=limit))
    console.print(f"Processed [bold]{n}[/bold] pages.")


async def _process(limit: int) -> int:
    store = await _init_store()
    client = _build_gemini_client()
    max_attempts = int(os.environ.get("MAX_ATTEMPTS", "3"))
    return await process_pending(
        store,
        client=client,
        data_root=_data_root(),
        limit=limit,
        max_attempts=max_attempts,
    )


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


if __name__ == "__main__":
    app()
