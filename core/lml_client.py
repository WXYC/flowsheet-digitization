"""Async client for the WXYC library-metadata-lookup (LML) bulk endpoint.

Only the `/api/v1/lookup/bulk` route is needed for Phase 2 reconciliation;
per-request `/api/v1/lookup` is left to upstream callers and the verifier
UI. Bulk amortizes the cold-cache cost across a page (~10-40 rows) and
keeps the in-process TTL caches warm.

Design notes
------------

* Dependency-injected `httpx.AsyncClient`. Production constructs the
  client with `httpx.AsyncClient(base_url=...)`; tests pass an
  `httpx.MockTransport`. Mirrors `core.gemini.GeminiClient`, which
  injects the google-genai SDK at construction.

* LML's bulk endpoint caps requests at 100 items. Callers don't think
  about this — `bulk_lookup` chunks transparently and merges results in
  input order.

* Server-side concurrency cap is 10 (documented in `lookup/router.py`).
  We cap our client at 5 concurrent batches by default to stay polite
  and leave headroom for other consumers of LML.

* 5xx errors raise `LMLError`. The reconciliation layer's policy is
  "log + skip a failed batch, keep original rows" — that policy lives
  *there*, not here, because this client doesn't know what semantic
  fallback the caller wants.

* The on-wire response carries a lot more than `corrected_artist` (full
  catalog items, artwork, identity, etc.). For Phase 2 we only need
  `corrected_artist`, the per-item `status`, and the optional error
  `message`. The client surfaces those as `LMLBulkItemResult` and
  discards the rest — keeps this module small and lets us iterate on
  LML's response shape without churn here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Self

import httpx

BulkStatus = Literal["match", "no_match", "error"]

# LML's documented per-request cap. See `lookup/router.py:_BULK_LOOKUP_INPUT_CAP`.
DEFAULT_BATCH_SIZE = 100

# Stay below LML's documented server-side concurrency cap (10) so we leave
# headroom for other consumers.
DEFAULT_MAX_CONCURRENT_BATCHES = 5

# Production default for the deployed service. Override via the LML_URL env
# var when running against a local LML or a staging deploy.
DEFAULT_LML_URL = "https://library-metadata-lookup-production.up.railway.app"


class LMLError(RuntimeError):
    """Raised for any LML failure surfaced upstream of per-item statuses.

    Per-item `status: error` results are NOT raised — they're returned
    in the result list with `corrected_artist=None` and the optional
    `message` filled. `LMLError` is reserved for whole-batch failures
    (5xx, malformed JSON, transport timeout).
    """


@dataclass(frozen=True)
class LMLBulkItemResult:
    """Phase-2 projection of LML's `BulkLookupResultItem`.

    We surface only what reconciliation needs:

    * `index` — 0-based offset into the *overall* `bulk_lookup` input
      list (NOT the per-batch index LML returns, which the client
      rewrites during merge).
    * `status` — `"match"`, `"no_match"`, or `"error"`.
    * `corrected_artist` — LML's fuzzy correction of the input artist,
      or `None` if LML didn't apply one (no match, error, or the input
      was already an exact catalog hit).
    * `message` — populated when `status == "error"`.
    """

    index: int
    status: BulkStatus
    corrected_artist: str | None
    message: str | None = None


class LMLClient:
    """Async client for the LML bulk endpoint.

    Use as an async context manager so the underlying httpx client is
    properly closed:

        async with LMLClient(http=httpx.AsyncClient(base_url=...)) as lml:
            results = await lml.bulk_lookup([{"artist": "Stereolab"}])

    Or pass an already-managed httpx client and skip the context manager.
    """

    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        api_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_concurrent_batches: int = DEFAULT_MAX_CONCURRENT_BATCHES,
    ) -> None:
        if batch_size < 1 or batch_size > DEFAULT_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {DEFAULT_BATCH_SIZE}], got {batch_size}")
        if max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be >= 1")
        self._http = http
        self._api_key = api_key
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def bulk_lookup(self, items: list[dict[str, Any]]) -> list[LMLBulkItemResult]:
        """Look up each item against LML, return one result per item in input order.

        Each `item` is a JSON-serializable mapping that fits LML's
        `LookupRequest` shape — typically `{"artist": "..."}` for Phase 2.
        Empty `items` returns `[]` without hitting the wire.

        Chunked across LML's 100-per-request cap; batches run under a
        semaphore bounded by `max_concurrent_batches`. Per-item `index`
        values in the returned list are 0-based offsets into `items`.

        Items missing `raw_message` get one auto-filled from the
        artist/album/song fields. LML's `LookupRequest` shape declares
        `raw_message` optional, but its internal `ParsedRequest` model
        rejects `null`, so the live endpoint 422s when only structured
        fields are sent. Defaulting `raw_message` here avoids that. See
        the upstream issue noted in `core.lml_client`'s module docs.
        """
        if not items:
            return []

        # Defensive copy + raw_message fill. Don't mutate caller's dicts.
        prepared: list[dict[str, Any]] = []
        for item in items:
            new_item = dict(item)
            if not new_item.get("raw_message"):
                new_item["raw_message"] = _synthesize_raw_message(new_item)
            prepared.append(new_item)
        items = prepared

        batches: list[tuple[int, list[dict[str, Any]]]] = []
        for offset in range(0, len(items), self._batch_size):
            batches.append((offset, items[offset : offset + self._batch_size]))

        async def run_one(offset: int, batch: list[dict[str, Any]]) -> list[LMLBulkItemResult]:
            async with self._semaphore:
                return await self._post_one(offset, batch)

        chunks = await asyncio.gather(*(run_one(o, b) for o, b in batches))
        return [item for chunk in chunks for item in chunk]

    async def _post_one(self, offset: int, batch: list[dict[str, Any]]) -> list[LMLBulkItemResult]:
        headers = {}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {"items": batch}
        try:
            response = await self._http.post(
                "/api/v1/lookup/bulk",
                json=body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise LMLError(f"LML transport error: {exc}") from exc

        if response.status_code >= 500:
            raise LMLError(f"LML returned {response.status_code} for batch of {len(batch)} items")
        if response.status_code >= 400:
            # 4xx is a contract problem (auth, malformed body, over-cap). Surface
            # it loudly — the caller almost certainly mis-built the request.
            raise LMLError(
                f"LML returned {response.status_code} for batch of {len(batch)} items: "
                f"{response.text[:200]}"
            )

        try:
            parsed = response.json()
        except ValueError as exc:
            raise LMLError(f"LML returned non-JSON body: {exc}") from exc

        results_raw = parsed.get("results")
        if not isinstance(results_raw, list):
            raise LMLError(
                f"LML response missing 'results' list (got {type(results_raw).__name__})"
            )

        return [_parse_item(raw, offset) for raw in results_raw]


def _synthesize_raw_message(item: dict[str, Any]) -> str:
    """Compose a `raw_message` string from artist/album/song fields.

    Caller didn't set one, but LML's internal `ParsedRequest` rejects
    `None`. We mimic the request-o-matic shape ("artist - song" / lone
    artist) so any heuristic LML applies to `raw_message` (e.g. format
    detection) sees a plausible input rather than an empty string.
    """
    parts = []
    artist = item.get("artist")
    song = item.get("song")
    album = item.get("album")
    if artist:
        parts.append(str(artist))
    if song:
        parts.append(f"- {song}")
    elif album:
        parts.append(f"- {album}")
    return " ".join(parts) if parts else ""


def _parse_item(raw: Any, offset: int) -> LMLBulkItemResult:
    """Project a single `BulkLookupResultItem` to the trimmed `LMLBulkItemResult`.

    Rewrites the per-batch `index` to a per-overall-call offset so the
    caller of `bulk_lookup` sees one contiguous index space.
    """
    if not isinstance(raw, dict):
        raise LMLError(f"LML response item is not an object: {raw!r}")
    try:
        local_index = int(raw["index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LMLError(f"LML response item missing/invalid 'index': {raw!r}") from exc
    status = raw.get("status")
    if status not in ("match", "no_match", "error"):
        raise LMLError(f"LML response item has unexpected status: {status!r}")
    lookup = raw.get("lookup")
    if lookup is None:
        corrected = None
    elif isinstance(lookup, dict):
        corrected = lookup.get("corrected_artist")
        if corrected is not None and not isinstance(corrected, str):
            raise LMLError(f"LML response item has non-string corrected_artist: {corrected!r}")
    else:
        raise LMLError(f"LML response 'lookup' is not an object: {lookup!r}")
    message = raw.get("message")
    if message is not None and not isinstance(message, str):
        message = str(message)
    return LMLBulkItemResult(
        index=offset + local_index,
        status=status,
        corrected_artist=corrected,
        message=message,
    )
