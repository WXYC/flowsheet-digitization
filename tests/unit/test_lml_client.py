"""Tests for `core.lml_client.LMLClient`.

Strategy: dependency-inject an httpx transport so tests never touch the
network. `httpx.MockTransport` accepts an async handler that receives the
outgoing request and returns a synthetic `httpx.Response`. The handler
captures call arguments (URL, headers, JSON body) for assertions.

Mirrors the testing pattern in `tests/unit/test_gemini.py`: inject at the
SDK boundary, not at a thin wrapper layer, so the test exercises the same
serialization path production runs through.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from core.lml_client import (
    LMLClient,
    LMLError,
)

# -- Transport helpers -------------------------------------------------------


def _ok_bulk_response(items: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"results": items},
    )


def _bulk_handler(
    captured: list[httpx.Request],
    *,
    response: httpx.Response | None = None,
    response_factory: Any = None,
) -> Any:
    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if response_factory is not None:
            return response_factory(request)
        assert response is not None
        return response

    return handler


def _make_client(handler: Any, *, base_url: str = "https://lml.test", **kwargs: Any) -> LMLClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url=base_url)
    return LMLClient(http=http, **kwargs)


# -- Tests -------------------------------------------------------------------


class TestLMLClientBulk:
    async def test_posts_to_bulk_endpoint_with_json_body(self) -> None:
        captured: list[httpx.Request] = []
        handler = _bulk_handler(
            captured,
            response=_ok_bulk_response(
                [
                    {"index": 0, "status": "match", "lookup": {"corrected_artist": "Stereolab"}},
                ]
            ),
        )

        async with _make_client(handler) as client:
            results = await client.bulk_lookup([{"artist": "stereo lab"}])

        assert len(captured) == 1
        req = captured[0]
        assert req.method == "POST"
        assert req.url.path == "/api/v1/lookup/bulk"
        body = json.loads(req.content)
        assert body["items"][0]["artist"] == "stereo lab"
        # Returns one result per request item, preserving order.
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].status == "match"
        assert results[0].corrected_artist == "Stereolab"

    async def test_sends_bearer_token_when_configured(self) -> None:
        captured: list[httpx.Request] = []
        handler = _bulk_handler(captured, response=_ok_bulk_response([]))
        async with _make_client(handler, api_key="secret-token") as client:
            await client.bulk_lookup([{"artist": "x"}])

        assert captured[0].headers.get("authorization") == "Bearer secret-token"

    async def test_omits_authorization_header_when_no_key(self) -> None:
        captured: list[httpx.Request] = []
        handler = _bulk_handler(captured, response=_ok_bulk_response([]))
        async with _make_client(handler, api_key=None) as client:
            await client.bulk_lookup([{"artist": "x"}])

        # httpx still sends a default Accept etc., but not Authorization.
        assert "authorization" not in {k.lower() for k in captured[0].headers.keys()}

    async def test_parses_corrected_artist_when_present(self) -> None:
        handler = _bulk_handler(
            [],
            response=_ok_bulk_response(
                [
                    {
                        "index": 0,
                        "status": "match",
                        "lookup": {"corrected_artist": "Sigur Rós", "results": []},
                    }
                ]
            ),
        )
        async with _make_client(handler) as client:
            results = await client.bulk_lookup([{"artist": "sigur ros"}])

        assert results[0].corrected_artist == "Sigur Rós"

    async def test_handles_null_corrected_artist(self) -> None:
        """LML returns `corrected_artist: null` when no fuzzy correction
        applied (or there was no match at all). The client must surface
        that as None, not raise."""
        handler = _bulk_handler(
            [],
            response=_ok_bulk_response(
                [
                    {"index": 0, "status": "no_match", "lookup": {"corrected_artist": None}},
                ]
            ),
        )
        async with _make_client(handler) as client:
            results = await client.bulk_lookup([{"artist": "asdfqwerty"}])

        assert results[0].corrected_artist is None
        assert results[0].status == "no_match"

    async def test_handles_error_items_without_lookup(self) -> None:
        """LML's bulk endpoint surfaces per-item failures as status='error'
        with `lookup: None`. The client surfaces that as a result with
        corrected_artist=None, never raises."""
        handler = _bulk_handler(
            [],
            response=_ok_bulk_response(
                [
                    {"index": 0, "status": "error", "lookup": None, "message": "boom"},
                ]
            ),
        )
        async with _make_client(handler) as client:
            results = await client.bulk_lookup([{"artist": "x"}])

        assert results[0].status == "error"
        assert results[0].corrected_artist is None
        assert results[0].message == "boom"

    async def test_raises_on_5xx(self) -> None:
        """A 5xx on a batch is upstream's problem. The client surfaces it as
        LMLError so the caller can decide policy (skip vs abort). It does
        NOT silently return an empty list — that would be a silent
        miscorrection (per the project data-safety rule)."""

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=503, text="overloaded")

        async with _make_client(handler) as client:
            with pytest.raises(LMLError):
                await client.bulk_lookup([{"artist": "x"}])

    async def test_raises_on_malformed_response(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code=200, text="not json")

        async with _make_client(handler) as client:
            with pytest.raises(LMLError):
                await client.bulk_lookup([{"artist": "x"}])

    async def test_splits_batches_above_cap(self) -> None:
        """LML caps each request at 100 items. The client must chunk and
        merge transparently — caller passes whatever it has, gets one
        result per item in input order."""
        captured: list[httpx.Request] = []

        def make_response(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            n = len(body["items"])
            return httpx.Response(
                status_code=200,
                json={
                    "results": [
                        {"index": i, "status": "match", "lookup": {"corrected_artist": f"A{i}"}}
                        for i in range(n)
                    ]
                },
            )

        handler = _bulk_handler(captured, response_factory=make_response)

        async with _make_client(handler, batch_size=100) as client:
            items = [{"artist": f"x{i}"} for i in range(250)]
            results = await client.bulk_lookup(items)

        assert len(captured) == 3  # 100 + 100 + 50
        # Each batch carries no more than the cap.
        for req in captured:
            n = len(json.loads(req.content)["items"])
            assert n <= 100
        # One result per input, in input order.
        assert len(results) == 250

    async def test_handles_empty_input(self) -> None:
        """Calling bulk_lookup with an empty list must not hit the wire."""
        captured: list[httpx.Request] = []
        handler = _bulk_handler(captured, response=_ok_bulk_response([]))
        async with _make_client(handler) as client:
            results = await client.bulk_lookup([])
        assert results == []
        assert captured == []  # no roundtrips

    async def test_auto_fills_raw_message_from_artist(self) -> None:
        """LML's live `LookupRequest` accepts items with `raw_message`
        absent, but its internal `ParsedRequest` model rejects `null`
        and 422s the whole batch. The client must auto-fill
        `raw_message` from the artist/song fields so callers can pass
        the documented shape (`{"artist": "..."}`) and get a working
        request. Mirrors the upstream-bug workaround noted in
        `core.lml_client`."""
        captured: list[httpx.Request] = []
        handler = _bulk_handler(captured, response=_ok_bulk_response([]))
        async with _make_client(handler) as client:
            await client.bulk_lookup([{"artist": "Stereolab"}])

        body = json.loads(captured[0].content)
        assert body["items"][0]["artist"] == "Stereolab"
        # raw_message synthesized — non-empty, includes the artist.
        assert "Stereolab" in body["items"][0]["raw_message"]

    async def test_preserves_caller_supplied_raw_message(self) -> None:
        """When the caller already set `raw_message`, the client does
        NOT overwrite it."""
        captured: list[httpx.Request] = []
        handler = _bulk_handler(captured, response=_ok_bulk_response([]))
        async with _make_client(handler) as client:
            await client.bulk_lookup(
                [{"artist": "Stereolab", "raw_message": "play me some stereolab"}]
            )
        body = json.loads(captured[0].content)
        assert body["items"][0]["raw_message"] == "play me some stereolab"

    async def test_caps_batch_concurrency(self) -> None:
        """Documented bulk-endpoint server-side cap is 10 concurrent; the
        client caps at 5 to stay polite. We verify by counting peak
        in-flight requests."""
        import asyncio

        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                # Hold long enough that other batches accumulate behind the cap.
                await asyncio.sleep(0.05)
                body = json.loads(request.content)
                n = len(body["items"])
                return httpx.Response(
                    status_code=200,
                    json={
                        "results": [
                            {
                                "index": i,
                                "status": "match",
                                "lookup": {"corrected_artist": "x"},
                            }
                            for i in range(n)
                        ]
                    },
                )
            finally:
                async with lock:
                    in_flight -= 1

        async with _make_client(
            handler,
            batch_size=10,
            max_concurrent_batches=5,
        ) as client:
            # 20 batches of 10 = 200 items, more than the concurrency cap.
            items = [{"artist": f"x{i}"} for i in range(200)]
            await client.bulk_lookup(items)

        assert peak <= 5, f"peak concurrency {peak} exceeded cap"
