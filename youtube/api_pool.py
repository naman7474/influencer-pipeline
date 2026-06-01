"""Multi-key YouTube Data API rotator with quota failover.

Each key carries a 10k-unit/day default quota. ``YouTubeAPIPool`` holds N
``YouTubeAPIClient`` instances and routes calls round-robin. When one key
hits ``quotaExceeded`` (raised by ``_execute_with_retry`` as
``QuotaExceededError``), the pool marks that key exhausted-for-the-day and
retries the same call against the next key. When every key is marked
exhausted the pool raises so the caller can give up.

This unblocks running batches that exceed any single key's daily budget
(e.g. 10k channels × ~10 units = 100k units = 10+ keys at default quota,
or fewer keys if one has a quota increase).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Iterable, Optional

from pipeline.youtube.youtube_api import (
    QuotaExceededError,
    YouTubeAPIClient,
)

logger = logging.getLogger(__name__)


class AllKeysExhaustedError(Exception):
    """All YT API keys in the pool have hit their daily quota."""


def _split_keys(raw: str) -> list[str]:
    """Parse a comma- or whitespace-separated list of API keys."""
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        trimmed = chunk.strip()
        if trimmed:
            parts.append(trimmed)
    return parts


def load_api_keys_from_env() -> list[str]:
    """Resolve the active set of YT API keys from environment.

    Order of precedence:
      1. ``YOUTUBE_API_KEYS`` (comma- or newline-separated list) — pool mode
      2. ``YOUTUBE_API_KEY_1``, ``YOUTUBE_API_KEY_2``, ... (numbered keys)
      3. ``YOUTUBE_API_KEY`` (single, legacy) — fallback
    Duplicates are removed while preserving first-seen order.
    """
    seen: set[str] = set()
    keys: list[str] = []

    for k in _split_keys(os.environ.get("YOUTUBE_API_KEYS", "")):
        if k not in seen:
            seen.add(k)
            keys.append(k)

    # Numbered fallback (1..32 is plenty)
    for n in range(1, 33):
        v = os.environ.get(f"YOUTUBE_API_KEY_{n}")
        if v and v.strip() and v.strip() not in seen:
            seen.add(v.strip())
            keys.append(v.strip())

    legacy = os.environ.get("YOUTUBE_API_KEY")
    if legacy and legacy.strip() and legacy.strip() not in seen:
        seen.add(legacy.strip())
        keys.append(legacy.strip())

    return keys


class YouTubeAPIPool:
    """A pool of ``YouTubeAPIClient`` rotating across multiple API keys.

    Public surface mirrors ``YouTubeAPIClient`` so callers don't need to
    know they're talking to a pool. Add new methods here as needed; each
    one should call ``self._call("method_name", *args, **kwargs)`` to
    inherit the rotation + failover.
    """

    def __init__(self, api_keys: Optional[Iterable[str]] = None):
        if api_keys is None:
            api_keys = load_api_keys_from_env()
        keys = [k for k in (api_keys or []) if k]
        # Build clients eagerly; each YouTubeAPIClient's actual `httplib2`
        # service is built lazily per-thread, so no cost here.
        self._clients: list[YouTubeAPIClient] = [
            YouTubeAPIClient(api_key=k) for k in keys
        ]
        self._lock = threading.Lock()
        self._cursor = 0
        self._exhausted: set[int] = set()

        if not self._clients:
            logger.warning("YouTubeAPIPool initialised with zero API keys")
        else:
            logger.info(
                "YouTubeAPIPool initialised with %d API key(s)",
                len(self._clients),
            )

    # ── Compatibility surface ─────────────────────────────────────

    @property
    def available(self) -> bool:
        return any(
            c.available
            for i, c in enumerate(self._clients)
            if i not in self._exhausted
        )

    @property
    def num_keys(self) -> int:
        return len(self._clients)

    @property
    def num_exhausted(self) -> int:
        return len(self._exhausted)

    # ── Internals ─────────────────────────────────────────────────

    def _next_client(self) -> Optional[YouTubeAPIClient]:
        with self._lock:
            if not self._clients:
                return None
            for _ in range(len(self._clients)):
                idx = self._cursor % len(self._clients)
                self._cursor += 1
                if idx in self._exhausted:
                    continue
                client = self._clients[idx]
                if client.available:
                    return client
            return None

    def _mark_exhausted(self, client: YouTubeAPIClient) -> None:
        with self._lock:
            for i, c in enumerate(self._clients):
                if c is client and i not in self._exhausted:
                    self._exhausted.add(i)
                    remaining = len(self._clients) - len(self._exhausted)
                    logger.warning(
                        "YT API key #%d quota exhausted; %d key(s) remain",
                        i + 1,
                        remaining,
                    )
                    return

    def _call(self, method_name: str, *args, **kwargs):
        last_quota_err: Optional[QuotaExceededError] = None
        # Try every still-active key once before giving up.
        attempts = max(1, len(self._clients))
        for _ in range(attempts):
            client = self._next_client()
            if client is None:
                break
            try:
                return getattr(client, method_name)(*args, **kwargs)
            except QuotaExceededError as e:
                self._mark_exhausted(client)
                last_quota_err = e
                continue
        if last_quota_err is not None:
            raise AllKeysExhaustedError(
                f"All {len(self._clients)} YT API keys hit daily quota"
            ) from last_quota_err
        # No keys at all (or none available)
        raise AllKeysExhaustedError(
            "YouTubeAPIPool has no usable keys. Set YOUTUBE_API_KEYS or YOUTUBE_API_KEY."
        )

    # ── Forwarders mirroring YouTubeAPIClient's public surface ────

    def resolve_handle_to_channel_id(self, handle: str) -> Optional[str]:
        return self._call("resolve_handle_to_channel_id", handle)

    def fetch_channel_stats(
        self, channel_ids: Iterable[str]
    ) -> dict[str, dict]:
        # `channel_ids` is iterable; coerce to a stable list once so we
        # don't re-iterate a generator after a quota retry.
        ids = list(channel_ids)
        return self._call("fetch_channel_stats", ids)

    def fetch_video_stats(self, video_ids: Iterable[str]) -> dict[str, dict]:
        return self._call("fetch_video_stats", list(video_ids))

    def list_channel_uploads(
        self, channel_id: str, limit: int = 20
    ) -> list[dict]:
        return self._call("list_channel_uploads", channel_id, limit=limit)

    def list_comment_threads(
        self, video_id: str, max_results: int = 100, order: str = "relevance"
    ) -> list[dict]:
        return self._call(
            "list_comment_threads",
            video_id,
            max_results=max_results,
            order=order,
        )

    # ── Keyword search (Phase 3 — on-demand discovery) ──────────

    def search_keyword_channels(
        self,
        query: str,
        max_results: int = 200,
        region_code: Optional[str] = None,
    ) -> list[dict]:
        """Search YouTube for channels matching `query`. 100 units per page.

        Pool forwarder for `YouTubeAPIClient.search_keyword_channels`.
        Auto-fails-over to the next key when one hits its daily quota.
        """
        return self._call(
            "search_keyword_channels",
            query,
            max_results=max_results,
            region_code=region_code,
        )

    def search_keyword_video_channels(
        self,
        query: str,
        max_videos: int = 200,
        region_code: Optional[str] = None,
    ) -> list[str]:
        """Search YouTube for videos matching `query`, return unique channel
        IDs. 100 units per page. See client method for full docs."""
        return self._call(
            "search_keyword_video_channels",
            query,
            max_videos=max_videos,
            region_code=region_code,
        )
