"""YouTube Data API v3 client — primary source for video / comment / stat data.

After the Phase 2.5 hybrid flip, the API is the *primary* source for:
  - Channel canonical stats (subs, views, videos count, topic categories)
  - Video list + per-video stats (views, likes, comments, duration, tags-if-owner)
  - Top comment threads per video

Bright Data remains the source for:
  - Channel "about" external links (cross-platform stitching signal —
    the API doesn't surface these at all)
  - Caption text fallback when `youtube-transcript-api` is rate-limited
    (`captions.download` is OAuth-as-owner gated, so unusable for us)

Quota accounting (10,000 units/day default):
  - channels.list:           1 unit per call (up to 50 ids per call)
  - videos.list:             1 unit per call (up to 50 ids per call)
  - playlistItems.list:      1 unit per call (up to 50 items per call)
  - commentThreads.list:     1 unit per call
  - search.list:             100 units per call — avoid
"""

from __future__ import annotations

import logging
import os
import socket
import ssl
import threading
import time
from typing import Iterable, Optional

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:  # google-api-python-client not installed
    build = None
    HttpError = Exception

try:
    from httplib2.error import HttpLib2Error  # parent of ServerNotFoundError
except ImportError:
    HttpLib2Error = OSError  # fallback so the except clause stays valid

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when a YT Data API key returns 403 ``quotaExceeded``.

    Distinct from ``HttpError`` so the multi-key pool (``YouTubeAPIPool``)
    can mark the offending key as exhausted and rotate to the next without
    treating it as a hard failure.
    """


def _is_quota_exceeded(err: HttpError) -> bool:
    """Detect the YT Data API daily-quota error.

    The body looks like ``{"error":{"code":403, "errors":[{"reason":"quotaExceeded", ...}]}}``.
    Older clients sometimes return ``"reason":"dailyLimitExceeded"`` for
    the same condition — we treat both as quota.
    """
    status = getattr(getattr(err, "resp", None), "status", 0)
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = 0
    if status != 403:
        return False
    content = getattr(err, "content", b"")
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            content = ""
    text = content or ""
    return "quotaExceeded" in text or "dailyLimitExceeded" in text


def _execute_with_retry(req, max_attempts: int = 6, base_backoff: float = 2.0):
    """Execute a googleapiclient request with retry on transient network errors.

    googleapiclient's built-in retry covers httplib2 errors but not raw
    ``ssl.SSLError`` ("record layer failure" etc.) which surface from
    Python's ``http.client`` underneath. We retry SSL/socket/connection
    errors and 5xx HTTP responses with exponential backoff. 4xx (auth,
    quota, not-found) is non-retryable and propagates immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return req.execute()
        except HttpError as e:
            # Surface daily-quota exhaustion as a typed error so the
            # multi-key pool can rotate without treating it as a real fail.
            if _is_quota_exceeded(e):
                raise QuotaExceededError(str(e)) from e
            status = getattr(getattr(e, "resp", None), "status", 0)
            try:
                status = int(status)
            except (TypeError, ValueError):
                status = 0
            if 500 <= status < 600 and attempt + 1 < max_attempts:
                last_exc = e
                logger.warning(
                    "YT API HTTP %s on attempt %d/%d, retrying",
                    status,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(base_backoff * (2**attempt))
                continue
            raise
        except (
            ssl.SSLError,
            socket.timeout,
            socket.gaierror,
            ConnectionError,
            OSError,
            HttpLib2Error,  # covers httplib2.ServerNotFoundError (DNS)
        ) as e:
            last_exc = e
            if attempt + 1 >= max_attempts:
                raise
            logger.warning(
                "YT API transient %s on attempt %d/%d, retrying",
                type(e).__name__,
                attempt + 1,
                max_attempts,
            )
            time.sleep(base_backoff * (2**attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("YT API retry loop ended unexpectedly")


class YouTubeAPIClient:
    """Thin wrapper over the Data API v3.

    All methods short-circuit if the library isn't installed so the module
    imports cleanly in environments where YouTube support is disabled.

    **Thread safety**: googleapiclient's `build()` returns a service that
    holds a single `httplib2.Http` connection-pool, which is *not*
    thread-safe — concurrent `.execute()` calls corrupt the underlying
    TLS socket and raise ``ssl.SSLError("record layer failure")``. To let
    callers share one ``YouTubeAPIClient`` across a ThreadPoolExecutor we
    build a lazy, thread-local service so each worker gets its own pool.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY")
        # `available` is decided once at construction — depends only on
        # the library being importable and an api key being set.
        self.available = bool(build is not None and self.api_key)
        # Per-thread service cache. Each worker thread builds its own
        # service the first time it touches the API.
        self._tls = threading.local()

    @property
    def _service(self):
        if not self.available:
            return None
        svc = getattr(self._tls, "service", None)
        if svc is not None:
            return svc
        svc = build(
            "youtube",
            "v3",
            developerKey=self.api_key,
            cache_discovery=False,
            static_discovery=True,
        )
        self._tls.service = svc
        return svc

    # ── Channel lookups ─────────────────────────────────────────

    def resolve_handle_to_channel_id(self, handle: str) -> Optional[str]:
        """Given `@mkbhd` (or `mkbhd`), return `UCBJycsmduvYEL83R_U4JriQ`.

        Uses `channels.list?forHandle=` which is 1 unit — much cheaper than
        the 100-unit search endpoint.
        """
        if not self.available or not handle:
            return None
        normalized = handle.lstrip("@")
        req = self._service.channels().list(
            part="id", forHandle=normalized, maxResults=1
        )
        try:
            resp = _execute_with_retry(req)
        except HttpError:
            return None
        items = resp.get("items") or []
        return items[0]["id"] if items else None

    def fetch_channel_stats(
        self, channel_ids: Iterable[str]
    ) -> dict[str, dict]:
        """Canonical channel stats for refresh. Returns {channel_id: stats}.

        Batches up to 50 per API call; 1 unit per batch.
        """
        if not self.available:
            return {}
        ids = [c for c in channel_ids if c]
        if not ids:
            return {}

        out: dict[str, dict] = {}
        for batch_start in range(0, len(ids), 50):
            batch = ids[batch_start : batch_start + 50]
            try:
                req = self._service.channels().list(
                    part="snippet,statistics,topicDetails,brandingSettings",
                    id=",".join(batch),
                    maxResults=50,
                )
                resp = _execute_with_retry(req)
            except HttpError:
                continue
            for item in resp.get("items") or []:
                snippet = item.get("snippet", {}) or {}
                stats = item.get("statistics", {}) or {}
                topics = item.get("topicDetails", {}) or {}
                branding = (item.get("brandingSettings") or {}).get(
                    "channel", {}
                ) or {}
                thumbs = snippet.get("thumbnails") or {}
                avatar_url = (
                    (thumbs.get("high") or {}).get("url")
                    or (thumbs.get("medium") or {}).get("url")
                    or (thumbs.get("default") or {}).get("url")
                )
                out[item["id"]] = {
                    "channel_id": item["id"],
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": snippet.get("publishedAt"),
                    "country": snippet.get("country")
                    or branding.get("country"),
                    "custom_url": snippet.get("customUrl"),
                    "avatar_url": avatar_url,
                    "subscriber_count": _to_int(stats.get("subscriberCount")),
                    "hidden_subscriber_count": bool(
                        stats.get("hiddenSubscriberCount", False)
                    ),
                    "view_count": _to_int(stats.get("viewCount")),
                    "video_count": _to_int(stats.get("videoCount")),
                    "topic_categories": topics.get("topicCategories") or [],
                    "topic_ids": topics.get("topicIds") or [],
                    "keywords": branding.get("keywords"),
                }
        return out

    # ── Video lookups ───────────────────────────────────────────

    def fetch_video_stats(
        self, video_ids: Iterable[str]
    ) -> dict[str, dict]:
        """Canonical video stats. Returns {video_id: stats}."""
        if not self.available:
            return {}
        ids = [v for v in video_ids if v]
        if not ids:
            return {}

        out: dict[str, dict] = {}
        for batch_start in range(0, len(ids), 50):
            batch = ids[batch_start : batch_start + 50]
            try:
                req = self._service.videos().list(
                    part="snippet,statistics,contentDetails,topicDetails",
                    id=",".join(batch),
                    maxResults=50,
                )
                resp = _execute_with_retry(req)
            except HttpError:
                continue
            for item in resp.get("items") or []:
                snippet = item.get("snippet", {}) or {}
                stats = item.get("statistics", {}) or {}
                content = item.get("contentDetails", {}) or {}
                topics = item.get("topicDetails", {}) or {}
                out[item["id"]] = {
                    "video_id": item["id"],
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": snippet.get("publishedAt"),
                    "channel_id": snippet.get("channelId"),
                    "tags": snippet.get("tags") or [],
                    "category_id": _to_int(snippet.get("categoryId")),
                    "default_language": snippet.get("defaultLanguage")
                    or snippet.get("defaultAudioLanguage"),
                    "view_count": _to_int(stats.get("viewCount")),
                    "like_count": _to_int(stats.get("likeCount")),
                    "comment_count": _to_int(stats.get("commentCount")),
                    "duration_iso8601": content.get("duration"),
                    "has_captions": _to_bool(content.get("caption")),
                    "live_broadcast_content": snippet.get(
                        "liveBroadcastContent"
                    ),
                    "topic_categories": topics.get("topicCategories") or [],
                }
        return out

    # ── Channel uploads (Phase 2.5 — primary video discovery) ───

    def _fetch_uploads_playlist_id(self, channel_id: str) -> Optional[str]:
        """Resolve a channel's uploads-playlist id (UU<channel_id_suffix>).

        1 quota unit. Every channel has exactly one auto-generated uploads
        playlist; it's the cheapest way to enumerate recent videos in
        reverse-chronological order.
        """
        if not self.available or not channel_id:
            return None
        req = self._service.channels().list(
            part="contentDetails", id=channel_id, maxResults=1
        )
        try:
            resp = _execute_with_retry(req)
        except HttpError:
            return None
        items = resp.get("items") or []
        if not items:
            return None
        related = (items[0].get("contentDetails") or {}).get(
            "relatedPlaylists", {}
        ) or {}
        return related.get("uploads")

    def list_channel_uploads(
        self, channel_id: str, limit: int = 20
    ) -> list[dict]:
        """Return up to `limit` most-recent uploads for the channel.

        Flow: channels.list (1 unit) → playlistItems.list (1 unit per 50)
        → videos.list (1 unit per 50) with full snippet/statistics/content.
        For limit<=50 the total is ~3 quota units per channel.

        Returns a list of normalized dicts matching `fetch_video_stats`
        output so downstream extractors don't need to know the source.
        """
        if not self.available or not channel_id:
            return []

        uploads_playlist_id = self._fetch_uploads_playlist_id(channel_id)
        if not uploads_playlist_id:
            return []

        # Paginate playlistItems.list until we've collected `limit` ids.
        video_ids: list[str] = []
        page_token: Optional[str] = None
        while len(video_ids) < limit:
            try:
                req_kwargs = {
                    "part": "contentDetails",
                    "playlistId": uploads_playlist_id,
                    "maxResults": min(50, limit - len(video_ids)),
                }
                if page_token:
                    req_kwargs["pageToken"] = page_token
                req = self._service.playlistItems().list(**req_kwargs)
                resp = _execute_with_retry(req)
            except HttpError:
                break
            for item in resp.get("items") or []:
                vid = (item.get("contentDetails") or {}).get("videoId")
                if vid:
                    video_ids.append(vid)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        if not video_ids:
            return []

        stats_by_id = self.fetch_video_stats(video_ids)
        # Preserve uploads-order (most recent first); videos.list returns
        # unordered so we walk the original id list.
        return [
            stats_by_id[vid] for vid in video_ids if vid in stats_by_id
        ]

    # ── Comment threads (Phase 2.5 — primary comment source) ────

    def list_comment_threads(
        self,
        video_id: str,
        order: str = "relevance",
        max_results: int = 20,
    ) -> list[dict]:
        """Return up to `max_results` top-level comment threads for a video.

        1 quota unit per call. `order='relevance'` mirrors Bright Data's
        default "top comments" behavior — denser signal for the audience
        intelligence layer than newest-first.

        Normalizes into the same shape as the Bright Data scraper so
        `extract_comment_metrics` works unchanged:
          {
            author, author_channel_id, text, date,
            like_count, reply_count, replies: [{author, author_channel_id, text}, ...]
          }
        """
        if not self.available or not video_id:
            return []

        req = self._service.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            order=order,
            maxResults=min(100, max_results),
            textFormat="plainText",
        )
        try:
            resp = _execute_with_retry(req)
        except HttpError:
            return []

        threads: list[dict] = []
        for item in resp.get("items") or []:
            top = (
                (item.get("snippet") or {}).get("topLevelComment") or {}
            ).get("snippet") or {}
            replies_raw = (item.get("replies") or {}).get("comments") or []
            replies = [
                {
                    "author": (r.get("snippet") or {}).get("authorDisplayName"),
                    "author_channel_id": (
                        (r.get("snippet") or {}).get("authorChannelId") or {}
                    ).get("value"),
                    "text": (r.get("snippet") or {}).get("textDisplay"),
                    "date": (r.get("snippet") or {}).get("publishedAt"),
                    "like_count": _to_int(
                        (r.get("snippet") or {}).get("likeCount")
                    ),
                }
                for r in replies_raw
            ]
            threads.append(
                {
                    "author": top.get("authorDisplayName"),
                    "author_channel_id": (
                        top.get("authorChannelId") or {}
                    ).get("value"),
                    "text": top.get("textDisplay"),
                    "date": top.get("publishedAt"),
                    "like_count": _to_int(top.get("likeCount")),
                    "reply_count": _to_int(
                        (item.get("snippet") or {}).get("totalReplyCount")
                    ),
                    "replies": replies,
                }
            )
        return threads

    # ── Keyword search (Phase 3 — on-demand discovery) ──────────
    #
    # `search.list` costs 100 quota units per call, vs 1 for channels.list.
    # We only use it in the discovery pipeline (interactive, user-triggered),
    # never for routine pipeline work. The multi-key pool absorbs the cost.

    def search_keyword_channels(
        self,
        query: str,
        max_results: int = 200,
        region_code: Optional[str] = None,
    ) -> list[dict]:
        """Return up to `max_results` channels matching `query` by YT relevance.

        Paginates `search.list?type=channel&order=relevance` 50 at a time.
        Each page is 100 quota units. Returns dicts with `channel_id, title,
        description, thumbnails, channel_url`. Region-code biases toward
        creators uploading in that geography (e.g. "IN" for India).
        """
        if not self.available or not query.strip():
            return []
        out: list[dict] = []
        page_token: Optional[str] = None
        # 200 max via 4 pages of 50; cap at API hard limit of 500.
        max_results = max(1, min(500, max_results))
        while len(out) < max_results:
            try:
                req_kwargs = {
                    "part": "snippet",
                    "q": query.strip(),
                    "type": "channel",
                    "order": "relevance",
                    "maxResults": min(50, max_results - len(out)),
                }
                if region_code:
                    req_kwargs["regionCode"] = region_code
                if page_token:
                    req_kwargs["pageToken"] = page_token
                req = self._service.search().list(**req_kwargs)
                resp = _execute_with_retry(req)
            except HttpError:
                break
            for item in resp.get("items") or []:
                snippet = item.get("snippet") or {}
                channel_id = (item.get("id") or {}).get("channelId")
                if not channel_id:
                    continue
                out.append(
                    {
                        "channel_id": channel_id,
                        "title": snippet.get("channelTitle")
                        or snippet.get("title"),
                        "description": snippet.get("description"),
                        "thumbnail": (
                            (snippet.get("thumbnails") or {}).get("default")
                            or {}
                        ).get("url"),
                        "channel_url": f"https://www.youtube.com/channel/{channel_id}",
                    }
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return out

    def search_keyword_video_channels(
        self,
        query: str,
        max_videos: int = 200,
        region_code: Optional[str] = None,
    ) -> list[str]:
        """Return unique channel IDs that have video matches for `query`.

        Paginates `search.list?type=video&order=relevance` and de-dups
        channel IDs from the snippet. Catches long-tail niche creators
        who don't rank in channel-search but whose recent uploads match.

        Each page costs 100 quota units. Returns up to ~max_videos worth
        of unique channels — typically ~30–60% dedup ratio (one famous
        creator gets multiple videos in the same search).
        """
        if not self.available or not query.strip():
            return []
        seen: set[str] = set()
        out: list[str] = []
        page_token: Optional[str] = None
        max_videos = max(1, min(500, max_videos))
        fetched = 0
        while fetched < max_videos:
            try:
                req_kwargs = {
                    "part": "snippet",
                    "q": query.strip(),
                    "type": "video",
                    "order": "relevance",
                    "maxResults": min(50, max_videos - fetched),
                }
                if region_code:
                    req_kwargs["regionCode"] = region_code
                if page_token:
                    req_kwargs["pageToken"] = page_token
                req = self._service.search().list(**req_kwargs)
                resp = _execute_with_retry(req)
            except HttpError:
                break
            items = resp.get("items") or []
            fetched += len(items)
            for item in items:
                snippet = item.get("snippet") or {}
                cid = snippet.get("channelId")
                if cid and cid not in seen:
                    seen.add(cid)
                    out.append(cid)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return out


def _to_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _to_bool(val) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    return str(val).lower() in {"true", "1", "yes"}
