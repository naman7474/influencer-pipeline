"""Per-creator Apify bundle cache for the public ``apify/instagram-scraper`` actor.

The pipeline expects BrightData-shaped records (``account``,
``followers``, ``content_type``, ``video_url`` etc.) but the public
default Apify actor returns its own shape (``username``, ``followersCount``,
``type``, ``videoUrl`` etc.). This module:

  1. Drives 2-3 actor runs per creator:
       - ``resultsType=details``  → 1 item, profile object (+ limited
         ``latestPosts`` array we don't rely on).
       - ``resultsType=posts``    → up to ``num_posts + num_reels`` items,
         full post objects with ``videoUrl`` / ``videoPlayCount`` /
         ``videoViewCount`` / ``videoDuration`` / ``timestamp`` etc.
       - ``resultsType=comments`` → only when ``comments_per_reel > 0``,
         scoped to the top reels selected from the second run.

  2. Translates every response into the BD-shaped dicts the existing
     ``extract_*_metrics`` extractors already know how to read.

  3. Caches the translated bundle by username for the lifetime of the
     process — the pipeline calls profile / posts / reels / comments as
     four separate steps, but all four read from the same cache so the
     actor only fires once per creator (twice when comments are needed).

Override the actor via ``APIFY_ACTOR_INSTAGRAM`` (e.g. point at a custom
actor that emits BD-shaped records natively); the translator is skipped
in that case via the ``APIFY_ACTOR_BD_NATIVE`` flag.
"""

import logging
import os
import re
from typing import Any

from pipeline.apify_client import ApifyClient, make_default_client

logger = logging.getLogger(__name__)


_CACHE: dict[str, dict[str, list[dict] | dict | None]] = {}

# Per-process default for the actor's comments scrape volume. The pipeline
# overrides this on a per-creator basis: if the creator has a commerce
# signal, we set it to 0 (skip scraping IG comments entirely) before the
# first bundle.fetch() call. This avoids threading a parameter through the
# four scrape_*_apify shims, all of which call into bundle.fetch().
_default_comments_per_reel: int = 10


def set_default_comments_per_reel(n: int) -> None:
    """Set the comments_per_reel that the next ``fetch()`` will use when
    its caller doesn't pass an explicit argument.

    Pipeline calls this before any scrape_*_apify function so the very
    first fetch (which populates the cache) honors the policy. The
    setting is process-global; reset between creators if you want a
    different policy per run.
    """
    global _default_comments_per_reel
    _default_comments_per_reel = max(0, int(n))


_DEFAULT_ACTOR = "apify/instagram-scraper"


def _actor_id() -> str:
    return os.environ.get("APIFY_ACTOR_INSTAGRAM", _DEFAULT_ACTOR)


def _actor_emits_bd_native() -> bool:
    """The custom Instaloader actor emits BD-shape directly (``_kind`` field).
    Anything else (including the default ``apify/instagram-scraper``) needs
    translation. Set ``APIFY_ACTOR_BD_NATIVE=1`` to opt into the legacy
    pass-through path when running the custom actor.
    """
    return os.environ.get("APIFY_ACTOR_BD_NATIVE", "").lower() in {"1", "true", "yes"}


def _profile_url(username: str) -> str:
    return f"https://www.instagram.com/{username.strip('/')}/"


def fetch(
    username: str,
    *,
    num_posts: int = 5,
    num_reels: int = 10,
    comments_per_reel: int | None = None,
    client: ApifyClient | None = None,
) -> dict[str, Any]:
    """Run the actor for ``username`` (or return cached).

    Returns a dict with four keys: ``profile`` (single dict or None),
    ``posts`` (list), ``reels`` (list), ``comments`` (list). Each value
    is BD-shaped — ready for the existing extractors regardless of which
    actor was used.

    ``comments_per_reel=None`` (the default) honors whatever
    ``set_default_comments_per_reel`` last set.
    """
    if comments_per_reel is None:
        comments_per_reel = _default_comments_per_reel
    key = username.lower().lstrip("@")
    if key in _CACHE:
        return _CACHE[key]

    client = client or make_default_client()
    actor_id = _actor_id()

    # Legacy path — custom Instaloader actor emits BD-shape natively.
    if _actor_emits_bd_native():
        bundle = _fetch_bd_native(
            client, actor_id, key, num_posts, num_reels, comments_per_reel,
        )
        _CACHE[key] = bundle
        return bundle

    # Default path — public ``apify/instagram-scraper``, response translated.
    bundle = _fetch_public_actor(
        client, actor_id, key, num_posts, num_reels, comments_per_reel,
    )
    _CACHE[key] = bundle
    return bundle


def get_cached(username: str) -> dict[str, Any] | None:
    return _CACHE.get(username.lower().lstrip("@"))


def any_cached_username() -> str | None:
    return next(iter(_CACHE.keys()), None)


def clear(username: str | None = None) -> None:
    if username is None:
        _CACHE.clear()
    else:
        _CACHE.pop(username.lower().lstrip("@"), None)


# ── Public-actor flow (default) ────────────────────────────────────


def _fetch_public_actor(
    client: ApifyClient,
    actor_id: str,
    username: str,
    num_posts: int,
    num_reels: int,
    comments_per_reel: int,
) -> dict[str, Any]:
    profile_url = _profile_url(username)
    total_posts_to_pull = max(1, num_posts + num_reels)

    logger.info(
        "Apify (public) bundle fetch: actor=%s user=@%s posts=%d reels=%d comments/reel=%d",
        actor_id, username, num_posts, num_reels, comments_per_reel,
    )

    # ── 1. Profile (details run) ──
    details_payload: dict[str, Any] = {
        "directUrls": [profile_url],
        "resultsType": "details",
        "resultsLimit": 1,
        "addParentData": False,
    }
    details_items = client.trigger_and_wait(actor_id, details_payload)
    profile_raw = details_items[0] if details_items else None

    # ── 2. Posts + reels in one run, split by content type ──
    posts_payload: dict[str, Any] = {
        "directUrls": [profile_url],
        "resultsType": "posts",
        "resultsLimit": total_posts_to_pull,
        "addParentData": False,
    }
    raw_post_items = client.trigger_and_wait(actor_id, posts_payload)

    bd_profile = _translate_profile(profile_raw) if profile_raw else None

    bd_posts: list[dict] = []
    bd_reels: list[dict] = []
    for item in raw_post_items or []:
        translated = _translate_post(item)
        if translated.get("content_type") == "Video":
            bd_reels.append(translated)
        else:
            bd_posts.append(translated)

    # Trim to requested counts. The actor returns most-recent-first within
    # the profile, so slicing keeps the freshest of each type.
    bd_posts = bd_posts[:num_posts]
    bd_reels = bd_reels[:num_reels]

    # ── 3. Comments (only if requested and we have reel URLs) ──
    bd_comments: list[dict] = []
    if comments_per_reel > 0 and bd_reels:
        # Use the top-N reel URLs. _translate_post emits the canonical post
        # URL as "url" (NOT "post_url" — that key is always None here), so the
        # old r["post_url"] gave an empty list and silently fetched 0 comments.
        comment_targets = [
            r.get("url") for r in bd_reels[: max(1, num_reels)]
            if r.get("url")
        ]
        if comment_targets:
            # Per-target cap — apify/instagram-scraper's ``resultsLimit`` is
            # per-input-URL when resultsType=comments. So passing 5 yields up
            # to 5 comments per reel URL.
            comments_payload: dict[str, Any] = {
                "directUrls": comment_targets,
                "resultsType": "comments",
                "resultsLimit": comments_per_reel,
                "addParentData": True,
            }
            try:
                raw_comments = client.trigger_and_wait(actor_id, comments_payload)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Comment scrape failed for @%s (continuing without): %s",
                    username, e,
                )
                raw_comments = []
            for c in raw_comments or []:
                bd_comments.append(_translate_comment(c))

    logger.info(
        "Apify (public) bundle for @%s: profile=%s posts=%d reels=%d comments=%d",
        username,
        "yes" if bd_profile else "no",
        len(bd_posts), len(bd_reels), len(bd_comments),
    )

    return {
        "profile": bd_profile,
        "posts": bd_posts,
        "reels": bd_reels,
        "comments": bd_comments,
    }


# ── Translators (apify/instagram-scraper → BD shape) ──────────────


_HASHTAG_RE = re.compile(r"#(\w+)")


def _translate_profile(item: dict) -> dict:
    """Map ``apify/instagram-scraper`` 'details' item to BD-shaped profile."""
    bio = item.get("biography") or ""
    bio_tags = _HASHTAG_RE.findall(bio)

    # Pull category from a couple of possible field names; the actor
    # returns ``businessCategoryName`` for business accounts and a
    # ``category_name`` field on some plan tiers.
    category = (
        item.get("businessCategoryName")
        or item.get("categoryName")
        or item.get("category_name")
        or item.get("category")
    )

    return {
        "account": item.get("username"),
        "id": item.get("id"),
        "fbid": item.get("fbid"),
        "profile_name": item.get("fullName"),
        "profile_image_link": item.get("profilePicUrlHD") or item.get("profilePicUrl"),
        "biography": bio or None,
        "external_url": item.get("externalUrl"),
        # Public actor doesn't return city/country; downstream is fine
        # with None and will lean on geo-synthesis from posts.
        "city": None,
        "country": None,
        "category": category,
        "followers": int(item.get("followersCount") or 0),
        "following": int(item.get("followsCount") or 0),
        "posts_count": int(item.get("postsCount") or 0),
        "is_business_account": bool(item.get("isBusinessAccount", False)),
        # Public actor exposes ``isBusinessAccount`` only; treat is_professional
        # as a no-op and let the existing extractor default to False.
        "is_professional_account": False,
        "is_verified": bool(item.get("verified", False)),
        # No avg_engagement from the actor; downstream re-computes from posts.
        "avg_engagement": None,
        "bio_hashtags": bio_tags,
        # post_hashtags will be aggregated by the post extractor — leave empty.
        "post_hashtags": [],
        "contact_email": None,
        "contact_phone_number": None,
    }


_PUBLIC_TYPE_TO_BD_CONTENT_TYPE = {
    "Video": "Video",
    "Image": "Image",
    "Sidecar": "Sidecar",  # extract_post_metrics normalises Sidecar → Carousel
}


def _translate_post(item: dict) -> dict:
    """Map a public-actor post item to BD-shape used by extract_post_metrics
    AND extract_reel_metrics. Both extractors consume the same dict shape
    (the latter just reads extra video-specific keys), so a single mapping
    covers both."""
    post_type = item.get("type") or "Image"
    bd_content_type = _PUBLIC_TYPE_TO_BD_CONTENT_TYPE.get(post_type, post_type)

    tagged_users = item.get("taggedUsers") or item.get("tagged_users") or []
    # The actor sometimes returns tagged users as objects {username:...};
    # normalise to flat strings to match BD's shape.
    tagged_handles: list[str] = []
    for t in tagged_users:
        if isinstance(t, dict):
            uname = t.get("username") or t.get("full_name")
            if uname:
                tagged_handles.append(uname)
        elif isinstance(t, str):
            tagged_handles.append(t)

    # Embedded comments preview — present on some plans. Translate to BD's
    # ``top_comments`` shape so extract_reel_metrics can pick them up
    # directly when the dedicated comments-scrape was skipped.
    raw_top_comments = (
        item.get("latestComments")
        or item.get("topComments")
        or item.get("firstComment")
        or []
    )
    if isinstance(raw_top_comments, dict):
        raw_top_comments = [raw_top_comments]
    top_comments = []
    for c in raw_top_comments or []:
        if not isinstance(c, dict):
            continue
        top_comments.append({
            "comment_user": c.get("ownerUsername") or c.get("owner", {}).get("username"),
            "comment": c.get("text"),
            "comment_date": c.get("timestamp"),
            "likes_number": c.get("likesCount") or 0,
        })

    return {
        "post_id": item.get("id") or item.get("shortCode"),
        # ``url`` is the canonical BD key — read by select_top_reels,
        # select_top_posts_for_comments, and the post-URL list extractors
        # in scraper_posts.py. Emitting ``post_url`` instead silently
        # drops every reel from select_top_reels' url-filter and
        # collapses the Whisper / comments steps to no-ops.
        "url": item.get("url"),
        "input_url": item.get("inputUrl"),
        # Cover image / thumbnail — the public actor returns it as displayUrl.
        # upsert_posts re-hosts this to Supabase Storage (persist_thumbnail).
        "thumbnail": item.get("displayUrl") or item.get("thumbnailSrc"),
        "display_url": item.get("displayUrl"),
        "images": item.get("images") or [],
        "description": item.get("caption") or "",
        "likes": int(item.get("likesCount") or 0),
        "num_comments": int(item.get("commentsCount") or 0),
        "content_type": bd_content_type,
        "date_posted": item.get("timestamp"),  # already ISO 8601
        "hashtags": item.get("hashtags") or [],
        "tagged_users": tagged_handles,
        "owner_id": item.get("ownerId"),
        "owner_username": item.get("ownerUsername"),
        # ── Reel-only fields (None for non-Video posts) ──
        "video_url": item.get("videoUrl"),
        "video_view_count": item.get("videoViewCount"),
        "video_play_count": item.get("videoPlayCount"),
        # extract_reel_metrics also reads ``views`` as a fallback.
        "views": item.get("videoViewCount") or item.get("videoPlayCount"),
        "length": item.get("videoDuration"),
        "top_comments": top_comments,
        # Coauthor / collab data (rare in public-actor output but pass through).
        "coauthor_producers": item.get("coauthorProducers") or [],
    }


def _translate_comment(item: dict) -> dict:
    """Map a public-actor comment item to BD-shape."""
    return {
        "comment_user": item.get("ownerUsername"),
        "user_commenting": item.get("ownerUsername"),
        "comment": item.get("text"),
        "text": item.get("text"),
        "comment_date": item.get("timestamp"),
        "date_of_comment": item.get("timestamp"),
        "likes_number": int(item.get("likesCount") or 0),
        "likes": int(item.get("likesCount") or 0),
        # The post URL the comment belongs to (added when addParentData=true).
        "source_post_url": item.get("postUrl") or item.get("inputUrl"),
    }


# ── Legacy path: custom actor that already emits BD-native records ────


def _fetch_bd_native(
    client: ApifyClient,
    actor_id: str,
    username: str,
    num_posts: int,
    num_reels: int,
    comments_per_reel: int,
) -> dict[str, Any]:
    """Fetch from the legacy custom Instaloader actor (BD-shape native)."""
    logger.info(
        "Apify (BD-native) bundle fetch: actor=%s user=@%s posts=%d reels=%d comments/reel=%d",
        actor_id, username, num_posts, num_reels, comments_per_reel,
    )
    payload: dict[str, Any] = {
        "username": username,
        "num_posts": num_posts,
        "num_reels": num_reels,
        "comments_per_reel": comments_per_reel,
    }
    ig_user = os.environ.get("IG_USERNAME")
    ig_pass = os.environ.get("IG_PASSWORD")
    if ig_user and ig_pass:
        payload["ig_username"] = ig_user
        payload["ig_password"] = ig_pass
    items = client.trigger_and_wait(actor_id, payload)
    return _split_bd_native(items)


def _split_bd_native(items: list[dict]) -> dict[str, Any]:
    profile = None
    posts: list[dict] = []
    reels: list[dict] = []
    comments: list[dict] = []
    for it in items or []:
        kind = it.pop("_kind", None)
        if kind == "profile" and profile is None:
            profile = it
        elif kind == "post":
            posts.append(it)
        elif kind == "reel":
            reels.append(it)
        elif kind == "comment":
            comments.append(it)
    return {
        "profile": profile,
        "posts": posts,
        "reels": reels,
        "comments": comments,
    }
