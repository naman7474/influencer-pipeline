"""
Media persistence — download ephemeral Instagram CDN media and re-host
in Supabase Storage so URLs never expire.

Public buckets (creator-avatars, post-thumbnails) return permanent public URLs.
"""

import hashlib
import logging
import mimetypes
from urllib.parse import urlparse

import requests
from supabase import Client

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT = 30  # seconds

# Map bucket → whether it's public (uses get_public_url) vs private (needs signed URL)
_PUBLIC_BUCKETS = {"creator-avatars", "post-thumbnails"}


def _guess_extension(url: str, content_type: str | None) -> str:
    """Derive a file extension from the URL path or Content-Type header."""
    # Try URL path first
    path = urlparse(url).path
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov"):
        if ext in path.lower():
            return ext

    # Fallback to Content-Type header
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext

    return ".jpg"  # safe default for Instagram images


def _content_hash(data: bytes) -> str:
    """Short SHA-256 prefix for deduplication."""
    return hashlib.sha256(data).hexdigest()[:16]


def persist_media(
    db: Client,
    bucket: str,
    source_url: str,
    path_prefix: str,
) -> str | None:
    """
    Download media from *source_url* and upload to Supabase Storage.

    Args:
        db: Supabase client (needs service_role key for storage writes).
        bucket: Storage bucket id, e.g. "creator-avatars" or "post-thumbnails".
        source_url: Ephemeral CDN URL to download from.
        path_prefix: Folder/name prefix inside the bucket,
                     e.g. "creators/<handle>" or "posts/<post_id>".

    Returns:
        Permanent public URL on success, or None on failure.
    """
    if not source_url:
        return None

    # IG/FB post-media CDN 403s plain server requests; a browser-ish UA + the
    # Apify residential proxy (if configured) maximise the chance the download
    # succeeds at scrape time (the only time these signed URLs are reachable).
    import os

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 Instagram"
        ),
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    }
    proxy = os.environ.get("MEDIA_DOWNLOAD_PROXY") or os.environ.get("YT_DLP_PROXY")
    proxies = {"http": proxy, "https": proxy} if proxy else None

    try:
        resp = requests.get(
            source_url, timeout=_DOWNLOAD_TIMEOUT, headers=headers, proxies=proxies
        )
        resp.raise_for_status()
    except Exception:
        # Retry once without the proxy (some CDNs reject proxied IPs).
        try:
            resp = requests.get(source_url, timeout=_DOWNLOAD_TIMEOUT, headers=headers)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"Failed to download media from {source_url}: {e}")
            return None

    data = resp.content
    if not data:
        logger.warning(f"Empty response body from {source_url}")
        return None

    content_type = resp.headers.get("Content-Type")
    ext = _guess_extension(source_url, content_type)
    file_hash = _content_hash(data)
    storage_path = f"{path_prefix}/{file_hash}{ext}"

    # Determine MIME type for upload
    mime = content_type or mimetypes.guess_type(f"file{ext}")[0] or "image/jpeg"

    try:
        store = db.storage.from_(bucket)

        # Upload (upsert: overwrite if same path exists)
        store.upload(
            path=storage_path,
            file=data,
            file_options={"content-type": mime, "upsert": "true"},
        )

        # For public buckets, return the permanent public URL
        if bucket in _PUBLIC_BUCKETS:
            public_url = store.get_public_url(storage_path)
            logger.info(f"Persisted media → {public_url}")
            return public_url

        # For private buckets we'd create a signed URL — not needed yet
        logger.info(f"Uploaded to private bucket {bucket}/{storage_path}")
        return storage_path

    except Exception as e:
        logger.warning(f"Failed to upload media to {bucket}/{storage_path}: {e}")
        return None


def persist_avatar(db: Client, handle: str, avatar_url: str) -> str | None:
    """Re-host a creator's avatar image in Supabase Storage."""
    return persist_media(db, "creator-avatars", avatar_url, f"creators/{handle}")


def persist_thumbnail(db: Client, post_id: str, thumbnail_url: str) -> str | None:
    """Re-host a post thumbnail in Supabase Storage."""
    return persist_media(db, "post-thumbnails", thumbnail_url, f"posts/{post_id}")
