"""Brand ↔ creator matching — Python in-process orchestration.

The TS engine at ``web/src/lib/matching/engine.ts`` (~2k LOC) remains the
canonical full-fidelity scorer: brand-safety, calibration percentiles,
niche-fit, geo, format-fit, semantic-similarity, past-collab — all of
that lives in TS and is exposed via ``POST /api/matching/compute`` (or
``/api/matching/recompute-creator``).

This module exists for two reasons:

  1. **Loud failures** — the legacy ``_trigger_matching_compute`` in
     ``handlers.py`` was fire-and-forget over HTTP and silently swallowed
     errors with a "scheduled recompute will catch up" comment that's
     fiction (no scheduled recompute exists). Now the worker calls
     :func:`recompute_for_brand` / :func:`recompute_for_creator` which
     raises on non-2xx, surfaces the failure in the job log, and (when
     the TS engine is unreachable) writes a baseline embedding-only
     score so the matches table at least has a row.

  2. **Embedding-only fallback** — :func:`python_baseline_score` does
     cosine similarity between the brand's platform embedding and each
     creator's content embedding, writing approximate ``match_score`` /
     ``content_style_score`` rows so the dashboard isn't empty when TS
     is down. The TS recompute (when it next runs) overwrites these
     with full-fidelity scores.

Future work: incrementally port the TS scoring functions
(``computeNicheFit`` / ``computeAudienceGeo`` / ``computeBrandSafetyScore``
etc.) into this module so we can retire the HTTP hop entirely.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import httpx

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────────


class MatchingError(RuntimeError):
    """Raised by recompute_* when the TS engine returns a hard failure
    and the Python baseline also fails to write any rows."""


def recompute_for_brand(
    db,
    brand_id: str,
    *,
    top_k: int = 200,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    """Trigger a brand-scoped matching recompute.

    Strategy:
      1. Call TS ``/api/matching/compute`` (the authoritative scorer).
      2. On success, return its response payload.
      3. On TS failure (transport error, non-2xx, timeout), run the
         Python embedding-only baseline so the matches table at least
         has fresh rows for this brand. Raise :class:`MatchingError`
         only if BOTH paths fail.
    """
    return _recompute(
        db,
        kind="brand",
        ts_path="/api/matching/compute",
        ts_payload={"brand_id": brand_id, **({"platforms": platforms} if platforms else {})},
        baseline=lambda: python_baseline_score_for_brand(
            db, brand_id, top_k=top_k, platforms=platforms,
        ),
    )


def recompute_for_creator(
    db,
    creator_id: str,
    *,
    top_k: int = 50,
) -> dict[str, Any]:
    """Trigger a creator-scoped matching recompute.

    Mirrors :func:`recompute_for_brand` against the
    ``/api/matching/recompute-creator`` TS endpoint, falling back to
    the Python baseline for this creator across all brands when TS is
    unreachable.
    """
    return _recompute(
        db,
        kind="creator",
        ts_path="/api/matching/recompute-creator",
        ts_payload={"creator_id": creator_id},
        baseline=lambda: python_baseline_score_for_creator(
            db, creator_id, top_k=top_k,
        ),
    )


# ── TS HTTP path ────────────────────────────────────────────────────────────


def _recompute(
    db,
    *,
    kind: str,
    ts_path: str,
    ts_payload: dict,
    baseline,
) -> dict[str, Any]:
    """Call the TS endpoint; fall back to the Python baseline on failure."""
    base = os.environ.get("WEB_APP_URL")
    secret = os.environ.get("MATCHING_COMPUTE_SECRET")

    ts_error: str | None = None
    if base:
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["X-Worker-Secret"] = secret
        url = f"{base.rstrip('/')}{ts_path}"
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=ts_payload, headers=headers)
            if resp.status_code // 100 == 2:
                logger.info(
                    "TS matching recompute (%s) succeeded: %s", kind, ts_payload
                )
                return {
                    "source": "ts_engine",
                    "status_code": resp.status_code,
                    "body": _try_json(resp.text),
                }
            ts_error = (
                f"TS {ts_path} returned {resp.status_code}: {resp.text[:300]}"
            )
        except Exception as e:  # noqa: BLE001
            ts_error = f"TS {ts_path} transport error: {type(e).__name__}: {e}"
    else:
        ts_error = "WEB_APP_URL not configured"

    logger.warning("Matching: %s; falling back to python baseline", ts_error)
    try:
        baseline_result = baseline()
    except Exception as e:  # noqa: BLE001
        # Both paths failed — make the caller see it.
        raise MatchingError(
            f"matching recompute failed; ts_error={ts_error}; "
            f"baseline_error={type(e).__name__}: {e}"
        )
    return {
        "source": "python_baseline",
        "ts_error": ts_error,
        **baseline_result,
    }


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ── Python baseline (embedding-only) ────────────────────────────────────────


def python_baseline_score_for_brand(
    db,
    brand_id: str,
    *,
    top_k: int = 200,
    platforms: list[str] | None = None,
) -> dict[str, Any]:
    """Embedding-only matching for a single brand.

    For each platform analysis with a content_embedding:
      1. Pull every creator content_embedding for that platform.
      2. Compute cosine similarity (0..1, shifted from the raw [-1, 1]).
      3. Upsert top-k rows into ``creator_brand_matches`` with the
         similarity as ``match_score * 100`` and ``content_style_score``.
         Other sub-scores stay 0 — the TS recompute will fill them in.
    """
    brand_analyses = _load_brand_embeddings(db, brand_id)
    if not brand_analyses:
        return {"matches_written": 0, "reason": "no_brand_embeddings"}

    targets = platforms or list(brand_analyses.keys())
    written = 0
    by_platform: dict[str, int] = {}
    for platform in targets:
        brand_emb = brand_analyses.get(platform)
        if brand_emb is None:
            continue
        creator_rows = _load_creator_embeddings_for_platform(db, platform)
        scored: list[tuple[str, float]] = []
        for cid, emb in creator_rows:
            sim = _cosine_similarity(brand_emb, emb)
            if sim > 0:
                scored.append((cid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        if not top:
            continue
        n = _upsert_match_rows(
            db, brand_id=brand_id, platform=platform, scored=top,
        )
        by_platform[platform] = n
        written += n

    return {"matches_written": written, "by_platform": by_platform}


def python_baseline_score_for_creator(
    db,
    creator_id: str,
    *,
    top_k: int = 50,
) -> dict[str, Any]:
    """Embedding-only matching for one creator across every brand."""
    creator_embeddings = _load_creator_embeddings_for_creator(db, creator_id)
    if not creator_embeddings:
        return {"matches_written": 0, "reason": "no_creator_embedding"}

    written = 0
    by_platform: dict[str, int] = {}
    for platform, creator_emb in creator_embeddings.items():
        brands_by_id = _load_brand_embeddings_for_platform(db, platform)
        scored: list[tuple[str, float]] = []
        for bid, brand_emb in brands_by_id.items():
            sim = _cosine_similarity(brand_emb, creator_emb)
            if sim > 0:
                scored.append((bid, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        for bid, sim in top:
            _upsert_match_rows(
                db, brand_id=bid, platform=platform,
                scored=[(creator_id, sim)],
            )
            written += 1
        by_platform[platform] = len(top)

    return {"matches_written": written, "by_platform": by_platform}


# ── Cosine ──────────────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity shifted from [-1, 1] to [0, 1], matching the
    convention in ``web/src/lib/matching/engine.ts:cosineSimilarity``.
    """
    if not a or not b or len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na == 0 or nb == 0:
        return 0.0
    raw = dot / (math.sqrt(na) * math.sqrt(nb))
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


# ── Embedding loaders ───────────────────────────────────────────────────────
# Postgres returns vector(1536) as a string "[0.1,-0.2,...]" via PostgREST.
# Parse on read so the cosine loop doesn't see strings.


def _parse_embedding(v: Any) -> list[float] | None:
    if v is None:
        return None
    if isinstance(v, list):
        return [float(x) for x in v]
    if isinstance(v, str) and v.startswith("["):
        try:
            arr = json.loads(v)
        except json.JSONDecodeError:
            return None
        if not isinstance(arr, list):
            return None
        return [float(x) for x in arr]
    return None


def _load_brand_embeddings(db, brand_id: str) -> dict[str, list[float]]:
    """Return ``{platform: embedding}`` for one brand's completed analyses."""
    rows = (
        db.table("brand_platform_analyses")
        .select("platform, content_embedding, analysis_status")
        .eq("brand_id", brand_id)
        .execute()
    )
    out: dict[str, list[float]] = {}
    for r in (rows.data or []):
        if r.get("analysis_status") not in (None, "completed"):
            # Allow "completed" or unset for backward compat
            if r.get("analysis_status") != "completed":
                continue
        emb = _parse_embedding(r.get("content_embedding"))
        if emb is not None:
            out[r["platform"]] = emb
    return out


def _load_brand_embeddings_for_platform(
    db, platform: str
) -> dict[str, list[float]]:
    """Return ``{brand_id: embedding}`` for every completed analysis on
    ``platform``. Used by the creator-scoped recompute baseline.
    """
    rows = (
        db.table("brand_platform_analyses")
        .select("brand_id, content_embedding, analysis_status")
        .eq("platform", platform)
        .execute()
    )
    out: dict[str, list[float]] = {}
    for r in (rows.data or []):
        if r.get("analysis_status") != "completed":
            continue
        emb = _parse_embedding(r.get("content_embedding"))
        if emb is not None:
            out[r["brand_id"]] = emb
    return out


_EMBEDDING_PAGE = 500  # PostgREST default limit; bigger pages cause silent truncation


def _load_creator_embeddings_for_platform(
    db, platform: str
) -> Iterable[tuple[str, list[float]]]:
    """Yield ``(creator_id, embedding)`` for every creator on ``platform``.

    Paginates via ``.range()`` so we don't get silently clamped at 1000.
    """
    offset = 0
    while True:
        page = (
            db.table("creator_embeddings")
            .select("creator_id, embedding")
            .eq("platform", platform)
            .range(offset, offset + _EMBEDDING_PAGE - 1)
            .execute()
        )
        rows = page.data or []
        if not rows:
            break
        for r in rows:
            emb = _parse_embedding(r.get("embedding"))
            if emb is not None:
                yield r["creator_id"], emb
        if len(rows) < _EMBEDDING_PAGE:
            break
        offset += _EMBEDDING_PAGE


def _load_creator_embeddings_for_creator(
    db, creator_id: str
) -> dict[str, list[float]]:
    """Return ``{platform: embedding}`` for one creator."""
    rows = (
        db.table("creator_embeddings")
        .select("platform, embedding")
        .eq("creator_id", creator_id)
        .execute()
    )
    out: dict[str, list[float]] = {}
    for r in (rows.data or []):
        emb = _parse_embedding(r.get("embedding"))
        if emb is not None:
            out[r["platform"]] = emb
    return out


# ── Upsert ──────────────────────────────────────────────────────────────────


def _upsert_match_rows(
    db,
    *,
    brand_id: str,
    platform: str,
    scored: list[tuple[str, float]],
) -> int:
    """Upsert one row per (creator, brand, platform) with the similarity
    score. Other sub-scores stay 0 — the TS recompute fills them in."""
    if not scored:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    for creator_id, sim in scored:
        score_100 = round(sim * 100.0, 1)
        rows.append(
            {
                "creator_id": creator_id,
                "brand_id": brand_id,
                "platform": platform,
                "match_score": score_100,
                "content_style_score": score_100,
                # Niche/geo/safety/etc. left at table defaults (0) so the
                # TS recompute fills them on the next pass.
                "algorithm_version": "python_baseline_1.0",
                "computed_at": now_iso,
            }
        )
    db.table("creator_brand_matches").upsert(
        rows, on_conflict="creator_id,brand_id,platform"
    ).execute()
    return len(rows)
