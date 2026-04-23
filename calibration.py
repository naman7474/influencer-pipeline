"""
Runtime loader for percentile-calibrated scoring benchmarks.

The scoring math in pipeline.py used to hardcode ER thresholds
(DEFAULT_ER_BENCHMARKS) — a 6% ER was "top nano" whether or not our
actual cohort's nano p75 sat at 4% or 9%. This module keeps the
benchmarks live: a weekly job recomputes percentiles into
er_benchmarks (is_active=true), and callers on the pipeline side
load the active row here before running compute_creator_scores.

Keeping this separate from pipeline.py lets compute_creator_scores
stay DB-agnostic and unit-testable without a Supabase mock.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pipeline.pipeline import DEFAULT_ER_BENCHMARKS

logger = logging.getLogger(__name__)

# Cached value + timestamp. 1h TTL is short enough that a fresh
# recalibration (run weekly) propagates well within a single worker
# lifetime, but long enough that per-creator workloads don't thrash
# the DB with identical reads.
_CACHE_TTL_SECONDS = 3600
_cache: dict[str, Any] = {"loaded_at": 0.0, "benchmarks": None}


def load_er_benchmarks(db) -> dict[str, float]:
    """Return `{tier: er_threshold}` where er_threshold is p75 of the
    tier's observed ER distribution (what a "strong" creator looks
    like for that size).

    Falls back to DEFAULT_ER_BENCHMARKS when the table is empty,
    unreachable, or the fetched row looks malformed. The fallback is
    not cached, so a recalibration job that writes rows after a cold
    start is picked up on the next call.
    """
    now = time.monotonic()
    cached = _cache.get("benchmarks")
    if cached is not None and now - _cache["loaded_at"] < _CACHE_TTL_SECONDS:
        return cached

    benchmarks = dict(DEFAULT_ER_BENCHMARKS)
    try:
        res = (
            db.table("er_benchmarks")
            .select("tier, p75")
            .eq("is_active", True)
            .execute()
        )
        rows = res.data or []
        for row in rows:
            tier = row.get("tier")
            p75 = row.get("p75")
            if tier and p75 is not None:
                try:
                    benchmarks[tier] = float(p75)
                except (TypeError, ValueError):
                    continue
    except Exception as e:  # pragma: no cover — defensive
        logger.warning(
            "load_er_benchmarks: falling back to defaults (%s)", e
        )
        return dict(DEFAULT_ER_BENCHMARKS)

    # Only cache when we actually loaded live rows. If the table was
    # empty and we stayed on defaults, the next call should retry.
    if any(
        benchmarks[t] != DEFAULT_ER_BENCHMARKS[t]
        for t in DEFAULT_ER_BENCHMARKS
    ):
        _cache["benchmarks"] = benchmarks
        _cache["loaded_at"] = now

    return benchmarks


def reset_cache() -> None:
    """For tests — clear the in-memory cache."""
    _cache["loaded_at"] = 0.0
    _cache["benchmarks"] = None
