"""Webhook-driven Instagram creator-scrape FSM.

The legacy synchronous path (``apify_instagram_bundle.fetch()``) polls
Apify for 30-90s per actor run, tying up the worker thread. This module
replaces that with a webhook-driven flow:

  job claimed → start_scrape() kicks off Apify run 1, persists state
                in payload.apify, returns ``JobPaused`` so the worker
                releases the thread without marking the job complete

  Apify webhook → /apify-webhook → resume() advances the FSM:
    stage=awaiting_posts   → fetch dataset, translate posts+profile+reels,
                             decide if comments-run is needed
       if yes              → start comments run, stage=awaiting_comments
       if no               → _finish_scrape: LLM + score + store + match

    stage=awaiting_comments → fetch dataset, translate comments
                            → _finish_scrape: LLM + score + store + match

Recovery: a stale-job sweep polls Apify with ``get_run_status`` for any
``running`` job whose webhook never landed. This is the only place
polling survives in the IG happy path.

Activated by ``APIFY_WEBHOOKS=1``. With the flag off, ``handle_creator_ig_scrape``
takes the legacy synchronous path (still working until Phase 6 cleanup).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pipeline import apify_instagram_bundle
from pipeline import db as pdb
from pipeline.apify_client import make_default_client

logger = logging.getLogger(__name__)


# ── Tunables ────────────────────────────────────────────────────────────────

DEFAULT_ACTOR = os.environ.get("APIFY_ACTOR_INSTAGRAM", "apify/instagram-scraper")

NUM_POSTS = 5
NUM_REELS = 7  # Per the rewrite plan — trimmed from 10
RESULTS_LIMIT = NUM_POSTS + NUM_REELS

# Comments policy: top 3 reels × 4 comments each = up to 12 comments,
# enough for sentiment + language inference at a fraction of the
# 50-comment legacy cost.
COMMENTS_PER_REEL = 4
TOP_REELS_FOR_COMMENTS = 3

# Skip the dedicated comments run when this many comments came back
# embedded on the posts items (``latestComments``).
EMBEDDED_COMMENT_THRESHOLD = 10


# ── FSM stage labels ────────────────────────────────────────────────────────

STAGE_AWAITING_POSTS = "awaiting_posts"
STAGE_AWAITING_COMMENTS = "awaiting_comments"


# ── Sentinel exception ──────────────────────────────────────────────────────


class JobPaused(Exception):
    """Raised by a handler to signal the worker that the job is parked
    in ``running`` awaiting an external callback (Apify webhook). The
    worker treats it as success-but-don't-complete.
    """


# ── Mode toggle ─────────────────────────────────────────────────────────────


def is_webhook_mode() -> bool:
    return os.environ.get("APIFY_WEBHOOKS", "").lower() in {"1", "true", "yes"}


# ── Stage 0: kickoff ────────────────────────────────────────────────────────


def start_scrape(db, job: dict) -> None:
    """Start the first Apify run for a creator_ig_scrape job.

    On success this function persists ``payload.apify`` and raises
    :class:`JobPaused` so the worker leaves the job in ``running``.
    """
    payload = job.get("payload") or {}
    handle = payload.get("handle")
    if not handle:
        raise ValueError("creator_ig_scrape job missing handle")

    username = handle.strip().lstrip("@").lower().rstrip("/")
    profile_url = f"https://www.instagram.com/{username}/"

    client = make_default_client()
    webhook_url = _build_webhook_url(job["id"])

    actor_input = {
        "directUrls": [profile_url],
        "resultsType": "posts",
        "resultsLimit": RESULTS_LIMIT,
        # `addParentData=true` makes the actor emit the parent profile
        # alongside posts so we don't need a separate `details` run.
        "addParentData": True,
    }
    webhooks = [
        {
            "eventTypes": [
                "ACTOR.RUN.SUCCEEDED",
                "ACTOR.RUN.FAILED",
                "ACTOR.RUN.ABORTED",
                "ACTOR.RUN.TIMED_OUT",
            ],
            "requestUrl": webhook_url,
        }
    ]
    run = client.start_run(DEFAULT_ACTOR, actor_input, webhooks=webhooks)

    apify_state = {
        "stage": STAGE_AWAITING_POSTS,
        "run_id": run["id"],
        "dataset_id": run["defaultDatasetId"],
        "actor_id": DEFAULT_ACTOR,
        "username": username,
        "profile_url": profile_url,
    }
    new_payload = {**payload, "apify": apify_state}
    pdb.update_background_job_payload(db, job["id"], new_payload)
    logger.info(
        "IG scrape started: job=%s username=@%s run_id=%s",
        job["id"], username, run["id"],
    )
    raise JobPaused("awaiting apify posts run")


# ── Resume from webhook ─────────────────────────────────────────────────────


def resume(db, job: dict, *, expected_run_id: str | None = None) -> bool:
    """Advance the FSM. Returns True when the job is fully finished
    (the caller should call :func:`complete_background_job`), False
    when another Apify run was kicked off and we're now awaiting that
    webhook.

    ``expected_run_id`` is the runId that came in on the webhook — used
    to dedupe stale callbacks if Apify re-delivers a webhook for a run
    we already advanced past.
    """
    apify_state = (job.get("payload") or {}).get("apify") or {}
    stage = apify_state.get("stage")
    if stage == STAGE_AWAITING_POSTS:
        return _on_posts_ready(db, job, apify_state, expected_run_id)
    if stage == STAGE_AWAITING_COMMENTS:
        return _on_comments_ready(db, job, apify_state, expected_run_id)
    raise RuntimeError(f"resume: unknown FSM stage {stage!r}")


def _on_posts_ready(
    db, job: dict, apify_state: dict, expected_run_id: str | None
) -> bool:
    if expected_run_id and expected_run_id != apify_state.get("run_id"):
        logger.warning(
            "Webhook run_id mismatch (got %s, awaiting %s); ignoring",
            expected_run_id, apify_state.get("run_id"),
        )
        return False

    client = make_default_client()
    items = client.fetch_dataset_items(apify_state["dataset_id"])
    bd_profile, bd_posts, bd_reels = _split_posts_run(items)

    embedded_comments = sum(len(r.get("top_comments") or []) for r in bd_reels)
    commerce_signal = (job.get("payload") or {}).get("commerce_signal")
    need_comments = (
        not commerce_signal
        and embedded_comments < EMBEDDED_COMMENT_THRESHOLD
        and bool(bd_reels)
    )

    bundle: dict[str, Any] = {
        "profile": bd_profile,
        "posts": bd_posts,
        "reels": bd_reels,
        "comments": [],
    }

    if need_comments:
        top_reel_urls = [
            r["url"]
            for r in bd_reels[:TOP_REELS_FOR_COMMENTS]
            if r.get("url")
        ]
        if top_reel_urls:
            comment_input = {
                "directUrls": top_reel_urls,
                "resultsType": "comments",
                "resultsLimit": COMMENTS_PER_REEL,
                "addParentData": True,
            }
            webhooks = [
                {
                    "eventTypes": [
                        "ACTOR.RUN.SUCCEEDED",
                        "ACTOR.RUN.FAILED",
                        "ACTOR.RUN.ABORTED",
                        "ACTOR.RUN.TIMED_OUT",
                    ],
                    "requestUrl": _build_webhook_url(job["id"]),
                }
            ]
            run = client.start_run(
                apify_state["actor_id"], comment_input, webhooks=webhooks
            )
            new_state = {
                **apify_state,
                "stage": STAGE_AWAITING_COMMENTS,
                "run_id": run["id"],
                "dataset_id": run["defaultDatasetId"],
                "partial_bundle": bundle,
            }
            new_payload = {**(job.get("payload") or {}), "apify": new_state}
            pdb.update_background_job_payload(db, job["id"], new_payload)
            logger.info(
                "IG scrape: comments run started job=%s run_id=%s",
                job["id"], run["id"],
            )
            return False

    # No comments run needed — finalize.
    return _finish_scrape(db, job, bundle)


def _on_comments_ready(
    db, job: dict, apify_state: dict, expected_run_id: str | None
) -> bool:
    if expected_run_id and expected_run_id != apify_state.get("run_id"):
        logger.warning(
            "Webhook run_id mismatch (got %s, awaiting %s); ignoring",
            expected_run_id, apify_state.get("run_id"),
        )
        return False

    client = make_default_client()
    items = client.fetch_dataset_items(apify_state["dataset_id"])
    bd_comments = [
        apify_instagram_bundle._translate_comment(c) for c in items or []
    ]
    bundle = dict(apify_state.get("partial_bundle") or {})
    bundle["comments"] = bd_comments
    return _finish_scrape(db, job, bundle)


# ── Finalize: pre-populate bundle cache and run the rest of the pipeline ─────


def _finish_scrape(db, job: dict, bundle: dict[str, Any]) -> bool:
    """Pre-populate the bundle cache and run the existing post-scrape
    pipeline (transcripts → LLM → score → store → embed → match).

    The trick: ``apify_instagram_bundle._CACHE`` is process-global. We
    stuff the webhook-fetched bundle into it, then call
    ``build_creator_intelligence_profile`` which will hit the cache and
    skip its own Apify calls.
    """
    from pipeline.handlers import _finalize_creator_ig_scrape

    apify_state = (job.get("payload") or {}).get("apify") or {}
    username = apify_state.get("username") or ""

    # Pre-populate the cache so the existing orchestrator's scrape_* calls
    # all become no-op cache reads.
    apify_instagram_bundle._CACHE[username] = bundle

    try:
        _finalize_creator_ig_scrape(db, job)
    finally:
        # Clear the cache slot so the next creator's run doesn't see stale data.
        apify_instagram_bundle.clear(username)
    return True


# ── Helpers ─────────────────────────────────────────────────────────────────


def _split_posts_run(
    items: list[dict],
) -> tuple[dict | None, list[dict], list[dict]]:
    """Split a posts-run-with-addParentData payload into (profile, posts, reels).

    With ``addParentData=true`` the actor emits a profile-shaped object as
    one of the items (no ``type`` field, has ``username``+``followersCount``).
    The other items are post records with a ``type`` field.
    """
    profile_raw: dict | None = None
    posts_raw: list[dict] = []
    for item in items or []:
        if "type" in item and item.get("type") in {"Video", "Image", "Sidecar"}:
            posts_raw.append(item)
        elif item.get("username") and item.get("followersCount") is not None:
            if profile_raw is None:
                profile_raw = item
        else:
            posts_raw.append(item)

    bd_profile = (
        apify_instagram_bundle._translate_profile(profile_raw)
        if profile_raw
        else None
    )

    bd_posts: list[dict] = []
    bd_reels: list[dict] = []
    for raw in posts_raw:
        translated = apify_instagram_bundle._translate_post(raw)
        if translated.get("content_type") == "Video":
            bd_reels.append(translated)
        else:
            bd_posts.append(translated)

    return bd_profile, bd_posts[:NUM_POSTS], bd_reels[:NUM_REELS]


def _build_webhook_url(job_id: str) -> str:
    base = os.environ.get("WORKER_PUBLIC_URL")
    secret = os.environ.get("APIFY_WEBHOOK_SECRET")
    if not base:
        raise RuntimeError(
            "WORKER_PUBLIC_URL not configured; required for Apify webhooks"
        )
    if not secret:
        raise RuntimeError(
            "APIFY_WEBHOOK_SECRET not configured; required for Apify webhooks"
        )
    return f"{base.rstrip('/')}/apify-webhook?job_id={job_id}&secret={secret}"


# ── Recovery sweep ──────────────────────────────────────────────────────────


def recover_stale_run(db, job: dict) -> bool:
    """One-shot poll for jobs whose webhook never arrived.

    Returns True when the job advanced (caller should re-check status),
    False when the Apify run is still in flight (job stays running).
    Raises on terminal-failed Apify runs so the caller can mark the
    background job failed.
    """
    apify_state = (job.get("payload") or {}).get("apify") or {}
    run_id = apify_state.get("run_id")
    if not run_id:
        return False

    client = make_default_client()
    run = client.get_run_status(run_id)
    status = (run.get("status") or "").upper()
    if status == "SUCCEEDED":
        logger.warning(
            "Recovery: webhook never arrived for job=%s run=%s; resuming via polling",
            job["id"], run_id,
        )
        return resume(db, job, expected_run_id=run_id)
    if status in {"FAILED", "ABORTED", "TIMED-OUT", "TIMED_OUT"}:
        raise RuntimeError(
            f"Apify run {run_id} ended with status {status}: "
            f"{run.get('statusMessage') or run.get('exitCode')}"
        )
    # Still running — leave the job parked.
    return False
