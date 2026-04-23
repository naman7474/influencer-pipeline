"""
FastAPI worker that drains `background_jobs` for the IG analysis pipeline.

Stateless by design: Supabase pg_cron hits `/process-next-job` every 30s
(via the `trigger-pipeline-worker` Edge Function) and this handler claims
and runs exactly one job per tick. That pushes all scheduling into Supabase
(monitoring + reliability for free) and keeps the Python side as a simple
request handler.

Endpoints:
  POST /enqueue            — idempotent job insert (called by web layer)
  POST /process-next-job   — cron tick: claim + run one queued job
  POST /recover-stale-jobs — every 5m: re-queue jobs stuck in running past
                             their per-job-type timeout (see JOB_TYPE_TIMEOUTS)
  GET  /health             — liveness

Auth: all mutating endpoints require X-Worker-Secret == PIPELINE_WORKER_SECRET.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from pipeline import db as pdb
from pipeline.handlers import dispatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline.api")

import json as _stale_json

# Per-job-type stale-running timeouts. Full pipeline runs can take 30+ min
# (Brightdata poll timeout is 900s per scrape × several scrapes), so a
# single hardcoded 15 min ceiling spuriously recovered healthy jobs.
DEFAULT_STALE_MINUTES = 30
JOB_TYPE_TIMEOUTS: dict[str, int] = {
    "brand_ig_scrape": 45,
    "creator_ig_scrape": 45,
    "content_video_analysis": 20,
    "shopify_sync": 15,
    "shopify_geo_sync": 30,
    "brand_matching": 10,
    "creator_match_recompute": 10,
}

# Env override: STALE_JOB_TIMEOUTS_JSON='{"brand_ig_scrape": 60, ...}'
_override = os.environ.get("STALE_JOB_TIMEOUTS_JSON")
if _override:
    try:
        JOB_TYPE_TIMEOUTS.update(_stale_json.loads(_override))
    except Exception:  # pragma: no cover
        logger.warning("Invalid STALE_JOB_TIMEOUTS_JSON; ignoring")

app = FastAPI(title="Influencer Pipeline Worker")


# ── Client factory ──────────────────────────────────────────────────────────

def _get_db():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not configured",
        )
    return pdb.init_supabase(url, key)


# ── Auth ────────────────────────────────────────────────────────────────────

def _auth(x_worker_secret: str | None = Header(default=None)) -> None:
    expected = os.environ.get("PIPELINE_WORKER_SECRET")
    if not expected:
        raise HTTPException(
            status_code=500, detail="PIPELINE_WORKER_SECRET not configured"
        )
    if not x_worker_secret or x_worker_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid worker secret")


# ── Models ──────────────────────────────────────────────────────────────────

class EnqueueRequest(BaseModel):
    job_type: str
    brand_id: str
    payload: dict[str, Any] | None = None


class EnqueueResponse(BaseModel):
    job_id: str
    created: bool


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/enqueue", response_model=EnqueueResponse)
def enqueue(req: EnqueueRequest, _: None = Depends(_auth)) -> EnqueueResponse:
    """Idempotent insert — if a non-terminal job of the same type exists for
    the same brand, return that job_id instead of creating a duplicate."""
    db = _get_db()

    existing = (
        db.table("background_jobs")
        .select("id,status")
        .eq("brand_id", req.brand_id)
        .eq("job_type", req.job_type)
        .in_("status", ["queued", "running"])
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if existing.data:
        return EnqueueResponse(job_id=existing.data[0]["id"], created=False)

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "job_type": req.job_type,
        "brand_id": req.brand_id,
        "status": "queued",
        "payload": req.payload or {},
        "available_at": now,
    }
    inserted = db.table("background_jobs").insert(row).execute()
    job_id = inserted.data[0]["id"]
    logger.info(f"Enqueued {req.job_type} for brand {req.brand_id} -> {job_id}")
    return EnqueueResponse(job_id=job_id, created=True)


@app.post("/process-next-job")
def process_next_job(_: None = Depends(_auth)) -> Response:
    """Cron tick. Claims one queued job and runs it to completion.
    Returns 204 when nothing is available."""
    db = _get_db()

    jobs = pdb.get_runnable_background_jobs(db, limit=1)
    if not jobs:
        return Response(status_code=204)

    job = jobs[0]
    claimed = pdb.claim_background_job(db, job["id"])
    if not claimed:
        # Raced with another worker; no-op.
        return Response(status_code=204)

    job_id = claimed["id"]
    job_type = claimed["job_type"]
    logger.info(f"Claimed job {job_id} ({job_type})")

    try:
        dispatch(db, claimed)
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        pdb.fail_background_job(db, job_id, str(e))
        # Mark brand status=failed for brand jobs so the UI can surface an error.
        if job_type == "brand_ig_scrape" and claimed.get("brand_id"):
            db.table("brands").update(
                {
                    "ig_analysis_status": "failed",
                    "ig_analysis_error": str(e)[:500],
                }
            ).eq("id", claimed["brand_id"]).execute()
        raise HTTPException(status_code=500, detail=f"Job {job_id} failed: {e}")

    pdb.complete_background_job(db, job_id)
    logger.info(f"Completed job {job_id}")
    return Response(
        status_code=200,
        content=f'{{"job_id":"{job_id}","status":"succeeded"}}',
        media_type="application/json",
    )


@app.post("/recover-stale-jobs")
def recover_stale_jobs(_: None = Depends(_auth)) -> dict[str, Any]:
    """Re-queue jobs stuck in `running` past their per-job-type timeout.

    Runs one lookup per job_type so we don't apply the same cutoff to a
    fast `brand_matching` job and a slow `brand_ig_scrape` that legitimately
    takes half an hour.
    """
    db = _get_db()
    now_ts = datetime.now(timezone.utc)
    recovered_by_type: dict[str, int] = {}
    total = 0

    known_types = set(JOB_TYPE_TIMEOUTS.keys())

    # Process each configured job_type with its own cutoff.
    for job_type, minutes in JOB_TYPE_TIMEOUTS.items():
        cutoff = (now_ts - timedelta(minutes=minutes)).isoformat()
        stale = (
            db.table("background_jobs")
            .select("id")
            .eq("status", "running")
            .eq("job_type", job_type)
            .lte("locked_at", cutoff)
            .execute()
        )
        ids = [r["id"] for r in (stale.data or [])]
        if not ids:
            continue
        db.table("background_jobs").update(
            {
                "status": "queued",
                "locked_at": None,
                "locked_by": None,
                "updated_at": now_ts.isoformat(),
            }
        ).in_("id", ids).execute()
        recovered_by_type[job_type] = len(ids)
        total += len(ids)

    # Catch-all for unknown job_types so a misconfigured row isn't stuck
    # forever. Uses DEFAULT_STALE_MINUTES.
    default_cutoff = (
        now_ts - timedelta(minutes=DEFAULT_STALE_MINUTES)
    ).isoformat()
    catchall = (
        db.table("background_jobs")
        .select("id, job_type")
        .eq("status", "running")
        .lte("locked_at", default_cutoff)
        .execute()
    )
    unknown_ids = [
        r["id"]
        for r in (catchall.data or [])
        if r.get("job_type") not in known_types
    ]
    if unknown_ids:
        db.table("background_jobs").update(
            {
                "status": "queued",
                "locked_at": None,
                "locked_by": None,
                "updated_at": now_ts.isoformat(),
            }
        ).in_("id", unknown_ids).execute()
        recovered_by_type["_unknown"] = len(unknown_ids)
        total += len(unknown_ids)

    if total:
        logger.warning(
            f"Recovered {total} stale jobs: {recovered_by_type}"
        )
    return {"recovered": total, "by_type": recovered_by_type}
