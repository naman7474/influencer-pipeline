"""Modal app entry point for the discovery service.

Deploy:
    modal deploy pipeline/discovery_service/app.py

Invoke from the web layer:
    import modal
    fn = modal.Function.lookup("discovery", "run_discovery")
    call = fn.spawn(request_id="...")
    # Persist call.object_id on discovery_requests.modal_call_id for resume.

Environment / secrets (Modal Secret name → env var inside the container):
    discovery-supabase     →  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    discovery-openai       →  OPENAI_API_KEY
    discovery-gemini       →  GEMINI_API_KEY
    discovery-youtube      →  YOUTUBE_API_KEYS (12 comma-separated keys)
    discovery-callback     →  DISCOVERY_WEB_BASE_URL, DISCOVERY_SERVICE_SECRET

`discovery-callback` is the shared secret the brand-match compute
endpoint on the Next.js side verifies, plus the base URL it lives at.
"""

from __future__ import annotations

import logging
import os
import traceback

try:
    import modal  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - allow stages.py imports w/o modal
    modal = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

# ── Image ───────────────────────────────────────────────────────────
# Build from the project root so all `pipeline/` modules are importable.
# Pinning to .from_dockerfile would be cleaner long-term; for v1 we lean
# on Modal's default Python + a pip install list mirroring requirements.txt.

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            # Core deps used across the discovery pipeline
            "supabase>=2.4",
            "httpx>=0.27",
            "google-api-python-client>=2.0",
            "google-auth>=2.0",
            "google-auth-httplib2",
            "google-auth-oauthlib",
            "google-genai>=0.3",
            "openai>=1.0",
            "youtube-transcript-api>=0.6",
            "pydantic>=2.0",
            "fastapi>=0.110",
            # Embeddings / vectors
            "numpy",
        )
        # Bring the existing `pipeline/` package into the container so we
        # can call build_youtube_creator_intelligence_profile + store_youtube_cip
        # without re-implementing anything.
        .add_local_dir(
            "pipeline",
            remote_path="/root/pipeline",
        )
    )

    app = modal.App("discovery")

    @app.function(
        image=image,
        secrets=[modal.Secret.from_name("discovery-callback")],
        # Thin orchestration: validate secret + spawn the heavy worker.
        # Returns in <1s.
        timeout=30,
    )
    @modal.fastapi_endpoint(method="POST")
    def enqueue(item: dict) -> dict:  # pragma: no cover - runs on Modal
        """HTTP entry point invoked by the Next.js POST endpoint.

        Body: `{"request_id": "<uuid>"}`. Header:
        `X-Discovery-Service-Secret: <shared secret>`.

        Returns `{"call_id": "<modal-FunctionCall-object-id>"}`. The
        web layer persists this on `discovery_requests.modal_call_id`
        so we can re-attach to a running call after a redeploy.
        """
        from fastapi import HTTPException, Request  # noqa: F401  (Modal injects)

        secret_env = os.environ.get("DISCOVERY_SERVICE_SECRET", "")
        # `item` is the FastAPI-parsed JSON body. Modal exposes the raw
        # request as a special parameter if we want headers — but for v1
        # we accept the secret inside the JSON payload too, since the
        # FastAPI binding for `Request` is awkward via Modal. The Next.js
        # caller sends it both ways.
        provided = item.get("secret") if isinstance(item, dict) else None
        if not secret_env or provided != secret_env:
            return {"error": "unauthorized"}

        request_id = (
            item.get("request_id") if isinstance(item, dict) else None
        )
        if not isinstance(request_id, str) or len(request_id) == 0:
            return {"error": "request_id required"}

        # Route by platform. Default = YouTube to keep backward-compat
        # with the original Phase 3 callers that don't send the field.
        # Web POST → /api/discover/requests sends `filters.platform`
        # which we forward here as a top-level `platform` field for
        # easy dispatching.
        platform = (
            (item.get("platform") or "").lower()
            if isinstance(item, dict)
            else ""
        )
        if platform == "instagram":
            call = run_discovery_instagram.spawn(request_id=request_id)
        else:
            call = run_discovery.spawn(request_id=request_id)
        return {"call_id": call.object_id, "platform": platform or "youtube"}

    @app.function(
        image=image,
        secrets=[
            modal.Secret.from_name("discovery-supabase"),
            modal.Secret.from_name("discovery-openai"),
            modal.Secret.from_name("discovery-gemini"),
            modal.Secret.from_name("discovery-youtube"),
            modal.Secret.from_name("discovery-callback"),
        ],
        # 200 creators × ~30s with parallelism=50 → ~2-3 min steady state.
        # Whisper cold start can add up to ~120s. 15 min gives generous slack
        # for the long tail (slow LLM responses, retries, etc).
        timeout=900,
        # One container per discovery request — internal parallelism is via
        # ThreadPoolExecutor inside the function, not horizontal scale.
        # Bump max_containers if we need to run multiple discoveries
        # simultaneously (Phase 4: bigger orgs).
        max_containers=10,
        # CPU-light + memory-medium. Most time is spent on outbound HTTP
        # to YouTube / Gemini / OpenAI. 4 CPU is enough for 50 threads.
        cpu=4,
        memory=4096,
    )
    def run_discovery(request_id: str) -> dict:  # pragma: no cover - runs on Modal
        """Orchestrate a single discovery request end-to-end.

        Idempotent on `status='queued' | 'failed'`: a retry from the web
        layer starts over from the search stage. Skipping stages on
        partial-state retries is deferred — for v1, we'd rather burn
        ~500 quota units re-searching than build the resume logic.

        Returns `{request_id, status, candidates_matched}` for the
        web layer's logs (Modal also persists this in the FunctionCall).
        """
        # Imports inside the function so the module imports cleanly off
        # Modal (e.g. for tests, or when the web layer imports just to
        # call `modal.Function.lookup`).
        from supabase import create_client

        from pipeline.discovery_service.stages import (
            stage_brand_match,
            stage_complete,
            stage_failed,
            stage_filter,
            stage_parallel_scrape,
            stage_search,
        )
        from pipeline.youtube.api_pool import YouTubeAPIPool

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        supabase_url = os.environ["SUPABASE_URL"]
        supabase_key = (
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("SUPABASE_SERVICE_KEY", "")
        )
        if not supabase_url or not supabase_key:
            raise RuntimeError(
                "Discovery service requires SUPABASE_URL + "
                "SUPABASE_SERVICE_ROLE_KEY in the discovery-supabase secret"
            )
        db = create_client(supabase_url, supabase_key)

        # ── Load the request row ──
        try:
            res = (
                db.table("discovery_requests")
                .select("id, brand_id, query_text, filters, status")
                .eq("id", request_id)
                .limit(1)
                .execute()
            )
        except Exception as e:
            raise RuntimeError(
                f"failed to load discovery_requests {request_id}: {e}"
            )
        if not res.data:
            raise RuntimeError(f"discovery_requests {request_id} not found")
        req = res.data[0]

        # Idempotency guard — already in-progress? Already done?
        if req["status"] not in ("queued", "failed"):
            logger.info(
                f"discovery {request_id} already in status={req['status']}; "
                "no-op"
            )
            return {
                "request_id": request_id,
                "status": req["status"],
                "candidates_matched": None,
            }

        brand_id = req["brand_id"]
        query = req["query_text"]
        filters = req.get("filters") or {}
        # Optional regional bias from the filters JSONB (matches the
        # /api/discover/search payload shape). Defaults: no region bias.
        region_code: str | None = filters.get("region_code") or None

        pool = YouTubeAPIPool()
        if not pool.available:
            stage_failed(db, request_id, "youtube api keys exhausted or absent")
            return {
                "request_id": request_id,
                "status": "failed",
                "candidates_matched": 0,
            }

        try:
            # 1. Search (channel + video lanes, merge + dedup)
            candidates = stage_search(
                db,
                request_id,
                query,
                pool=pool,
                channel_max=int(os.environ.get("DISCOVERY_CHANNEL_MAX", "200")),
                video_max=int(os.environ.get("DISCOVERY_VIDEO_MAX", "200")),
                region_code=region_code,
            )
            if not candidates:
                stage_failed(
                    db,
                    request_id,
                    "no candidates found for query",
                )
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "candidates_matched": 0,
                }

            # 2. Filter (competitor names + already-in-DB + user's tier/
            #    follower chips). Passing `pool` enables the cheap
            #    channels.list profile fetch needed for tier classification.
            survivors, existing_ids = stage_filter(
                db,
                request_id,
                brand_id,
                candidates,
                pool=pool,
                user_filters=filters,
            )

            # 3. Per-creator deep scrape in parallel.
            #
            # 15 threads is a deliberate choice: each creator triggers
            # ~10 Supabase REST writes (creators, creator_scores,
            # caption/audience/transcript intelligence, social_profile,
            # youtube_videos, creator_embeddings, tag update). With 50
            # threads that's ~500 concurrent HTTP connections against
            # PostgREST, and we observed wholesale "Server disconnected"
            # errors as the pool saturated and Supabase started dropping
            # idle conns. 15 threads keeps us well under PostgREST's
            # default per-IP cap and the success rate jumps from ~14%
            # to >90%.
            new_creator_ids = stage_parallel_scrape(
                db,
                request_id,
                brand_id,
                survivors,
                parallelism=int(
                    os.environ.get("DISCOVERY_PARALLELISM", "15")
                ),
                num_videos=int(
                    os.environ.get("DISCOVERY_NUM_VIDEOS", "5")
                ),
                num_transcripts=int(
                    os.environ.get("DISCOVERY_NUM_TRANSCRIPTS", "5")
                ),
            )

            # ── Also include pre-existing creators in the brand-match pass:
            # the user clicked Search for X niche; they expect to see ALL
            # X-niche creators ranked by brand fit, not just the newly
            # discovered ones. Look up creator_ids for the existing channels.
            existing_creator_ids = _resolve_existing_creator_ids(
                db, existing_ids
            )
            all_creator_ids = list(
                set(new_creator_ids) | set(existing_creator_ids)
            )

            # Tag existing creators with this discovery_request_id too so
            # the GET /api/discover/requests/:id/creators endpoint surfaces
            # them. Without this, the endpoint filters by
            # discovery_request_id (which only newly-scraped creators have)
            # and the user sees an empty results screen even when the
            # match-batch step succeeded.
            _tag_existing_creators(db, existing_creator_ids, request_id)

            # 4. Brand-match scoring via Next.js endpoint
            stage_brand_match(db, request_id, brand_id, all_creator_ids)

            # 5. Done
            stage_complete(db, request_id)
            return {
                "request_id": request_id,
                "status": "completed",
                "candidates_matched": len(all_creator_ids),
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"discovery {request_id} crashed: {e}\n{tb}")
            stage_failed(db, request_id, f"unhandled: {e}")
            return {
                "request_id": request_id,
                "status": "failed",
                "candidates_matched": 0,
            }

    @app.function(
        image=image,
        secrets=[
            modal.Secret.from_name("discovery-supabase"),
            modal.Secret.from_name("discovery-openai"),
            modal.Secret.from_name("discovery-gemini"),
            modal.Secret.from_name("discovery-apify"),
            modal.Secret.from_name("discovery-callback"),
        ],
        # IG discovery is slower per-creator than YT (Apify actor runs
        # take 30-90s each vs ~5s for a YT API call), but we run lower
        # parallelism (10) to stay under Apify's per-token rate limit.
        # 100 creators × ~60s / 10 threads ≈ 10 min — keep the timeout
        # generous for the long tail.
        timeout=1500,
        max_containers=10,
        cpu=2,
        memory=2048,
    )
    def run_discovery_instagram(request_id: str) -> dict:  # pragma: no cover
        """Orchestrate one Instagram discovery request end-to-end.

        Mirrors the YT path but uses Apify (search + bundle) instead of
        the YouTube Data API. Comments are explicitly skipped — see
        `pipeline.discovery_service.instagram.stages._scrape_one_ig_creator`.

        Returns `{request_id, status, candidates_matched, platform}`.
        """
        from supabase import create_client

        from pipeline.discovery_service.instagram.stages import (
            resolve_existing_ig_creator_ids,
            stage_brand_match_ig,
            stage_complete,
            stage_failed,
            stage_filter_ig,
            stage_parallel_scrape_ig,
            stage_search_ig,
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        supabase_url = os.environ["SUPABASE_URL"]
        supabase_key = (
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ.get("SUPABASE_SERVICE_KEY", "")
        )
        if not supabase_url or not supabase_key:
            raise RuntimeError(
                "IG discovery requires SUPABASE_URL + "
                "SUPABASE_SERVICE_ROLE_KEY in the discovery-supabase secret"
            )
        db = create_client(supabase_url, supabase_key)

        # Load + idempotency guard (same as YT path).
        try:
            res = (
                db.table("discovery_requests")
                .select("id, brand_id, query_text, filters, status")
                .eq("id", request_id)
                .limit(1)
                .execute()
            )
        except Exception as e:
            raise RuntimeError(
                f"failed to load discovery_requests {request_id}: {e}"
            )
        if not res.data:
            raise RuntimeError(f"discovery_requests {request_id} not found")
        req = res.data[0]
        if req["status"] not in ("queued", "failed"):
            logger.info(
                f"ig discovery {request_id} already status={req['status']}; "
                "no-op"
            )
            return {
                "request_id": request_id,
                "status": req["status"],
                "candidates_matched": None,
                "platform": "instagram",
            }

        brand_id = req["brand_id"]
        query = req["query_text"]
        filters = req.get("filters") or {}

        try:
            # 1. Search via Apify
            candidates = stage_search_ig(
                db,
                request_id,
                query,
                max_results=int(
                    os.environ.get("DISCOVERY_IG_MAX_RESULTS", "100")
                ),
            )
            if not candidates:
                stage_failed(
                    db,
                    request_id,
                    "no Instagram candidates found for query",
                )
                return {
                    "request_id": request_id,
                    "status": "failed",
                    "candidates_matched": 0,
                    "platform": "instagram",
                }

            # 2. Filter (competitor + already-in-DB + tier)
            survivors, existing_handles = stage_filter_ig(
                db,
                request_id,
                brand_id,
                candidates,
                user_filters=filters,
            )

            # 3. Deep scrape new candidates
            new_creator_ids = stage_parallel_scrape_ig(
                db,
                request_id,
                brand_id,
                survivors,
                parallelism=int(
                    os.environ.get("DISCOVERY_IG_PARALLELISM", "10")
                ),
                num_posts=int(
                    os.environ.get("DISCOVERY_IG_NUM_POSTS", "5")
                ),
                num_reels=int(
                    os.environ.get("DISCOVERY_IG_NUM_REELS", "10")
                ),
            )

            # 4. Resolve + tag pre-existing creators (mirror YT logic).
            existing_creator_ids = resolve_existing_ig_creator_ids(
                db, existing_handles
            )
            all_creator_ids = list(
                set(new_creator_ids) | set(existing_creator_ids)
            )
            _tag_existing_creators(db, existing_creator_ids, request_id)

            # 5. Brand-match scoring via Next.js compute-batch
            stage_brand_match_ig(
                db, request_id, brand_id, all_creator_ids
            )

            # 6. Done
            stage_complete(db, request_id)
            return {
                "request_id": request_id,
                "status": "completed",
                "candidates_matched": len(all_creator_ids),
                "platform": "instagram",
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"ig discovery {request_id} crashed: {e}\n{tb}")
            stage_failed(db, request_id, f"unhandled: {e}")
            return {
                "request_id": request_id,
                "status": "failed",
                "candidates_matched": 0,
                "platform": "instagram",
            }


def _resolve_existing_creator_ids(db, channel_ids):  # pragma: no cover - thin
    """Map a set of YT channel IDs to creators.id via creator_social_profiles."""
    ids = list(channel_ids)
    out: list[str] = []
    if not ids:
        return out
    for i in range(0, len(ids), 100):
        batch = ids[i : i + 100]
        try:
            res = (
                db.table("creator_social_profiles")
                .select("creator_id, platform_user_id")
                .eq("platform", "youtube")
                .in_("platform_user_id", batch)
                .execute()
            )
        except Exception:
            continue
        for row in res.data or []:
            cid = row.get("creator_id")
            if cid:
                out.append(cid)
    return out


def _tag_existing_creators(db, creator_ids, request_id):  # pragma: no cover - thin
    """Mark existing creators as having been surfaced by this discovery.

    The GET /api/discover/requests/:id/creators endpoint filters by
    `discovery_request_id`; without this tag, creators that matched the
    keyword but were already in DB don't appear in the result set, even
    though they were brand-match-scored. Idempotent — overwriting an
    earlier request_id is acceptable for v1 (last-discovery-wins).
    """
    from datetime import datetime as _dt

    ids = list(set(c for c in creator_ids if c))
    if not ids:
        return
    now = _dt.utcnow().isoformat()
    for i in range(0, len(ids), 100):
        batch = ids[i : i + 100]
        try:
            db.table("creators").update(
                {
                    "discovery_request_id": request_id,
                    "discovered_at": now,
                }
            ).in_("id", batch).execute()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tag existing creators batch failed: {e}")
