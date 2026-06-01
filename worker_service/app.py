"""Modal entry point for the pipeline-worker FastAPI app.

Wraps the existing `pipeline.api` ASGI app — no behaviour change. Same
routes, same handler dispatch, same Supabase / Apify / LLM / Modal
Whisper integrations. The lift-and-shift target for the Railway →
Modal migration (Phase 4).

Deploy:
    modal deploy pipeline/worker_service/app.py

The deploy command prints a stable HTTPS URL such as
    https://<workspace>--pipeline-worker-fastapi-app.modal.run
which exposes the existing FastAPI routes verbatim. Repoint the
Supabase pg_cron jobs and the Apify webhook receivers at that URL
(see migration 076 + DEPLOY.md) and Railway can be turned off.
"""

from __future__ import annotations

import os

try:
    import modal  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - allow imports without modal installed
    modal = None  # type: ignore[assignment]


if modal is not None:
    # Project root is one level up from `pipeline/`. Modal ships the
    # whole `pipeline/` directory into the container; the FastAPI app
    # imports its handlers from there.
    image = (
        modal.Image.debian_slim(python_version="3.11")
        # Base deps mirror Railway: same versions, same pins. Source of
        # truth lives at pipeline/requirements.txt.
        .pip_install_from_requirements("pipeline/requirements.txt")
        # Extras the handlers reach for that aren't in
        # pipeline/requirements.txt because the Railway deploy also
        # picked them up from the root requirements.txt. Listing them
        # explicitly here keeps the Modal build self-contained.
        .pip_install(
            "google-api-python-client>=2.0",
            "google-auth>=2.0",
            "google-auth-httplib2",
            "google-auth-oauthlib",
            "requests>=2.31",
            "youtube-transcript-api>=0.6",
            "numpy",
        )
        # Ship the entire `pipeline/` package into the container so
        # `pipeline.api.app` (and every transitive import in
        # `pipeline.handlers`, scrapers, llm_*.py, etc.) loads cleanly.
        .add_local_dir("pipeline", remote_path="/root/pipeline")
    )

    app = modal.App("pipeline-worker")

    @app.function(
        image=image,
        secrets=[
            # One secret per concern keeps rotation surgical. See
            # pipeline/worker_service/DEPLOY.md for the env-var → secret map.
            modal.Secret.from_name("pipeline-supabase"),
            modal.Secret.from_name("pipeline-apify"),
            modal.Secret.from_name("pipeline-openai"),
            modal.Secret.from_name("pipeline-gemini"),
            modal.Secret.from_name("pipeline-anthropic"),
            modal.Secret.from_name("pipeline-youtube"),
            modal.Secret.from_name("pipeline-modal-whisper"),
            modal.Secret.from_name("pipeline-worker-auth"),
        ],
        # pg_cron ticks every 30s. min_containers=1 keeps a container
        # warm so cron POSTs never pay a cold-start penalty.
        min_containers=1,
        # Steady-state load is light (one job/tick). 3 caps the spend
        # if /process-next-job ever takes longer than the tick interval
        # under retry storms.
        max_containers=3,
        # Per-call timeout. Match the longest STALE_JOB_TIMEOUTS entry:
        # brand_ig_scrape can run 30+ min when Whisper transcription
        # queues lag. 45 min ceiling matches the Python-side default.
        timeout=2700,
        cpu=2,
        memory=4096,
    )
    @modal.asgi_app()
    def fastapi_app():  # pragma: no cover - runs on Modal
        """Expose `pipeline.api.app` as a Modal HTTPS endpoint.

        The function returns the FastAPI instance once per container
        start; Modal's runtime handles request fan-out from there.
        """
        # Import inside the function so this module loads cleanly off
        # Modal too (e.g. for syntax checks / pytest in CI).
        from pipeline.api import app as fastapi_application

        # Sanity: cron + apify both authenticate via X-Worker-Secret
        # or ?secret= URL param. If PIPELINE_WORKER_SECRET isn't set
        # the worker will accept all requests — log loud so misconfig
        # is obvious in Modal logs.
        if not os.environ.get("PIPELINE_WORKER_SECRET"):
            print(
                "[pipeline-worker] WARNING: PIPELINE_WORKER_SECRET "
                "unset — routes will skip auth checks",
                flush=True,
            )
        return fastapi_application
