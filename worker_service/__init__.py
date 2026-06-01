"""Modal-hosted pipeline worker (Phase 4).

Lift-and-shift Modal home for the existing FastAPI worker at
`pipeline.api` so we can decommission Railway. No handler refactor —
the same 5 routes (`/health`, `/enqueue`, `/process-next-job`,
`/apify-webhook`, `/recover-stale-jobs`) and the same `pipeline.handlers`
dispatch run unchanged on Modal.

See `pipeline/worker_service/DEPLOY.md` for the secrets list, deploy
command, pg_cron repoint (migration 076), and Apify webhook URL switch.
"""
