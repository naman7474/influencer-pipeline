"""On-demand YouTube creator discovery service (Phase 3).

Standalone Modal-hosted FastAPI/Python app that turns a user's free-text
search query into a deep-scraped, brand-match-scored set of ~200 creators
in ~3-5 min. Runs OFF the existing pg_cron worker queue so its concurrency
profile (50 threads burst) doesn't compete with nightly batch jobs.

Entry point: `pipeline.discovery_service.app.run_discovery`. The web layer
invokes it via `modal.Function.lookup("discovery", "run_discovery").spawn(
request_id=...)` after inserting a `discovery_requests` row.

See /Users/namanjain/.claude/plans/we-have-got-a-valiant-giraffe.md for
the architectural rationale and the full per-stage timing budget.
"""
