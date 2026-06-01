# Pipeline worker — Modal deploy runbook

The existing FastAPI worker at `pipeline.api.app` is moving off Railway
onto Modal as part of Phase 4. This document is the step-by-step for
that migration.

The behaviour is identical to Railway: same routes (`/health`,
`/enqueue`, `/process-next-job`, `/apify-webhook`, `/recover-stale-jobs`),
same handler dispatch, same Supabase / Apify / LLM / Modal Whisper
integrations. Only the host changes.

---

## 1. Create Modal secrets

Modal scopes secrets per-workspace. Each command below creates one
secret bundling related env vars. **All values come from the existing
Railway dashboard's env tab** — copy verbatim, don't regenerate.

```bash
# Supabase
modal secret create pipeline-supabase \
  SUPABASE_URL='<from-railway>' \
  SUPABASE_SERVICE_ROLE_KEY='<from-railway>'

# Apify (token + actor IDs + webhook config)
modal secret create pipeline-apify \
  APIFY_TOKEN='<from-railway>' \
  APIFY_ACTOR_INSTAGRAM='<from-railway-or-default-apify/instagram-scraper>' \
  APIFY_ACTOR_IG_POST='<from-railway-or-default-apify/instagram-post-scraper>' \
  APIFY_IG_DM_SEND_ACTOR='<from-railway-or-default-am_production/instagram-direct-messages-dms-automation>' \
  APIFY_WEBHOOK_SECRET='<from-railway>' \
  APIFY_WEBHOOKS='1' \
  APIFY_WEBHOOK_MAX_MIN='25' \
  APIFY_ACTOR_BD_NATIVE='<from-railway-or-empty>'

# LLM API keys
modal secret create pipeline-openai     OPENAI_API_KEY='<from-railway>'
modal secret create pipeline-gemini     GEMINI_API_KEY='<from-railway>'
modal secret create pipeline-anthropic  ANTHROPIC_API_KEY='<from-railway>'

# YouTube Data API — the same 12-key pool the discovery service uses.
# Comma-separated, no spaces.
modal secret create pipeline-youtube \
  YOUTUBE_API_KEYS='key1,key2,...,key12'

# Modal Whisper sidecar (the worker calls into a separate Modal app
# for ASR; the function ids live here).
modal secret create pipeline-modal-whisper \
  MODAL_TOKEN_ID='<from-railway>' \
  MODAL_TOKEN_SECRET='<from-railway>' \
  WHISPER_MODAL_APP_NAME='whisper-transcribe' \
  WHISPER_MODAL_FUNCTION_NAME='transcribe'

# Worker auth + behaviour flags. WORKER_PUBLIC_URL must be the Modal
# URL you'll get back from `modal deploy` below — see step 3.
modal secret create pipeline-worker-auth \
  PIPELINE_WORKER_SECRET='<from-railway>' \
  WORKER_PUBLIC_URL='<TBD-fill-after-deploy>' \
  STALE_JOB_TIMEOUTS_JSON='<from-railway-or-leave-default>' \
  PIPELINE_USE_TX_RPC='<from-railway-or-0>' \
  LLM_MERGED='<from-railway-or-0>' \
  WHISPER_ASYNC='<from-railway-or-0>'
```

Verify with `modal secret list` — all 8 should appear under your workspace.

---

## 2. Deploy the Modal app

From the project root:

```bash
cd "/Users/namanjain/Documents/influencer copy"
modal deploy pipeline/worker_service/app.py
```

The deploy output prints the stable URL of the FastAPI ASGI endpoint:

```
✓ App deployed
└─ fastapi_app: created web function at
     https://<workspace>--pipeline-worker-fastapi-app.modal.run
```

**Copy that URL** — it's the new home for everything pg_cron and Apify
used to POST to on Railway.

---

## 3. Update `WORKER_PUBLIC_URL` in the auth secret

The Apify webhook builder (`pipeline/ig.py::_build_webhook_url`) reads
`WORKER_PUBLIC_URL` to construct the callback URL Apify hits when a
scrape run finishes. After step 2, refresh the secret with the real URL:

```bash
modal secret create pipeline-worker-auth --force \
  PIPELINE_WORKER_SECRET='<same-as-before>' \
  WORKER_PUBLIC_URL='https://<workspace>--pipeline-worker-fastapi-app.modal.run' \
  STALE_JOB_TIMEOUTS_JSON='<...>' \
  PIPELINE_USE_TX_RPC='<...>' \
  LLM_MERGED='<...>' \
  WHISPER_ASYNC='<...>'
```

Then redeploy so containers pick up the refreshed secret:

```bash
modal deploy pipeline/worker_service/app.py
```

---

## 4. Smoke-test the new endpoint

Health check:

```bash
curl -i https://<workspace>--pipeline-worker-fastapi-app.modal.run/health
```

Expect HTTP 200 and the same JSON the Railway worker returned.

Try a manual `/process-next-job` POST with the worker secret — should
either return `{"claimed": false}` (nothing queued) or claim a job:

```bash
curl -i -X POST \
  https://<workspace>--pipeline-worker-fastapi-app.modal.run/process-next-job \
  -H "X-Worker-Secret: <PIPELINE_WORKER_SECRET>"
```

If both succeed, the Modal deploy is healthy. Move to step 5.

---

## 5. Repoint pg_cron from Railway → Modal

Apply migration
`web/supabase/migrations/20260521_repoint_pipeline_worker_to_modal.sql`
in the Supabase SQL editor. **Before applying, replace the placeholder
URL** in the migration with the Modal URL from step 2:

```sql
alter database postgres set app.settings.pipeline_worker_url =
    'https://<workspace>--pipeline-worker-fastapi-app.modal.run/process-next-job';
```

The existing crons (`pipeline-worker-tick` every 30s and
`pipeline-worker-recover` every 5 min — registered in
`20260415_brand_ig_analysis.sql`) read this setting at tick time, so
this single `alter database` atomically repoints both. No need to
unschedule / reschedule.

Confirm the new URL is in play by checking the pg_net request log
~30 seconds after applying:

```sql
select status_code, content::text, created
from net._http_response
order by created desc
limit 3;
```

You should see HTTP 200 responses with `creatorgoose`-shaped JSON
bodies, originating from the Modal URL.

---

## 6. Update Apify webhook URLs

Each Apify actor we use has a webhook configured in its Apify dashboard
that POSTs to `/apify-webhook` when a run completes. After Modal is
live, update each one from the Railway URL to the Modal URL.

Actors to update:

| Actor (env var) | Default value | Where to update |
|---|---|---|
| `APIFY_ACTOR_INSTAGRAM` | `apify/instagram-scraper` | Apify console → actor settings → webhooks |
| `APIFY_ACTOR_IG_POST` | `apify/instagram-post-scraper` | same |
| `APIFY_IG_DM_SEND_ACTOR` | `am_production/instagram-direct-messages-dms-automation` | same |

For each, the webhook URL pattern is:

```
https://<workspace>--pipeline-worker-fastapi-app.modal.run/apify-webhook?secret=<APIFY_WEBHOOK_SECRET>
```

> Note: most webhooks are also built dynamically per-run by
> `pipeline/ig.py::_build_webhook_url`, which already reads
> `WORKER_PUBLIC_URL` (updated in step 3). The Apify-console-side
> webhook is only for the global completion webhook some actors fire
> for all runs.

---

## 7. Soak (24-48h)

Watch the system for 24-48 hours before decommissioning Railway.

```sql
-- Healthy queue draining
select status, count(*) from background_jobs
where created_at > now() - interval '1 hour'
group by 1;
```

Expected: `succeeded` grows, `queued` and `running` stay small (typically
0-3 sustained).

```sql
-- No timeout surge
select last_error, count(*) from background_jobs
where status = 'failed'
  and created_at > now() - interval '6 hours'
group by 1
order by 2 desc;
```

Modal logs should show `/process-next-job` POSTs every 30s and
`/apify-webhook` POSTs as Apify runs complete.

---

## 8. Decommission Railway

Once the 24h soak is clean:

1. Pause (don't delete) the Railway service for 24h. If nothing
   breaks, the project is safe to delete.
2. Remove Railway env vars from the project's `.env.example` (if any
   are Railway-specific).
3. Update the README / any deploy docs that reference Railway.
4. Delete the Railway project from the dashboard.

---

## Rollback

If anything breaks after migration 076 and you need to fall back to
Railway:

1. **Re-apply the previous pg_cron config.** Manually `cron.schedule()`
   the two jobs back to the Railway URL.
2. **Restore Apify webhook URLs** in the Apify console (the prior URLs
   were Railway-hosted).
3. **Restart the Railway service** if it was paused.

Modal stays deployed; switching back is a SQL + Apify-console operation.
