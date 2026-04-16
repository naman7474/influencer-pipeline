# Pipeline Worker — Deployment

The Brand IG Analysis pipeline runs as a stateless Python container that
Supabase pg_cron pokes every 30 seconds. It can't run on Vercel (no Python
runtime, 60s timeout). Pick any container host that supports Python +
outbound HTTP + a public URL.

## 1. Pick a host

| Host | Why | Rough cost |
|------|-----|------------|
| **Railway** | One-click Dockerfile deploys, env-var UI, gives a public URL. Easiest. | Free tier covers dev, ~$5/mo for always-on |
| Render | Similar to Railway. Zero-config Dockerfile. | Free tier exists (spins down after inactivity — fine here, cron will wake it) |
| Fly.io | Cheapest at scale, regional control, needs `flyctl`. | ~$2–3/mo for a shared-cpu-1x |

Recommendation for v1: **Railway**. Switch later if cost/latency matters.

## 2. Build + run locally (sanity check)

```bash
cd "influencer copy"
cp pipeline/.env.example pipeline/.env   # fill values
docker build -f pipeline/Dockerfile -t influencer-pipeline .
docker run --rm -p 8000:8000 --env-file pipeline/.env influencer-pipeline

# From another terminal:
curl http://localhost:8000/health                  # → {"status":"ok"}
curl -X POST http://localhost:8000/process-next-job \
     -H "X-Worker-Secret: <your-secret>"           # → 204 if nothing queued
```

## 3. Deploy to Railway

1. `railway login && railway init` in the repo root.
2. Point Railway at `pipeline/Dockerfile` (Settings → Build → Dockerfile path).
3. Add env vars from `pipeline/.env.example` (Variables tab).
4. Deploy. Railway gives you an HTTPS URL like
   `https://influencer-pipeline-production.up.railway.app`.
5. Test: `curl https://<your-url>/health`.

## 4. Wire Supabase cron to the worker

Set two GUCs (Database → Settings → Database → Custom Postgres Config):

```
app.settings.pipeline_worker_url    = https://<your-url>/process-next-job
app.settings.pipeline_worker_secret = <same as PIPELINE_WORKER_SECRET>
```

Then enable `pg_cron` in Database → Extensions. The migration
`20260415_brand_ig_analysis.sql` already schedules the jobs conditionally —
re-run its `DO $$ ... $$` block if you enabled pg_cron after applying the
migration:

```sql
-- From the SQL editor:
SELECT cron.schedule(
  'pipeline-worker-tick',
  '30 seconds',
  $$ SELECT net.http_post(
       url := current_setting('app.settings.pipeline_worker_url', true),
       headers := jsonb_build_object(
         'Content-Type', 'application/json',
         'X-Worker-Secret', current_setting('app.settings.pipeline_worker_secret', true)
       ),
       body := '{}'::jsonb,
       timeout_milliseconds := 1000
     ); $$
);
```

Verify with `SELECT * FROM cron.job;` and `SELECT * FROM cron.job_run_details ORDER BY start_time DESC LIMIT 5;`.

## 5. Wire the Next.js web app to the worker

Add to your Vercel project env vars (or `.env.local` for local dev):

```
PIPELINE_WORKER_URL=https://<your-railway-url>
PIPELINE_WORKER_SECRET=<same secret>
```

Redeploy the web app.

## 6. End-to-end smoke test

1. Complete brand onboarding as a test user; provide an Instagram handle.
2. After Step 3 submit you should land on `/processing/<brandId>` with the
   full-screen loader.
3. Watch it cycle: `queued → scraping_brand → extracting_collaborators →
   scoring_creators → ranking → complete`, then route to `/dashboard`.
4. Inspect in Supabase:
   - `brands` row: `ig_analysis_status = 'completed'`, `ig_content_dna`,
     `ig_collaborators`, `content_embedding` populated.
   - `background_jobs`: one `brand_ig_scrape` + up to 10 `creator_ig_scrape`
     rows, all `succeeded`.
   - `creator_brand_matches`: rows with `used_ig_signals = true` and
     `match_score_breakdown` JSONB for the brand.

## Rough costs per onboarding

- 1 brand CIP run: ~$0.13
- Up to 10 creator CIP runs (capped, skips creators scraped < 30 days ago): ~$1.30
- OpenAI embeddings: ~$0.0001 per run (text-embedding-3-small is cheap)
- **Worst case: ~$1.43 per onboarding.** Typical (fewer tagged collaborators): $0.30–$0.80.

## Monitoring

- Railway logs are live; use them for tailing during rollout.
- Supabase: `SELECT status, count(*) FROM background_jobs GROUP BY status;` —
  alert if `queued` depth > 50 or `failed` count is climbing.
- The `/recover-stale-jobs` endpoint (cron every 5m) re-queues anything
  stuck > 15 min in `running`. Manual kick:
  ```
  curl -X POST https://<your-url>/recover-stale-jobs \
       -H "X-Worker-Secret: <secret>"
  ```
