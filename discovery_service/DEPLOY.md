# Discovery service — deployment

The discovery service runs on Modal. It's invoked by the Next.js app at
`/api/discover/requests` POST when a user clicks the "Search YouTube"
CTA on the Discover page (alignment = weak or empty).

## 1. Apply migrations

```bash
# In order — 073 first (schema), 074 second (RPCs).
psql $SUPABASE_DB_URL -f supabase/migrations/073_discovery_requests.sql
psql $SUPABASE_DB_URL -f supabase/migrations/074_refresh_creator_leaderboard_rpc.sql
```

Then regenerate Supabase types so the new tables show up in
`web/src/lib/types/database.ts`:

```bash
supabase gen types typescript --linked > web/src/lib/types/database.ts
```

(The route handlers use untyped casts in the meantime, so this is
non-blocking — but typed reads are nicer.)

## 2. Create Modal secrets

```bash
modal secret create discovery-supabase \
  SUPABASE_URL=$SUPABASE_URL \
  SUPABASE_SERVICE_ROLE_KEY=$SUPABASE_SERVICE_ROLE_KEY

modal secret create discovery-openai OPENAI_API_KEY=$OPENAI_API_KEY

modal secret create discovery-gemini GEMINI_API_KEY=$GEMINI_API_KEY

# 12 YT keys, comma-separated. Pool round-robins + fails-over on quota.
modal secret create discovery-youtube YOUTUBE_API_KEYS="key1,key2,...,key12"

# Apify — required for the Instagram discovery path (Phase 5).
# APIFY_TOKEN drives the SDK; APIFY_ACTOR_IG_SEARCH_USERS is optional
# and defaults to `patient_discovery/instagram-search-users` in code.
# APIFY_ACTOR_INSTAGRAM names the bundle/deep-scrape actor the existing
# pipeline already uses; reuse the same value the pipeline-worker has.
modal secret create discovery-apify \
  APIFY_TOKEN='<from-pipeline-apify-secret>' \
  APIFY_ACTOR_INSTAGRAM='apify/instagram-scraper' \
  APIFY_ACTOR_IG_SEARCH_USERS='patient_discovery/instagram-search-users'

# Shared HMAC between Modal and the Next.js app. Generate one secret
# and put it in BOTH places. The Modal enqueue endpoint validates it
# on inbound requests; the brand-match compute endpoint validates the
# same secret on its inbound (Modal → Next.js) callbacks.
SECRET=$(openssl rand -hex 32)
modal secret create discovery-callback \
  DISCOVERY_WEB_BASE_URL=https://<your-web-host> \
  DISCOVERY_SERVICE_SECRET=$SECRET
```

## 3. Deploy the Modal app

```bash
cd /Users/namanjain/Documents/influencer\ copy
modal deploy pipeline/discovery_service/app.py
```

The deploy output prints the URL of the `enqueue` web endpoint:

```
✓ Created web endpoint for enqueue => https://<workspace>--discovery-enqueue.modal.run
```

Copy that URL — you'll set it on the Next.js side next.

## 4. Set Next.js env vars

In your Vercel / hosting platform (or `.env.local` for dev):

```bash
# The Modal enqueue URL from step 3.
MODAL_DISCOVERY_ENQUEUE_URL=https://<workspace>--discovery-enqueue.modal.run

# Same secret as in `discovery-callback` Modal secret.
DISCOVERY_SERVICE_SECRET=<same-as-step-2>

# Per-brand caps + cooldown (defaults in code if unset).
DISCOVERY_DAILY_CAP_PER_BRAND=5
DISCOVERY_COOLDOWN_SECONDS=300

# Semantic dedup thresholds.
DISCOVERY_SIM_BLOCK=0.90
DISCOVERY_SIM_WARN=0.80

# Flip the UI from "Coming soon" to "Search YouTube" — set after Modal
# is deployed and you've verified the round-trip works end-to-end.
NEXT_PUBLIC_DISCOVERY_PIPELINE_READY=1
```

## 5. End-to-end smoke test

1. Sign in to the web app, go to Discover.
2. Search for something that gets `alignment="empty"` (e.g. "blockchain ASMR").
3. The empty-state card appears. Click "Search YouTube for creators".
4. Loader modal opens. `discovery_requests` row exists in DB with
   `status='queued'`, `modal_call_id` populated within ~1s.
5. Watch status progress: searching → profiling → scraping → … → completed.
6. Modal closes; result list shows discovered creators ranked by match score.
7. Hard-refresh the tab: nothing pending; `localStorage.discovery_request_id` empty.
8. Re-trigger and close the tab during `scraping`; reopen Discover — the
   loader re-attaches via `DiscoveryResume` + `/api/discover/requests/active`.

## 6. Per-discovery cost (rough)

| Component | Cost |
|-----------|------|
| YouTube Data API | ~1600 quota units (~1.3% of daily 120K with 12 keys) |
| Modal Whisper | ~$0.50–$2.00 (varies with cold starts + transcript miss rate) |
| Gemini caption+audience LLM | ~$0.20–$0.60 across 200 creators |
| OpenAI text-embedding-3-small | <$0.01 |
| Modal container time | ~$0.10–$0.30 (3–5 min × 4 CPU × 4 GB) |
| **Total** | **~$1–$3 per discovery** |

The per-brand 5/day cap + 5-min cooldown + semantic dedup at 0.90/0.80
should keep org-wide spend predictable. Watch `discovery_requests`
status distribution + Modal billing for the first 2 weeks.

## 7. Troubleshooting

- **`call_id` never populates on discovery_requests** — Modal enqueue
  request failed. Check Next.js server logs for the modal enqueue error
  message; usually a wrong URL or stale secret.
- **Loader stays in `queued` forever** — the Modal worker never picked
  up the spawn. Run `modal app logs discovery` and look for the spawn
  call. If absent, the `enqueue` web endpoint isn't reachable.
- **Loader gets to `matching` then fails with `brand_match_client unconfigured`**
  — `DISCOVERY_WEB_BASE_URL` or `DISCOVERY_SERVICE_SECRET` missing on
  the Modal side. Update the `discovery-callback` secret and redeploy.
- **Discoveries take 8+ minutes consistently** — likely Modal Whisper
  cold-start storm. Set `DISCOVERY_NUM_TRANSCRIPTS=0` to skip transcription
  on the discovery path (the regular per-creator scrape adds transcripts
  later via the normal queue).

## 8. Bulk discovery (Phase 6)

Same pipeline, batched from the command line. Useful for seeding a new
brand's creator pool with many niche queries at once, or for scheduled
background refreshes.

```bash
cd "/Users/namanjain/Documents/influencer copy"
python3 scripts/run_bulk_discovery.py \
  --queries-csv scripts/sample_bulk_queries.csv \
  --brand-id <BRAND_UUID> \
  [--delay 5] [--dry-run]
```

CSV header is `query_text,platform`. Each row spawns one Modal worker
through the same `enqueue` endpoint the Discover-page CTA uses, so the
behaviour and cost profile match exactly.

Dedup: skips queries where (brand_id, lower(query_text)) was already
discovered + completed in the last 7 days. Re-runs of the same CSV are
safe.

Env vars required (same as the web side):
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `MODAL_DISCOVERY_ENQUEUE_URL`,
`DISCOVERY_SERVICE_SECRET`. Picked up automatically from `.env` at the
repo root if present.

Dry-run mode (`--dry-run`) validates the CSV + reports what would
happen without writing to DB or calling Modal — useful when reviewing
a long query list before committing budget.
