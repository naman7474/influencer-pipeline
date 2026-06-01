"""On-demand Instagram creator discovery (Phase 5).

Mirrors the YouTube discovery service at `pipeline.discovery_service`
but speaks Apify instead of the YouTube Data API. Calls
`patient_discovery/instagram-search-users` to turn a keyword query into
a candidate username list, then drives each survivor through the
existing IG creator pipeline (`pipeline.apify_instagram_bundle.fetch`)
with `comments_per_reel=0` — discovery never scrapes comments by
design (cost reduction, user-confirmed in Phase 5 planning).
"""
