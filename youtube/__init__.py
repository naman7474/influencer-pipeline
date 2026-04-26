"""YouTube ingestion subpackage.

Mirrors the layout of the Instagram scrapers at the top level of `pipeline/`:
    scraper_channels.py  -> scraper_profiles.py
    scraper_videos.py    -> scraper_posts.py / scraper_reels.py
    scraper_comments.py  -> scraper_comments.py

Plus two non-IG helpers:
    youtube_api.py       -> YouTube Data API v3 client (canonical stats refresh,
                            topic categories, handle -> channelId resolution)
    handle_resolver.py   -> URL / @handle -> UCxxxxxxxx channelId
"""
