"""Tests for pipeline.youtube.scraper_videos."""

from unittest.mock import MagicMock

from pipeline.youtube.scraper_videos import (
    DATASET_YT_VIDEOS,
    aggregate_channel_metrics,
    extract_video_metrics,
    scrape_videos_discovery,
    select_top_videos,
    _extract_video_id_from_url,
    _is_short,
    _to_int,
    _to_num,
)


class TestHelpers:
    def test_to_int_none(self):
        assert _to_int(None) == 0

    def test_to_int_int(self):
        assert _to_int(10) == 10

    def test_to_int_int_str(self):
        assert _to_int("42") == 42

    def test_to_int_float_str(self):
        assert _to_int("3.14") == 3

    def test_to_int_bad(self):
        assert _to_int("abc", default=-1) == -1

    def test_to_num_none(self):
        assert _to_num(None) == 0.0

    def test_to_num_float(self):
        assert _to_num(3.5) == 3.5

    def test_to_num_str(self):
        assert _to_num("1.5") == 1.5

    def test_to_num_bad(self):
        assert _to_num("xyz", default=9.9) == 9.9

    def test_extract_video_id_watch_url(self):
        assert (
            _extract_video_id_from_url("https://www.youtube.com/watch?v=abc123&t=30")
            == "abc123"
        )

    def test_extract_video_id_short_url(self):
        assert (
            _extract_video_id_from_url("https://youtu.be/abc123?si=xxx")
            == "abc123"
        )

    def test_extract_video_id_shorts(self):
        assert (
            _extract_video_id_from_url("https://www.youtube.com/shorts/abc123")
            == "abc123"
        )

    def test_extract_video_id_none(self):
        assert _extract_video_id_from_url("") is None
        assert _extract_video_id_from_url("https://example.com") is None


class TestIsShort:
    def test_shorts_url(self):
        assert _is_short({"url": "https://www.youtube.com/shorts/abc"}) is True

    def test_duration_60s_or_less(self):
        assert _is_short({"url": "https://youtube.com/watch?v=abc", "length": 30}) is True
        assert _is_short({"url": "https://youtube.com/watch?v=abc", "length": 60}) is True

    def test_long_form(self):
        assert _is_short({"url": "https://youtube.com/watch?v=abc", "length": 300}) is False

    def test_zero_length_not_short(self):
        # Zero length is indeterminate, should NOT classify as short
        assert _is_short({"url": "https://youtube.com/watch?v=abc", "length": 0}) is False

    def test_duration_seconds_alias(self):
        assert _is_short({"duration_seconds": 45, "url": ""}) is True


class TestScrapeVideosDiscovery:
    def test_invokes_scrape_and_wait(self):
        client = MagicMock()
        client.scrape_and_wait.return_value = [{"video_id": "a"}]
        result = scrape_videos_discovery(
            client, "https://youtube.com/@x", num_videos=10
        )
        assert result == [{"video_id": "a"}]
        args, _ = client.scrape_and_wait.call_args
        assert args[0] == DATASET_YT_VIDEOS
        assert args[1][0]["url"] == "https://youtube.com/@x"
        assert args[1][0]["num_of_posts"] == 10


class TestSelectTopVideos:
    def test_orders_by_engagement(self):
        vids = [
            {"url": "a", "views": 100, "likes": 10, "num_comments": 5},
            {"url": "b", "views": 50, "likes": 50, "num_comments": 50},
            {"url": "c", "views": 1000, "likes": 1, "num_comments": 0},
        ]
        top = select_top_videos(vids, top_n=2)
        assert len(top) == 2
        # "b" scores highest: 50 + 500 + 1000 = 1550
        assert top[0]["url"] == "b"

    def test_drops_videos_without_url(self):
        vids = [{"likes": 100}, {"url": "a", "likes": 10}]
        top = select_top_videos(vids, top_n=5)
        assert len(top) == 1

    def test_include_shorts_toggle(self):
        vids = [
            {"url": "https://yt.com/shorts/x", "length": 30, "views": 100},
            {"url": "https://yt.com/watch?v=y", "length": 600, "views": 50},
        ]
        long_only = select_top_videos(vids, top_n=5, include_shorts=False)
        assert len(long_only) == 1
        assert long_only[0]["url"] == "https://yt.com/watch?v=y"


class TestExtractVideoMetrics:
    def test_full_record(self):
        raw = {
            "video_id": "abc",
            "url": "https://www.youtube.com/watch?v=abc",
            "title": "Review",
            "description": "Check this out",
            "tags": ["tech", "review"],
            "category_id": "22",
            "length": 600,
            "views": 100_000,
            "likes": 5_000,
            "num_comments": 300,
            "thumbnail": "https://img",
            "has_captions": True,
            "date_posted": "2026-01-01T00:00:00Z",
            "top_comments": [{"author": "u", "text": "nice"}],
        }
        out = extract_video_metrics(raw)
        assert out["video_id"] == "abc"
        assert out["title"] == "Review"
        assert out["tags"] == ["tech", "review"]
        assert out["category_id"] == 22
        assert out["is_short"] is False
        assert out["duration_seconds"] == 600
        assert out["view_count"] == 100_000
        assert out["like_count"] == 5_000
        assert out["comment_count"] == 300
        assert out["has_captions"] is True
        assert out["caption_source"] == "youtube_auto"
        assert out["top_comments"] == [{"author": "u", "text": "nice"}]

    def test_video_id_from_url_when_missing(self):
        out = extract_video_metrics({"url": "https://youtu.be/zzz"})
        assert out["video_id"] == "zzz"

    def test_short_detection_via_url(self):
        out = extract_video_metrics(
            {"url": "https://www.youtube.com/shorts/abc", "length": 45}
        )
        assert out["is_short"] is True

    def test_missing_fields(self):
        out = extract_video_metrics({})
        assert out["video_id"] is None
        assert out["view_count"] == 0
        assert out["has_captions"] is False
        assert out["caption_source"] is None

    def test_transcript_inline_implies_has_captions(self):
        out = extract_video_metrics(
            {"url": "https://yt.com/watch?v=a", "transcript": "Hello world"}
        )
        assert out["has_captions"] is True
        assert out["transcript_inline"] == "Hello world"

    def test_livestream_flag(self):
        out = extract_video_metrics(
            {"url": "https://yt.com/watch?v=a", "is_live": True}
        )
        assert out["is_livestream"] is True

    def test_zero_category_id_is_none(self):
        out = extract_video_metrics(
            {"url": "https://yt.com/watch?v=a", "category_id": "0"}
        )
        # `_to_int("0")` returns 0; the `or None` in extract makes it None
        assert out["category_id"] is None


class TestAggregateChannelMetrics:
    def test_empty_videos(self):
        assert aggregate_channel_metrics([]) == {}

    def test_aggregate(self):
        videos = [
            {
                "video_id": "a",
                "view_count": 100,
                "like_count": 10,
                "duration_seconds": 600,
                "is_short": False,
                "published_at": "2026-01-01T00:00:00Z",
                "top_comments": [{"text": "nice"}],
            },
            {
                "video_id": "b",
                "view_count": 200,
                "like_count": 50,
                "duration_seconds": 30,
                "is_short": True,
                "published_at": "2026-01-08T00:00:00Z",
                "top_comments": [],
            },
            {
                "video_id": "c",
                "view_count": 0,
                "like_count": 0,
                "duration_seconds": 0,
                "is_livestream": True,
                "top_comments": [],
            },
        ]
        out = aggregate_channel_metrics(videos)
        assert out["avg_view_count"] == 150.0  # (100+200)/2 (c dropped bc views=0)
        # likes/view for a=0.1, b=0.25 -> avg 0.175
        assert abs(out["watch_through_proxy"] - 0.175) < 0.01
        assert out["content_mix"]["youtube_long"] == 1
        assert out["content_mix"]["youtube_short"] == 1
        assert out["content_mix"]["youtube_live"] == 1
        assert out["upload_cadence_days"] == 7.0

    def test_cadence_needs_two_timestamps(self):
        out = aggregate_channel_metrics(
            [{"view_count": 1, "like_count": 1, "published_at": "2026-01-01T00:00:00Z"}]
        )
        assert out["upload_cadence_days"] is None

    def test_top_comments_passed_through(self):
        videos = [
            {
                "video_id": "a",
                "view_count": 100,
                "like_count": 10,
                "top_comments": [
                    {"author": "u1", "text": "t1", "published_at": "2026", "like_count": 3}
                ],
            }
        ]
        out = aggregate_channel_metrics(videos)
        assert len(out["top_comments_from_videos"]) == 1
        assert out["top_comments_from_videos"][0]["user"] == "u1"
        assert out["top_comments_from_videos"][0]["source_video_id"] == "a"

    def test_unparseable_timestamp_ignored(self):
        videos = [
            {"video_id": "a", "view_count": 100, "like_count": 1, "published_at": "not-a-date"},
            {"video_id": "b", "view_count": 100, "like_count": 1, "published_at": "not-a-date"},
        ]
        out = aggregate_channel_metrics(videos)
        assert out["upload_cadence_days"] is None
