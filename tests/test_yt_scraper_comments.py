"""Tests for pipeline.youtube.scraper_comments."""

from datetime import datetime
from unittest.mock import MagicMock

from pipeline.youtube.scraper_comments import (
    DATASET_YT_COMMENTS,
    extract_comment_metrics,
    scrape_comments,
    select_top_videos_for_comments,
    _cluster_comment_hours,
)


class TestScrapeComments:
    def test_trigger_and_wait_called(self):
        client = MagicMock()
        client.trigger_and_wait.return_value = [{"text": "hi"}]
        out = scrape_comments(client, ["https://yt.com/watch?v=a"])
        assert out == [{"text": "hi"}]
        client.trigger_and_wait.assert_called_once_with(
            DATASET_YT_COMMENTS, [{"url": "https://yt.com/watch?v=a"}]
        )


class TestSelectTopVideosForComments:
    def test_orders_by_comment_count(self):
        vids = [
            {"url": "a", "comment_count": 5},
            {"url": "b", "comment_count": 20},
            {"url": "c", "comment_count": 0},
        ]
        top = select_top_videos_for_comments(vids, top_n=2)
        assert top == ["b", "a"]

    def test_skips_missing_url(self):
        vids = [{"comment_count": 100}]
        assert select_top_videos_for_comments(vids) == []

    def test_accepts_num_comments_alias(self):
        vids = [{"url": "a", "num_comments": 10}]
        assert select_top_videos_for_comments(vids) == ["a"]


class TestExtractCommentMetrics:
    def test_empty_returns_empty_dict(self):
        assert extract_comment_metrics([], None, "") == {}

    def test_basic_metrics(self):
        comments = [
            {
                "author_channel_id": "UC_user1",
                "author": "@user1",
                "text": "great video",
                "date": "2026-01-01T12:00:00Z",
                "replies": [],
            },
            {
                "author_channel_id": "UC_user2",
                "author": "user2",
                "text": "nice",
                "date": "2026-01-02T15:00:00Z",
                "replies": [],
            },
        ]
        out = extract_comment_metrics(comments, "UC_creator", "creator")
        assert out["creator_reply_count"] == 0
        assert out["unique_commenter_count"] == 2
        assert out["_comment_texts"] == ["great video", "nice"]
        assert "12" in out["comment_hour_distribution_utc"]

    def test_creator_reply_by_channel_id(self):
        comments = [
            {
                "author_channel_id": "UC_user1",
                "text": "q?",
                "replies": [
                    {"author_channel_id": "UC_creator", "author": "creator"},
                    {"author_channel_id": "UC_user2", "author": "user2"},
                ],
            }
        ]
        out = extract_comment_metrics(comments, "UC_creator", "creator")
        assert out["creator_reply_count"] == 1

    def test_creator_reply_by_handle_fallback(self):
        # No channel_id on reply; falls back to handle match
        comments = [
            {
                "author_channel_id": "UC_u",
                "text": "q?",
                "replies": [{"author": "@creator"}],
            }
        ]
        out = extract_comment_metrics(comments, None, "creator")
        assert out["creator_reply_count"] == 1

    def test_malformed_date_ignored(self):
        comments = [
            {"author_channel_id": "UC_u", "text": "x", "date": "bad-date"}
        ]
        out = extract_comment_metrics(comments, None, "")
        assert out["_comment_timestamps"] == []

    def test_no_channel_id_uses_handle_as_identity(self):
        comments = [
            {"author": "@u1", "text": "a"},
            {"author": "@u1", "text": "b"},
        ]
        out = extract_comment_metrics(comments, None, "")
        # Same author -> one unique commenter
        assert out["unique_commenter_count"] == 1


class TestClusterCommentHours:
    def test_empty_returns_empty(self):
        assert _cluster_comment_hours([]) == {}

    def test_even_distribution(self):
        ts = [datetime(2026, 1, 1, h, 0) for h in (10, 10, 14, 18)]
        out = _cluster_comment_hours(ts)
        assert out["10"] == 0.5
        assert out["14"] == 0.25
        assert out["18"] == 0.25
