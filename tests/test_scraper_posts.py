"""Tests for pipeline/scraper_posts.py — scrape_single_post utility."""

from unittest.mock import MagicMock
import pytest

from pipeline.scraper_posts import scrape_single_post


class TestScrapeSinglePost:
    def test_returns_post_data_with_video_url(self):
        client = MagicMock()
        client.scrape_and_wait.return_value = [
            {
                "post_id": "abc123",
                "video_url": "https://cdn.instagram.com/video.mp4",
                "description": "Great reel!",
                "likes": 500,
                "views": 10000,
                "length": 30,
            }
        ]

        result = scrape_single_post(
            client, "https://www.instagram.com/reel/abc123/"
        )

        assert result is not None
        assert result["post_id"] == "abc123"
        assert result["video_url"] == "https://cdn.instagram.com/video.mp4"
        assert result["length"] == 30
        client.scrape_and_wait.assert_called_once()

    def test_returns_none_when_no_results(self):
        client = MagicMock()
        client.scrape_and_wait.return_value = []

        result = scrape_single_post(
            client, "https://www.instagram.com/reel/missing/"
        )

        assert result is None

    def test_returns_post_without_video_url(self):
        """Static posts may not have video_url — function should still return data."""
        client = MagicMock()
        client.scrape_and_wait.return_value = [
            {
                "post_id": "static_post",
                "video_url": None,
                "description": "Photo post",
                "likes": 200,
            }
        ]

        result = scrape_single_post(
            client, "https://www.instagram.com/p/static_post/"
        )

        assert result is not None
        assert result["video_url"] is None

    def test_passes_correct_url_to_client(self):
        client = MagicMock()
        client.scrape_and_wait.return_value = [{"post_id": "x"}]

        url = "https://www.instagram.com/reel/CxYz123/"
        scrape_single_post(client, url)

        call_args = client.scrape_and_wait.call_args
        payload = call_args[0][1]  # second positional arg
        assert payload == [{"url": url}]

    def test_uses_reels_dataset_id(self):
        client = MagicMock()
        client.scrape_and_wait.return_value = [{"post_id": "x"}]

        scrape_single_post(client, "https://www.instagram.com/reel/test/")

        call_args = client.scrape_and_wait.call_args
        dataset_id = call_args[0][0]  # first positional arg
        assert dataset_id == "gd_lyclm20il4r5helnj"
