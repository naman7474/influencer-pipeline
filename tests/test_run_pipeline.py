"""Tests for run_pipeline.py platform detection."""

import pytest

from run_pipeline import detect_platform


class TestDetectPlatform:
    def test_instagram(self):
        assert detect_platform("https://www.instagram.com/mkbhd/") == "instagram"

    def test_instagram_reel(self):
        assert (
            detect_platform("https://www.instagram.com/reel/abc/") == "instagram"
        )

    def test_youtube_watch(self):
        assert (
            detect_platform("https://www.youtube.com/watch?v=abc") == "youtube"
        )

    def test_youtube_handle(self):
        assert detect_platform("https://www.youtube.com/@mkbhd") == "youtube"

    def test_youtube_channel(self):
        assert (
            detect_platform("https://www.youtube.com/channel/UCabc") == "youtube"
        )

    def test_youtu_be_short(self):
        assert detect_platform("https://youtu.be/abc") == "youtube"

    def test_unknown_host_raises(self):
        with pytest.raises(ValueError, match="Cannot detect platform"):
            detect_platform("https://tiktok.com/@x")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            detect_platform("")

    def test_case_insensitive(self):
        assert detect_platform("HTTPS://INSTAGRAM.COM/x") == "instagram"
        assert detect_platform("HTTPS://YOUTUBE.COM/@x") == "youtube"
