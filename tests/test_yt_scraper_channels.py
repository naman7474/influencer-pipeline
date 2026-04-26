"""Tests for pipeline.youtube.scraper_channels."""

from unittest.mock import MagicMock

from pipeline.youtube.scraper_channels import (
    DATASET_YT_CHANNELS,
    LOW_SUBSCRIBER_CUTOFF,
    classify_creator_tier,
    extract_channel_metrics,
    scrape_channels,
    _to_int,
)


class TestToInt:
    def test_none(self):
        assert _to_int(None) == 0

    def test_int(self):
        assert _to_int(42) == 42

    def test_numeric_string(self):
        assert _to_int("100") == 100

    def test_non_numeric_string(self):
        assert _to_int("abc") == 0

    def test_default(self):
        assert _to_int(None, default=-1) == -1


class TestClassifyCreatorTier:
    def test_nano(self):
        assert classify_creator_tier(5_000) == "nano"
        assert classify_creator_tier(0) == "nano"

    def test_micro(self):
        assert classify_creator_tier(10_000) == "micro"
        assert classify_creator_tier(49_999) == "micro"

    def test_mid(self):
        assert classify_creator_tier(50_000) == "mid"
        assert classify_creator_tier(499_999) == "mid"

    def test_macro(self):
        assert classify_creator_tier(500_000) == "macro"
        assert classify_creator_tier(999_999) == "macro"

    def test_mega(self):
        assert classify_creator_tier(1_000_000) == "mega"
        assert classify_creator_tier(10_000_000) == "mega"


class TestScrapeChannels:
    def test_calls_trigger_and_wait_with_urls(self):
        client = MagicMock()
        client.trigger_and_wait.return_value = [{"id": "UC1"}]
        out = scrape_channels(client, ["https://www.youtube.com/@mkbhd"])
        assert out == [{"id": "UC1"}]
        client.trigger_and_wait.assert_called_once_with(
            DATASET_YT_CHANNELS, [{"url": "https://www.youtube.com/@mkbhd"}]
        )


class TestExtractChannelMetrics:
    def test_typical_record(self):
        raw = {
            "handle": "@mkbhd",
            "channel_id": "UCBJycsmduvYEL83R_U4JriQ",
            "url": "https://www.youtube.com/@mkbhd",
            "name": "Marques Brownlee",
            "profile_image": "https://yt3.ggpht.com/...",
            "description": "Quality tech videos.",
            "country": "US",
            "category": "Tech",
            "verified": True,
            "subscribers": 20_000_000,
            "videos_count": 1600,
            "views": 4_000_000_000,
            "links": [{"label": "IG", "url": "https://instagram.com/mkbhd"}],
        }
        out = extract_channel_metrics(raw)
        assert out["handle"] == "mkbhd"  # @ stripped
        assert out["platform_user_id"] == "UCBJycsmduvYEL83R_U4JriQ"
        assert out["display_name"] == "Marques Brownlee"
        assert out["is_verified"] is True
        assert out["followers_or_subs"] == 20_000_000
        assert out["posts_or_videos_count"] == 1600
        assert out["total_views"] == 4_000_000_000
        assert out["tier"] == "mega"
        assert out["country"] == "US"
        assert out["category"] == "Tech"
        assert out["external_links"] == [
            {"label": "IG", "url": "https://instagram.com/mkbhd"}
        ]
        assert "low_subscribers" not in out["data_quality_flags"]

    def test_missing_fields_return_none(self):
        out = extract_channel_metrics({})
        assert out["handle"] is None
        assert out["platform_user_id"] is None
        assert out["followers_or_subs"] == 0
        assert out["tier"] == "nano"
        assert out["is_verified"] is False
        assert "low_subscribers" in out["data_quality_flags"]

    def test_low_subscribers_flag(self):
        out = extract_channel_metrics({"subscribers": LOW_SUBSCRIBER_CUTOFF - 1})
        assert "low_subscribers" in out["data_quality_flags"]

    def test_external_links_as_dict_normalized(self):
        out = extract_channel_metrics(
            {
                "subscribers": 200,
                "links": {"instagram": "https://instagram.com/x", "empty": None},
            }
        )
        # None values are filtered out
        assert out["external_links"] == [
            {"label": "instagram", "url": "https://instagram.com/x"}
        ]

    def test_brightdata_capitalized_links_list(self):
        """Bright Data's actual response uses capitalized `Links` and a flat
        list of URL strings (no scheme). Match that real shape."""
        out = extract_channel_metrics(
            {
                "subscribers": 1_000_000,
                "Links": [
                    "twitter.com/MKBHD",
                    "instagram.com/MKBHD",
                    "youtube.com/c/TheStudio",
                    "discord.gg/MKBHD",
                ],
            }
        )
        # Each URL becomes {label, url} for stitching.extract_handles_from_links
        urls = [link["url"] for link in out["external_links"]]
        assert "instagram.com/MKBHD" in urls
        assert "twitter.com/MKBHD" in urls

    def test_brightdata_capitalized_description_and_details(self):
        """Description / Details / created_date are capitalized in BD output."""
        out = extract_channel_metrics(
            {
                "subscribers": 1_000_000,
                "Description": "Quality Tech Videos | business@MKBHD.com",
                "Details": {"location": "United States"},
                "created_date": "2008-03-21T00:00:00.000Z",
            }
        )
        assert "business@MKBHD.com" in (out["bio"] or "")
        assert out["country"] == "United States"
        assert out["channel_created_at"] == "2008-03-21T00:00:00.000Z"

    def test_email_extracted_from_bio(self):
        """YT has no dedicated email field — must be extracted from bio."""
        out = extract_channel_metrics(
            {
                "subscribers": 1_000_000,
                "Description": "Tech reviews | business@brand.co | NYC",
            }
        )
        assert out["email"] == "business@brand.co"

    def test_phone_extracted_from_bio(self):
        out = extract_channel_metrics(
            {
                "subscribers": 1_000_000,
                "Description": "DM 9876543210 for collabs",
            }
        )
        assert out["phone"] == "9876543210"

    def test_email_none_when_no_bio(self):
        out = extract_channel_metrics({"subscribers": 1_000_000})
        assert out["email"] is None
        assert out["phone"] is None

    def test_alias_fields(self):
        out = extract_channel_metrics(
            {
                "subscriber_count": 100_000,
                "video_count": 50,
                "view_count": 5_000_000,
                "custom_url": "@creator",
                "id": "UC_alias",
                "title": "Creator",
                "thumbnail": "https://img",
                "about": "bio text",
                "topic": "Gaming",
                "country_code": "IN",
            }
        )
        assert out["followers_or_subs"] == 100_000
        assert out["posts_or_videos_count"] == 50
        assert out["total_views"] == 5_000_000
        assert out["handle"] == "creator"
        assert out["platform_user_id"] == "UC_alias"
        assert out["display_name"] == "Creator"
        assert out["bio"] == "bio text"
        assert out["category"] == "Gaming"
        assert out["country"] == "IN"
        assert out["tier"] == "mid"

    def test_channel_id_only_builds_url(self):
        out = extract_channel_metrics({"channel_id": "UCabc"})
        assert out["profile_url"] == "https://www.youtube.com/channel/UCabc"

    def test_handle_with_whitespace_stripped(self):
        out = extract_channel_metrics({"handle": "  @x  "})
        assert out["handle"] == "x"
