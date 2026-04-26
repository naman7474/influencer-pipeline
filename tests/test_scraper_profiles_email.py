"""Tests for the bio-fallback email/phone extraction in IG profile scraper."""

from pipeline.scraper_profiles import extract_profile_metrics


class TestEmailFallback:
    def test_dedicated_field_wins_over_bio(self):
        """When IG returns contact_email explicitly, that's the source of truth."""
        out = extract_profile_metrics(
            {
                "account": "x",
                "biography": "DM business@bio.com for collabs",
                "contact_email": "explicit@official.com",
            }
        )
        assert out["email"] == "explicit@official.com"

    def test_falls_back_to_bio_when_field_empty(self):
        out = extract_profile_metrics(
            {
                "account": "x",
                "biography": "Tech reviews | hello@brand.co | LA",
                "contact_email": None,
            }
        )
        assert out["email"] == "hello@brand.co"

    def test_falls_back_to_bio_when_field_missing(self):
        out = extract_profile_metrics(
            {
                "account": "x",
                "biography": "press@studio.io for partnerships",
            }
        )
        assert out["email"] == "press@studio.io"

    def test_business_prefix_preferred_in_bio(self):
        out = extract_profile_metrics(
            {
                "account": "x",
                "biography": "personal jane@gmail.com\nbusiness business@brand.co",
                "contact_email": None,
            }
        )
        assert out["email"] == "business@brand.co"

    def test_no_email_anywhere(self):
        out = extract_profile_metrics(
            {
                "account": "x",
                "biography": "Just photos and travel",
            }
        )
        assert out["email"] is None


class TestPhoneFallback:
    def test_dedicated_field_wins(self):
        out = extract_profile_metrics(
            {
                "account": "x",
                "biography": "WhatsApp 9876543210",
                "contact_phone_number": "+91 1234567890",
            }
        )
        assert out["phone"] == "+91 1234567890"

    def test_falls_back_to_bio(self):
        out = extract_profile_metrics(
            {
                "account": "x",
                "biography": "WhatsApp 9876543210 for orders",
                "contact_phone_number": None,
            }
        )
        assert out["phone"] == "9876543210"
