"""Tests for pipeline.contact_extract — bio email/phone scanning."""

from pipeline.contact_extract import (
    extract_email_from_text,
    extract_phone_from_text,
)


class TestExtractEmail:
    def test_none_input(self):
        assert extract_email_from_text(None) is None

    def test_empty(self):
        assert extract_email_from_text("") is None

    def test_no_email(self):
        assert extract_email_from_text("Just a regular bio") is None

    def test_simple(self):
        assert (
            extract_email_from_text("Reach me at hello@example.com")
            == "hello@example.com"
        )

    def test_business_prefix_preferred_over_personal(self):
        text = "personal: jane@gmail.com\nWork: business@brand.co"
        assert extract_email_from_text(text) == "business@brand.co"

    def test_falls_back_to_first_match(self):
        # No business-intent local-part — first match wins
        assert (
            extract_email_from_text("ping me jane@gmail.com or john@gmail.com")
            == "jane@gmail.com"
        )

    def test_mkbhd_style(self):
        text = (
            "MKBHD: Quality Tech Videos | YouTuber | Geek "
            "| Consumer Electronics\n\nbusiness@MKBHD.com\n\nNYC"
        )
        assert extract_email_from_text(text) == "business@MKBHD.com"

    def test_with_plus_addressing(self):
        assert (
            extract_email_from_text("dm collabs+yt@brand.io")
            == "collabs+yt@brand.io"
        )

    def test_press_prefix_recognized(self):
        text = "fan: x@y.com  press: press@brand.com"
        assert extract_email_from_text(text) == "press@brand.com"

    def test_collabs_prefix_recognized(self):
        text = "collabs@studio.io for partnership inquiries"
        assert extract_email_from_text(text) == "collabs@studio.io"

    def test_rejects_ip_form(self):
        # TLD must be alphabetic
        assert extract_email_from_text("oddly@127.0.0.1 fake") is None

    def test_strips_word_boundary_punctuation(self):
        # Email followed by sentence punctuation
        assert (
            extract_email_from_text("email me at hi@there.com.")
            == "hi@there.com"
        )

    def test_dotted_local_part(self):
        assert (
            extract_email_from_text("first.last@company.io")
            == "first.last@company.io"
        )


class TestExtractPhone:
    def test_none_input(self):
        assert extract_phone_from_text(None) is None

    def test_no_phone(self):
        assert extract_phone_from_text("Just a regular bio") is None

    def test_no_short_numbers(self):
        # Years/follower-counts shouldn't match
        assert extract_phone_from_text("Started in 2021") is None
        assert extract_phone_from_text("100k followers") is None

    def test_indian_mobile(self):
        assert extract_phone_from_text("WhatsApp 9876543210") == "9876543210"

    def test_international(self):
        out = extract_phone_from_text("Call +1 (555) 123-4567 anytime")
        assert out is not None
        assert "555" in out

    def test_us_dashed(self):
        assert extract_phone_from_text("Call 555-123-4567") == "555-123-4567"

    def test_intl_with_spaces(self):
        out = extract_phone_from_text("Reach: +91 98765 43210")
        assert out is not None
        assert "91" in out and "98765" in out
