"""Tests for pipeline.youtube.handle_resolver."""

from unittest.mock import MagicMock

from pipeline.youtube.handle_resolver import resolve, ResolvedChannel


class TestResolve:
    def test_empty_input(self):
        r = resolve("")
        assert r.channel_id is None
        assert r.handle is None
        assert r.url == ""

    def test_bare_handle_with_at(self):
        api = MagicMock()
        api.available = True
        api.resolve_handle_to_channel_id.return_value = "UCBJycsmduvYEL83R_U4JriQ"
        r = resolve("@mkbhd", api=api)
        assert r.channel_id == "UCBJycsmduvYEL83R_U4JriQ"
        assert r.handle == "mkbhd"
        assert "UCBJycsmduvYEL83R_U4JriQ" in r.url
        api.resolve_handle_to_channel_id.assert_called_once_with("mkbhd")

    def test_bare_handle_without_at(self):
        api = MagicMock()
        api.available = True
        api.resolve_handle_to_channel_id.return_value = "UCabc"
        r = resolve("mkbhd", api=api)
        assert r.channel_id == "UCabc"
        assert r.handle == "mkbhd"

    def test_bare_handle_api_unavailable_keeps_handle(self):
        api = MagicMock()
        api.available = False
        r = resolve("@mkbhd", api=api)
        assert r.channel_id is None
        assert r.handle == "mkbhd"
        assert r.url == "https://www.youtube.com/@mkbhd"

    def test_channel_url_with_ucid(self):
        r = resolve("https://www.youtube.com/channel/UCBJycsmduvYEL83R_U4JriQ")
        assert r.channel_id == "UCBJycsmduvYEL83R_U4JriQ"
        assert r.handle is None
        assert r.url == "https://www.youtube.com/channel/UCBJycsmduvYEL83R_U4JriQ"

    def test_handle_url(self):
        api = MagicMock()
        api.available = True
        api.resolve_handle_to_channel_id.return_value = "UCmkbhd"
        r = resolve("https://www.youtube.com/@mkbhd", api=api)
        assert r.channel_id == "UCmkbhd"
        assert r.handle == "mkbhd"

    def test_handle_url_without_api_resolution(self):
        api = MagicMock()
        api.available = False
        r = resolve("https://www.youtube.com/@mkbhd", api=api)
        assert r.channel_id is None
        assert r.handle == "mkbhd"
        assert r.url == "https://www.youtube.com/@mkbhd"

    def test_custom_url(self):
        r = resolve("https://www.youtube.com/c/MarquesBrownlee")
        assert r.channel_id is None
        assert r.handle == "marquesbrownlee"
        assert r.url == "https://www.youtube.com/c/MarquesBrownlee"

    def test_legacy_user_url(self):
        r = resolve("https://www.youtube.com/user/marquesbrownlee")
        assert r.channel_id is None
        assert r.handle == "marquesbrownlee"

    def test_whitespace_is_stripped(self):
        r = resolve("  @mkbhd  ", api=MagicMock(available=False))
        assert r.handle == "mkbhd"

    def test_unknown_url_returns_raw(self):
        r = resolve("https://example.com/profile")
        assert r.channel_id is None
        assert r.handle is None
        assert r.url == "https://example.com/profile"

    def test_resolved_channel_dataclass(self):
        rc = ResolvedChannel(channel_id="UC1", handle="a", url="https://yt.com")
        assert rc.channel_id == "UC1"
        assert rc.handle == "a"

    def test_handle_case_normalization(self):
        api = MagicMock()
        api.available = False
        r = resolve("@MKBHD", api=api)
        assert r.handle == "mkbhd"

    def test_api_returns_none_keeps_handle(self):
        api = MagicMock()
        api.available = True
        api.resolve_handle_to_channel_id.return_value = None
        r = resolve("@nonexistent", api=api)
        assert r.channel_id is None
        assert r.handle == "nonexistent"
        assert r.url == "https://www.youtube.com/@nonexistent"

    def test_no_api_argument_creates_default(self, monkeypatch):
        # resolve() should work without an api argument — it creates a
        # YouTubeAPIClient internally. We just ensure the default path
        # is exercised without raising.
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        r = resolve("@mkbhd")
        # Without a key, the API is unavailable and we fall back to the
        # handle-only URL form.
        assert r.handle == "mkbhd"
