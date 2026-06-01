"""Tests for pipeline/llm_post.py — JSON coercion, item assembly, degradation."""

import os

from pipeline import llm_post


class TestCoercePostsList:
    def test_dict_with_posts_key(self):
        assert llm_post._coerce_posts_list({"posts": [{"item_id": "A"}]}) == [
            {"item_id": "A"}
        ]

    def test_bare_list(self):
        out = llm_post._coerce_posts_list([{"item_id": "A"}, {"item_id": "B"}])
        assert len(out) == 2

    def test_single_object(self):
        out = llm_post._coerce_posts_list({"item_id": "A", "post_intent": "sell"})
        assert out == [{"item_id": "A", "post_intent": "sell"}]

    def test_alt_keys(self):
        assert llm_post._coerce_posts_list({"results": [{"item_id": "A"}]})
        assert llm_post._coerce_posts_list({"data": [{"item_id": "A"}]})

    def test_none_and_garbage(self):
        assert llm_post._coerce_posts_list(None) is None
        assert llm_post._coerce_posts_list(42) is None
        assert llm_post._coerce_posts_list({"nope": 1}) is None


class TestRenderBatch:
    def test_includes_every_item_id(self):
        items = [
            {"item_id": "A", "caption": "x", "transcript": None, "comments": []},
            {"item_id": "B", "caption": "y", "transcript": None, "comments": []},
        ]
        block = llm_post._render_batch("acme", items)
        assert "BEGIN_POST A" in block and "BEGIN_POST B" in block
        assert "@acme" in block

    def test_music_transcript_flagged(self):
        items = [{
            "item_id": "A", "caption": "c",
            "transcript": {"transcript_text": "la la", "is_likely_music": True},
            "comments": [],
        }]
        assert "[MUSIC-ONLY]" in llm_post._render_batch("acme", items)


class TestBuildItems:
    def test_ig_mapping_and_engagement(self):
        posts = [{
            "post_id": "p1", "url": "https://insta.com/p/p1/",
            "description": "my founder journey", "content_type": "Reel",
            "likes": 90, "num_comments": 10, "video_view_count": 1000,
        }]
        transcripts = [{"post_id": "p1", "transcript_text": "today", "hook_text": "hi"}]
        comments_by_post = {
            "https://insta.com/p/p1": [{"user": "u", "text": "great"}]
        }
        items, meta = llm_post.build_items(
            "instagram", posts, transcripts, comments_by_post
        )
        assert items[0]["item_id"] == "p1"
        assert items[0]["caption"] == "my founder journey"
        assert items[0]["transcript"]["transcript_text"] == "today"
        assert len(items[0]["comments"]) == 1
        # (90 + 10) / 1000 = 0.1
        assert meta["p1"]["engagement_rate"] == 0.1
        assert meta["p1"]["has_transcript"] is True
        assert meta["p1"]["comment_sample_size"] == 1

    def test_yt_mapping(self):
        videos = [{
            "video_id": "v1", "url": "https://youtu.be/v1", "title": "5 AI tools",
            "description": "list", "is_short": True, "view_count": 500,
            "like_count": 40, "comment_count": 10,
        }]
        items, meta = llm_post.build_items("youtube", videos)
        assert items[0]["item_id"] == "v1"
        assert "5 AI tools" in items[0]["caption"]
        assert meta["v1"]["content_type"] == "short"
        assert meta["v1"]["engagement_rate"] == 0.1

    def test_zero_views_engagement_none(self):
        posts = [{"post_id": "p1", "likes": 5, "num_comments": 1, "video_view_count": 0}]
        _, meta = llm_post.build_items("instagram", posts)
        assert meta["p1"]["engagement_rate"] is None

    def test_skips_items_without_id(self):
        items, _ = llm_post.build_items("instagram", [{"description": "no id"}])
        assert items == []


class TestDegradation:
    def test_no_api_key_degrades_all(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        items = [
            {"item_id": "A", "caption": "x", "comments": []},
            {"item_id": "B", "caption": "y", "comments": []},
        ]
        out = llm_post.analyze_posts_batch("acme", items, batch_size=4)
        assert len(out) == 2
        assert all(r["_defaulted"] for r in out)
        assert {r["item_id"] for r in out} == {"A", "B"}
        # defaulted payloads carry no classification
        assert out[0]["post_intent"] is None

    def test_empty_items(self):
        assert llm_post.analyze_posts_batch("acme", []) == []

    def test_default_payload_shape(self):
        d = llm_post._default_payload("Z", reason="test")
        assert d["item_id"] == "Z"
        assert d["_defaulted"] is True
        assert d["_default_reason"] == "test"
