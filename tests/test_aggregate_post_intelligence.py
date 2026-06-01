"""Tests for pipeline/aggregate_post_intelligence.py — distributions + medians."""

from pipeline.aggregate_post_intelligence import aggregate_posts, build_post_rows


def _p(item_id, **kw):
    base = {
        "item_id": item_id,
        "post_intent": None,
        "content_pillar": None,
        "hook_style": None,
        "hook_quality": None,
        "emotional_trigger": None,
        "cta_type": None,
        "content_orientation": None,
        "comment_classification": {},
        "demographics_signal": {},
    }
    base.update(kw)
    return base


class TestAggregatePosts:
    def test_distributions_and_medians(self):
        payloads = [
            _p("A", post_intent="educate", content_pillar="AI",
               hook_style="question", hook_quality=0.8, content_orientation="informational",
               comment_classification={"discussion_pct": 0.5, "sentiment_score": 0.6}),
            _p("B", post_intent="personal", content_pillar="journey",
               hook_style="story", hook_quality=0.6, content_orientation="personal",
               comment_classification={"discussion_pct": 0.3, "sentiment_score": 0.2}),
            _p("C", post_intent="educate", content_pillar="AI",
               hook_style="music", hook_quality=0.2, content_orientation="informational",
               comment_classification={}),
        ]
        eng = {"A": {"engagement_rate": 0.05}, "B": {"engagement_rate": 0.03},
               "C": {"engagement_rate": 0.20}}
        agg = aggregate_posts(payloads, eng)

        assert agg["posts_analyzed"] == 3
        # intent: educate x2, personal x1
        intent = {b["label"]: b["count"] for b in agg["intent_distribution"]}
        assert intent == {"educate": 2, "personal": 1}
        # music excluded from hook pie
        hooks = {b["label"] for b in agg["hook_style_distribution"]}
        assert "music" not in hooks and hooks == {"question", "story"}
        # medians
        assert agg["median_hook_quality"] == 0.6   # median(0.8,0.6,0.2)
        assert agg["median_engagement_rate"] == 0.05
        assert agg["median_discussion_pct"] == 0.4  # median(0.5,0.3)

    def test_defaulted_excluded(self):
        payloads = [
            _p("A", post_intent="sell"),
            {"item_id": "B", "_defaulted": True, "post_intent": None},
        ]
        agg = aggregate_posts(payloads)
        assert agg["posts_analyzed"] == 1
        assert len(agg["intent_distribution"]) == 1

    def test_empty(self):
        agg = aggregate_posts([])
        assert agg["posts_analyzed"] == 0
        assert agg["intent_distribution"] == []
        assert agg["median_hook_quality"] is None

    def test_pct_sums_to_100ish(self):
        payloads = [_p(str(i), post_intent="educate") for i in range(3)]
        payloads.append(_p("x", post_intent="sell"))
        agg = aggregate_posts(payloads)
        total = sum(b["pct"] for b in agg["intent_distribution"])
        assert abs(total - 100.0) < 0.5


class TestBuildPostRows:
    def test_maps_fields_and_meta(self):
        payloads = [_p("A", post_intent="educate", hook_quality=0.8,
                       comment_classification={"emoji_only_pct": 0.2},
                       demographics_signal={"estimated_age_group": "25-34"})]
        meta = {"A": {"content_type": "reel", "views": 1000,
                      "engagement_rate": 0.05, "has_transcript": True,
                      "comment_sample_size": 7}}
        rows = build_post_rows("cid", "instagram", payloads, meta)
        r = rows[0]
        assert r["creator_id"] == "cid" and r["platform"] == "instagram"
        assert r["item_id"] == "A" and r["post_intent"] == "educate"
        assert r["hook_quality"] == 0.8
        assert r["comment_class_emoji_pct"] == 0.2
        assert r["demo_age_signal"] == "25-34"
        assert r["views"] == 1000 and r["has_transcript"] is True
        assert r["comment_sample_size"] == 7
        assert r["data_quality"]["was_defaulted"] is False

    def test_defaulted_flag_and_skip_missing_id(self):
        payloads = [
            {"item_id": "", "post_intent": None},        # skipped (no id)
            {"item_id": "B", "_defaulted": True},        # kept, flagged
        ]
        rows = build_post_rows("cid", "instagram", payloads)
        assert len(rows) == 1
        assert rows[0]["item_id"] == "B"
        assert rows[0]["data_quality"]["was_defaulted"] is True
