"""Tests for YT benchmarks added to pipeline.confidence."""

from pipeline.confidence import (
    IG_ER_BENCHMARKS,
    YT_ER_BENCHMARKS,
    YT_VIEWS_PER_SUB_STRONG,
    YT_VIEWS_PER_SUB_WEAK,
    CPI_WEIGHTS,
)


class TestYTConstants:
    def test_yt_benchmarks_cover_all_tiers(self):
        for tier in ("nano", "micro", "mid", "macro", "mega"):
            assert tier in YT_ER_BENCHMARKS
            assert tier in IG_ER_BENCHMARKS

    def test_yt_benchmarks_higher_than_ig(self):
        # YT denominator = views (smaller); ratio therefore runs higher.
        for tier in ("nano", "micro", "mid", "macro", "mega"):
            assert YT_ER_BENCHMARKS[tier] > IG_ER_BENCHMARKS[tier]

    def test_yt_benchmarks_descending_by_tier(self):
        assert (
            YT_ER_BENCHMARKS["nano"]
            > YT_ER_BENCHMARKS["micro"]
            > YT_ER_BENCHMARKS["mid"]
            > YT_ER_BENCHMARKS["macro"]
            > YT_ER_BENCHMARKS["mega"]
        )

    def test_views_per_sub_thresholds(self):
        assert 0 < YT_VIEWS_PER_SUB_WEAK < YT_VIEWS_PER_SUB_STRONG

    def test_cpi_weights_sum_to_one(self):
        assert abs(sum(CPI_WEIGHTS.values()) - 1.0) < 1e-9
