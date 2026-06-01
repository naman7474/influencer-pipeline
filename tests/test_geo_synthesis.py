"""Tests for pipeline/geo_synthesis.py — synthetic brand_shopify_geo rows."""

from pipeline.geo_synthesis import (
    derive_synthetic_geo_rows,
    resolve_state,
)


class TestResolveState:
    def test_direct_state_match(self):
        assert resolve_state("Maharashtra") == "maharashtra"
        assert resolve_state("delhi") == "delhi"

    def test_city_backtracks_to_state(self):
        assert resolve_state("Mumbai") == "maharashtra"
        assert resolve_state("Bangalore") == "karnataka"
        assert resolve_state("Hyderabad") == "telangana"

    def test_substring_match(self):
        assert resolve_state("Greater Mumbai Region") == "maharashtra"

    def test_unknown_returns_none(self):
        assert resolve_state("Atlantis") is None
        assert resolve_state("") is None
        assert resolve_state(None) is None


class TestDeriveSyntheticGeoRows:
    def test_all_india_seeds_pan_india_states(self):
        rows = derive_synthetic_geo_rows(
            brand_id="brand-1",
            shipping_zones=["All India"],
            target_regions=None,
            ig_audience_profile=None,
        )
        assert len(rows) >= 10
        assert all(r["source"] == "synthetic" for r in rows)
        assert all(r["problem_type"] == "awareness_gap" for r in rows)
        states = {r["state"] for r in rows}
        # Sanity: pan-India seed should cover the major metros' states.
        assert "maharashtra" in states
        assert "karnataka" in states
        assert "delhi" in states

    def test_specific_cities_resolve_to_states(self):
        rows = derive_synthetic_geo_rows(
            brand_id="brand-1",
            shipping_zones=["Delhi", "Mumbai"],
            target_regions=None,
            ig_audience_profile=None,
        )
        states = {r["state"] for r in rows}
        assert states == {"delhi", "maharashtra"}

    def test_empty_shipping_with_india_audience_seeds_pan_india(self):
        # Physics Wallah scenario: no shipping_zones (digital product),
        # IG audience is Indian → fall back to pan-India seed.
        rows = derive_synthetic_geo_rows(
            brand_id="brand-1",
            shipping_zones=None,
            target_regions=None,
            ig_audience_profile={"primary_country": "IN"},
        )
        assert len(rows) > 0
        assert all(r["source"] == "synthetic" for r in rows)

    def test_no_signals_emits_nothing(self):
        # Truly empty input + foreign audience → no synthetic rows. Better
        # to leave audience_geo at the 0.3 floor than fabricate.
        rows = derive_synthetic_geo_rows(
            brand_id="brand-1",
            shipping_zones=None,
            target_regions=None,
            ig_audience_profile={"primary_country": "US"},
        )
        assert rows == []

    def test_target_regions_resolve(self):
        rows = derive_synthetic_geo_rows(
            brand_id="brand-1",
            shipping_zones=None,
            target_regions=["Karnataka", "Tamil Nadu"],
            ig_audience_profile=None,
        )
        states = {r["state"] for r in rows}
        assert states == {"karnataka", "tamil nadu"}

    def test_unknown_zone_silently_dropped(self):
        # Unrecognized strings shouldn't crash or emit garbage rows.
        rows = derive_synthetic_geo_rows(
            brand_id="brand-1",
            shipping_zones=["Atlantis", "Mumbai"],
            target_regions=None,
            ig_audience_profile=None,
        )
        states = {r["state"] for r in rows}
        assert states == {"maharashtra"}

    def test_row_shape_matches_brand_shopify_geo_columns(self):
        rows = derive_synthetic_geo_rows(
            brand_id="brand-1",
            shipping_zones=["Mumbai"],
            target_regions=None,
            ig_audience_profile=None,
        )
        row = rows[0]
        # Required columns for the upsert path in handlers._upsert_synthetic_geo_rows.
        for col in (
            "brand_id",
            "state",
            "city",
            "country",
            "sessions",
            "orders",
            "revenue",
            "population_weight",
            "gap_score",
            "problem_type",
            "source",
        ):
            assert col in row
        assert row["country"] == "IN"
        assert row["sessions"] == 0
        assert row["orders"] == 0
