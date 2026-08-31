#!/usr/bin/env python3
"""Expand the common-sense exposure set to cover more domain dimensions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


POINT = Path(__file__).resolve().parents[1]
TAB = POINT / "tables"

STRICT = TAB / "point2_common_sense_directional_variable_selection.csv"
OUT_SELECTION = TAB / "point2_common_sense_expanded_variable_selection.csv"
OUT_KEPT = TAB / "point2_common_sense_expanded_kept_variables.csv"

DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]

EXPANDED_KEEP = {
    "Heat": {
        "daymet_dairy_weighted_consec_thi72_maxrun": "duration dimension: maximum consecutive THI>=72 run",
        "daymet_dairy_weighted_dry_hot_days_t72wb_lt22": "compound dry-hot days",
        "daymet_dairy_weighted_humid_hot_days_t72wb24": "compound humid-hot days",
        "daymet_dairy_weighted_thi_days_ge_68": "mild heat THI frequency threshold",
        "daymet_dairy_weighted_thi_days_ge_72": "core heat-stress THI frequency threshold",
        "daymet_dairy_weighted_thi_days_ge_79": "severe heat THI frequency threshold",
        "daymet_dairy_weighted_thi_heatload_ge72": "cumulative THI heatload",
        "daymet_dairy_weighted_tmax_days_ge_30c": "moderate dry daytime heat threshold",
        "daymet_dairy_weighted_tmax_days_ge_32c": "core dry daytime extreme heat threshold",
        "daymet_dairy_weighted_tmax_days_ge_35c": "severe dry daytime extreme heat threshold",
        "daymet_dairy_weighted_vpd_days_ge_3kpa": "moderate atmospheric dryness threshold",
        "daymet_dairy_weighted_vpd_days_ge_4kpa": "severe atmospheric dryness threshold",
        "daymet_dairy_weighted_vpd_heatload_ge2": "cumulative VPD heatload",
        "daymet_dairy_weighted_warm_nights_tmin_ge_20c": "core warm-night recovery stress threshold",
        "daymet_dairy_weighted_warm_nights_tmin_ge_22c": "severe warm-night threshold",
        "daymet_dairy_weighted_wetbulb_days_ge_22c": "moderate humid-heat wet-bulb threshold",
        "daymet_dairy_weighted_wetbulb_days_ge_26c": "severe humid-heat wet-bulb threshold",
        "daymet_dairy_weighted_wetbulb_heatload_ge22": "cumulative wet-bulb heatload",
    },
    "Cold": {
        "daymet_dairy_weighted_cold_load_lt50": "cumulative cold-load threshold",
        "daymet_dairy_weighted_cold_nights_tmin_le0": "freezing-night frequency",
        "daymet_dairy_weighted_consec_ice_days_maxrun": "duration dimension: maximum consecutive ice days",
        "daymet_dairy_weighted_consec_thi_lt45_maxrun": "duration dimension: maximum consecutive cold-THI run",
        "daymet_dairy_weighted_dry_cold_days_lt39": "severe dry-cold days",
        "daymet_dairy_weighted_dry_cold_days_lt45": "core dry-cold days",
        "daymet_dairy_weighted_dry_cold_days_lt50": "moderate dry-cold days",
        "daymet_dairy_weighted_dry_cold_load_lt45": "cumulative dry-cold load",
        "daymet_dairy_weighted_hard_freeze_le10": "severe hard-freeze threshold",
        "daymet_dairy_weighted_hard_freeze_nights": "hard-freeze nights",
        "daymet_dairy_weighted_ice_days": "daytime freezing exposure",
        "daymet_dairy_weighted_thi_days_lt_39": "severe cold-THI threshold",
        "daymet_dairy_weighted_thi_days_lt_45": "core cold-THI threshold",
        "daymet_dairy_weighted_thi_days_lt_50": "moderate cold-THI threshold",
        "daymet_dairy_weighted_wet_cold_days": "wet-cold day frequency",
        "daymet_dairy_weighted_wet_cold_days_lt39": "severe wet-cold threshold",
        "daymet_dairy_weighted_wet_cold_days_lt50": "moderate wet-cold threshold",
        "daymet_dairy_weighted_wet_cold_load_lt45": "cumulative wet-cold load",
    },
    "Severe weather": {
        "storm_any_disruptive_event": "any disruptive storm event frequency",
        "storm_event_types": "event diversity / breadth",
        "storm_fire_events": "fire-related storm disruption",
        "storm_flood_events": "flood disruption",
    },
    "Forage": {
        "nass_forage_condition_index_1to5": "forage condition level",
        "nass_forage_good_or_excellent_pct": "forage favorable-condition tail",
        "nass_forage_poor_minus_good_pct": "forage stress-vs-good contrast",
        "nass_forage_poor_or_very_poor_pct": "forage poor-condition stress tail",
        "nass_pastureland_condition_index_1to5": "pastureland condition level",
        "nass_pastureland_good_or_excellent_pct": "pastureland favorable-condition tail",
        "nass_pastureland_poor_minus_good_pct": "pastureland stress-vs-good contrast",
        "nass_pastureland_poor_or_very_poor_pct": "pastureland poor-condition stress tail",
    },
    "Feed market": {
        "feed_alfalfa_hay_price_ratio": "alfalfa relative price pressure",
        "feed_alfalfa_hay_price_spread_ton": "alfalfa price spread",
        "feed_corn_price_received_bu": "corn price level",
        "feed_corn_price_received_bu_index_2000_2004": "corn price index",
        "feed_hay_excl_alfalfa_price_received_ton_state_month_robust_z": "non-alfalfa hay price pressure; raw level unavailable in signal pool",
        "feed_hay_to_corn_price_ratio": "hay-vs-corn relative price",
        "feed_price_index_2000_2004": "composite feed price index",
    },
    "Dairy market": {
        "milk_price_all_classes_received_cwt": "all-classes milk price level",
        "milk_price_feed_ratio_proxy_index_2000_2004": "milk-feed margin proxy",
        "milk_price_fluid_grade_received_cwt": "fluid milk price level",
    },
    "Market demand": {
        "market_log_population_total": "absolute market population level",
        "market_population_change_1y": "absolute annual population change",
        "market_population_index_2000": "market size growth index",
        "market_population_share_us": "relative national market share",
        "market_population_yoy_growth_pct": "annual market growth rate",
    },
}

EXCLUDE_REASONS = {
    "daymet_dairy_weighted_diurnal_range_c": "excluded: diurnal range is not a direct heat-load intensity",
    "milk_price_fluid_grade_received_cwt_state_month_anomaly": "excluded: transformed anomaly duplicate with poor yearly availability",
}


def classify(row: pd.Series) -> tuple[str, str]:
    domain = row["domain_label"]
    exposure = row["exposure"]
    if exposure in EXPANDED_KEEP.get(domain, {}):
        status = "kept_expanded"
        if row["trend_match"] == "does_not_match" and domain in {"Heat", "Cold", "Severe weather", "Market demand"}:
            status = "kept_expanded_review_trend"
        if exposure.endswith("_state_month_robust_z") or exposure.endswith("_state_month_anomaly"):
            status = "kept_expanded_transform_unique_dimension"
        return status, EXPANDED_KEEP[domain][exposure]
    if exposure in EXCLUDE_REASONS:
        return "dropped_expanded", EXCLUDE_REASONS[exposure]
    if exposure.endswith("_state_month_anomaly") or exposure.endswith("_state_month_robust_z"):
        return "dropped_expanded", "excluded: transformed anomaly/robust-z duplicate of an already represented feed/market dimension"
    return "dropped_expanded", "excluded: no added definition dimension beyond expanded set"


def main() -> int:
    d = pd.read_csv(STRICT)
    classes = d.apply(classify, axis=1, result_type="expand")
    d["expanded_selection_status"] = classes[0]
    d["expanded_selection_reason"] = classes[1]
    d["domain_label"] = pd.Categorical(d["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    d = d.sort_values(["domain_label", "expanded_selection_status", "exposure"])
    kept = d[d["expanded_selection_status"].str.startswith("kept_expanded", na=False)].copy()
    d.to_csv(OUT_SELECTION, index=False)
    kept.to_csv(OUT_KEPT, index=False)
    print(f"Wrote {OUT_SELECTION}")
    print(f"Wrote {OUT_KEPT}")
    print("\nExpanded kept counts")
    print(kept.groupby(["domain_label", "expanded_selection_status"], observed=True)["exposure"].nunique().to_string())
    print("\nExpanded kept by domain")
    print(kept.groupby("domain_label", observed=True)["exposure"].nunique().to_string())
    print("\nDropped by domain")
    print(d[d["expanded_selection_status"].eq("dropped_expanded")].groupby("domain_label", observed=True)["exposure"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
