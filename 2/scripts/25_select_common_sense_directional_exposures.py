#!/usr/bin/env python3
"""Select domain representatives by definition and directional data pattern.

This replaces the earlier correlation-first nonredundant selection. The goal is
to keep variables that are definitionally distinct and whose signed-z temporal
pattern matches domain expectations where those expectations are clear.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
TAB = POINT / "tables"
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402


DECOMP = TAB / "point2_total_insensitivity_log_decomposition.csv"
OUT_TRENDS = TAB / "point2_common_sense_directional_candidate_trends.csv"
OUT_SELECTION = TAB / "point2_common_sense_directional_variable_selection.csv"

DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]

# Curated, definition-first representatives. Correlation is not used as a hard
# filter here; highly correlated variables can still be dropped when they encode
# the same definition at a nearby threshold.
SELECTED = {
    "Heat": {
        "daymet_dairy_weighted_thi_days_ge_72": "core THI frequency; direct heat-stress threshold",
        "daymet_dairy_weighted_thi_heatload_ge72": "cumulative THI load; complements frequency",
        "daymet_dairy_weighted_wetbulb_days_ge_26c": "humid heat; physiologically distinct from THI count",
        "daymet_dairy_weighted_tmax_days_ge_32c": "dry daytime extreme heat; direct temperature threshold",
        "daymet_dairy_weighted_vpd_days_ge_4kpa": "dry heat/aridity stress; distinct from temperature and wet-bulb",
        "daymet_dairy_weighted_warm_nights_tmin_ge_20c": "nighttime recovery stress",
    },
    "Cold": {
        "daymet_dairy_weighted_thi_days_lt_45": "cold THI frequency; direct cold-stress threshold",
        "daymet_dairy_weighted_hard_freeze_nights": "hard freeze nights; distinct freeze mechanism",
        "daymet_dairy_weighted_ice_days": "ice days; daytime freezing exposure",
        "daymet_dairy_weighted_snow_cover_days": "snow-cover persistence; snowpack mechanism",
        "daymet_dairy_weighted_swe_max_mm": "snow-water burden magnitude; complements snow-cover days",
        "daymet_dairy_weighted_wet_cold_load_lt45": "wet cold load; distinct wet-cold stress",
    },
    "Severe weather": {
        "storm_flood_events": "flood disruption",
        "storm_fire_events": "fire disruption",
        "storm_event_types": "event diversity/severity breadth",
    },
    "Forage": {
        "nass_forage_condition_index_1to5": "forage condition level; primary forage quality index",
        "nass_forage_poor_or_very_poor_pct": "forage stress tail; poor-condition burden",
        "nass_pastureland_condition_index_1to5": "pastureland condition level; related but distinct land class",
        "nass_pastureland_poor_or_very_poor_pct": "pastureland stress tail",
    },
    "Feed market": {
        "feed_corn_price_received_bu": "corn price level",
        "feed_alfalfa_hay_price_ratio": "alfalfa relative price pressure",
        "feed_hay_to_corn_price_ratio": "hay-vs-corn relative feed price",
        "feed_price_index_2000_2004": "composite feed price index",
    },
    "Dairy market": {
        "milk_price_fluid_grade_received_cwt": "fluid milk price level",
        "milk_price_feed_ratio_proxy_index_2000_2004": "milk-feed margin proxy",
    },
    "Market demand": {
        "market_population_index_2000": "market size growth level",
        "market_population_yoy_growth_pct": "market growth rate",
        "market_population_share_us": "relative national market share",
    },
}

EXCLUDE_CONTAINS = {
    "Heat": {
        "diurnal_range_c": "not a direct heat-load intensity; direction is ambiguous for heat stress",
    },
    "Dairy market": {
        "state_month_anomaly": "transformed anomaly duplicate; poor yearly availability for trend figure",
    },
}


def candidate_pool() -> pd.DataFrame:
    d = pd.read_csv(DECOMP)
    d["domain_label"] = d["domain_plot"].replace({"Forage condition": "Forage"})
    return d[
        d["domain_label"].isin(DOMAIN_ORDER)
        & ((d["total_p"] < 0.05) | (d["per_cow_p"] < 0.05) | (d["cows_p"] < 0.05))
    ].copy()


def annual_signed_z(panel: pd.DataFrame, exposure: str) -> pd.Series:
    d = panel.loc[panel["year"].between(2000, 2024), ["year", exposure]].replace([np.inf, -np.inf], np.nan).dropna()
    sd = d[exposure].std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(dtype=float)
    d["z"] = (d[exposure] - d[exposure].mean()) / sd
    return d.groupby("year")["z"].mean()


def slope(s: pd.Series, start: int, end: int) -> float:
    x = s.loc[start:end].dropna()
    if len(x) < 5:
        return np.nan
    return float(np.polyfit(x.index.to_numpy(float), x.to_numpy(float), 1)[0])


def expected_direction(domain: str, exposure: str) -> str:
    if domain == "Heat":
        return "increase"
    if domain == "Cold":
        return "decrease"
    if domain == "Severe weather":
        return "increase"
    if domain == "Market demand":
        return "increase"
    if domain in {"Feed market", "Dairy market"}:
        return "increase_or_context_specific"
    if domain == "Forage":
        if "poor" in exposure:
            return "stress_increase_or_context_specific"
        return "condition_context_specific"
    return "context_specific"


def trend_match(domain: str, exposure: str, slope_2000_2024: float, endpoint_change: float) -> str:
    direction = expected_direction(domain, exposure)
    if not np.isfinite(slope_2000_2024) or not np.isfinite(endpoint_change):
        return "insufficient"
    if direction == "increase":
        return "matches" if slope_2000_2024 > 0 and endpoint_change > 0 else "does_not_match"
    if direction == "decrease":
        return "matches" if slope_2000_2024 < 0 and endpoint_change < 0 else "does_not_match"
    return "context_specific"


def exclusion_reason(row: pd.Series) -> str | None:
    domain = row["domain_label"]
    exposure = row["exposure"]
    for pat, reason in EXCLUDE_CONTAINS.get(domain, {}).items():
        if pat in exposure:
            return reason
    if exposure.endswith("_state_month_anomaly") or exposure.endswith("_state_month_robust_z"):
        return "transformed state-month anomaly/robust-z duplicate; prefer interpretable level/index variable"
    if "_index_2000_2004_state_month_anomaly" in exposure:
        return "transformed anomaly duplicate of index"
    return None


def classify(row: pd.Series) -> tuple[str, str]:
    domain = row["domain_label"]
    exposure = row["exposure"]
    selected = SELECTED.get(domain, {})
    reason = exclusion_reason(row)
    if exposure in selected:
        if row["trend_match"] == "does_not_match" and domain in {"Heat", "Cold", "Severe weather", "Market demand"}:
            return "review_trend_mismatch", selected[exposure]
        return "kept_common_sense", selected[exposure]
    if reason is not None:
        return "dropped_definition_or_transform", reason
    if row["trend_match"] == "does_not_match":
        return "dropped_trend_mismatch", f"does not match expected {expected_direction(domain, exposure)} pattern"
    return "dropped_definition_redundant", "definitionally redundant with selected representative in the same domain"


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = candidate_pool()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel().copy()

    trend_rows = []
    for _, row in candidates.iterrows():
        exposure = row["exposure"]
        if exposure not in panel.columns:
            continue
        ann = annual_signed_z(panel, exposure)
        if ann.empty:
            continue
        trend_rows.append(
            {
                "exposure": exposure,
                "z_2000": ann.get(2000, np.nan),
                "z_2014": ann.get(2014, np.nan),
                "z_2024": ann.get(2024, np.nan),
                "endpoint_change_2000_2024": ann.get(2024, np.nan) - ann.get(2000, np.nan),
                "slope_signed_z_2000_2024": slope(ann, 2000, 2024),
                "slope_signed_z_2000_2014": slope(ann, 2000, 2014),
                "slope_signed_z_2015_2024": slope(ann, 2015, 2024),
                "n_years": int(ann.dropna().shape[0]),
            }
        )
    trends = candidates.merge(pd.DataFrame(trend_rows), on="exposure", how="left")
    trends["expected_direction"] = [
        expected_direction(d, e) for d, e in zip(trends["domain_label"], trends["exposure"])
    ]
    trends["trend_match"] = [
        trend_match(d, e, s, c)
        for d, e, s, c in zip(
            trends["domain_label"],
            trends["exposure"],
            trends["slope_signed_z_2000_2024"],
            trends["endpoint_change_2000_2024"],
        )
    ]
    classes = trends.apply(classify, axis=1, result_type="expand")
    trends["selection_status"] = classes[0]
    trends["selection_reason"] = classes[1]
    trends["candidate_pool"] = "7_domain_nominal_signal"
    trends["candidate_pool_n"] = trends["exposure"].nunique()

    selection = trends[
        [
            "candidate_pool",
            "candidate_pool_n",
            "domain_label",
            "exposure",
            "selection_status",
            "selection_reason",
            "expected_direction",
            "trend_match",
            "z_2000",
            "z_2014",
            "z_2024",
            "slope_signed_z_2000_2024",
            "slope_signed_z_2000_2014",
            "slope_signed_z_2015_2024",
            "total_p",
            "per_cow_p",
            "cows_p",
        ]
    ].copy()
    selection["domain_label"] = pd.Categorical(selection["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    selection = selection.sort_values(["domain_label", "selection_status", "exposure"])
    return trends, selection


def main() -> int:
    trends, selection = build()
    trends.to_csv(OUT_TRENDS, index=False)
    selection.to_csv(OUT_SELECTION, index=False)
    print(f"Wrote {OUT_TRENDS}")
    print(f"Wrote {OUT_SELECTION}")
    print("\nCandidate pool")
    print(selection.groupby("domain_label", observed=True)["exposure"].nunique().to_string())
    print("\nKept")
    kept = selection[selection["selection_status"].eq("kept_common_sense")]
    print(kept.groupby("domain_label", observed=True)["exposure"].nunique().to_string())
    print(kept[["domain_label", "exposure", "expected_direction", "trend_match", "selection_reason"]].to_string(index=False))
    print("\nReview")
    review = selection[selection["selection_status"].str.contains("review", na=False)]
    if len(review):
        print(review[["domain_label", "exposure", "expected_direction", "trend_match", "selection_reason"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
