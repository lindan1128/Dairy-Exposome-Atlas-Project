#!/usr/bin/env python3
"""Scatter 2015-2024 exposure intensity trends against |beta| trends."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
TAB = POINT / "tables"

PANEL = TAB / "point2_expanded_sensitivity_panel_for_r.csv"
BASE_BETA = TAB / "point2_herd_adjusted_yearly_sensitivity.csv"
NOMINAL_SCREEN = TAB / "point2_2015_2024_percow_clean_curated_7class_screen.csv"
NOMINAL_BETA = TAB / "point2_2015_2024_percow_nominal_alpha02_yearly_sensitivity.csv"
OUT_TABLE = TAB / "point2_2015_2024_exposure_intensity_vs_abs_beta_change.csv"

YEARS = list(range(2015, 2025))
DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
LABELS = {
    "daymet_dairy_weighted_consec_thi72_maxrun": "THI≥72 run",
    "daymet_dairy_weighted_diurnal_range_c": "Diurnal range",
    "daymet_dairy_weighted_dry_hot_days_t72wb_lt22": "Dry hot",
    "daymet_dairy_weighted_hot_no_relief_days_t72n20": "Hot no relief",
    "daymet_dairy_weighted_humid_hot_days_t72wb24": "Humid hot",
    "daymet_dairy_weighted_thi_days_ge_68": "THI≥68",
    "daymet_dairy_weighted_thi_days_ge_72": "THI≥72",
    "daymet_dairy_weighted_thi_days_ge_79": "THI≥79",
    "daymet_dairy_weighted_thi_heatload_ge72": "THI load≥72",
    "daymet_dairy_weighted_tmax_c": "Tmax mean",
    "daymet_dairy_weighted_tmax_days_ge_30c": "Tmax≥30C",
    "daymet_dairy_weighted_tmax_days_ge_32c": "Tmax≥32C",
    "daymet_dairy_weighted_tmax_days_ge_35c": "Tmax≥35C",
    "daymet_dairy_weighted_vpd_days_ge_3kpa": "VPD≥3",
    "daymet_dairy_weighted_vpd_days_ge_4kpa": "VPD≥4",
    "daymet_dairy_weighted_vpd_heatload_ge2": "VPD load≥2",
    "daymet_dairy_weighted_warm_nights_tmin_ge_18c": "Tmin≥18C",
    "daymet_dairy_weighted_warm_nights_tmin_ge_20c": "Tmin≥20C",
    "daymet_dairy_weighted_warm_nights_tmin_ge_22c": "Tmin≥22C",
    "daymet_dairy_weighted_wetbulb_days_ge_22c": "Wet-bulb≥22C",
    "daymet_dairy_weighted_wetbulb_days_ge_26c": "Wet-bulb≥26C",
    "daymet_dairy_weighted_wetbulb_heatload_ge22": "WB load≥22C",
    "daymet_dairy_weighted_cold_load_lt50": "Cold load<50",
    "daymet_dairy_weighted_cold_nights_tmin_le0": "Tmin≤0C",
    "daymet_dairy_weighted_consec_ice_days_maxrun": "Consecutive ice days",
    "daymet_dairy_weighted_consec_snowcover_maxrun": "Snow-cover run",
    "daymet_dairy_weighted_consec_thi_lt45_maxrun": "THI<45 run",
    "daymet_dairy_weighted_dry_cold_days_lt39": "Dry cold<39",
    "daymet_dairy_weighted_dry_cold_days_lt45": "Dry cold<45",
    "daymet_dairy_weighted_dry_cold_days_lt50": "Dry cold<50",
    "daymet_dairy_weighted_dry_cold_load_lt45": "Dry-cold load",
    "daymet_dairy_weighted_hard_freeze_le10": "Freeze≤10",
    "daymet_dairy_weighted_hard_freeze_nights": "Hard freeze",
    "daymet_dairy_weighted_ice_days": "Ice days",
    "daymet_dairy_weighted_thi_days_lt_39": "THI<39",
    "daymet_dairy_weighted_thi_days_lt_45": "THI<45",
    "daymet_dairy_weighted_thi_days_lt_50": "THI<50",
    "daymet_dairy_weighted_snow_cold_days_lt45": "Snow cold<45",
    "daymet_dairy_weighted_snow_cover_days": "Snow-cover days",
    "daymet_dairy_weighted_swe_days_ge_25mm": "SWE≥25mm",
    "daymet_dairy_weighted_swe_max_mm": "SWE max",
    "daymet_dairy_weighted_swe_mm": "SWE mean",
    "daymet_dairy_weighted_wet_cold_days": "Wet cold",
    "daymet_dairy_weighted_wet_cold_days_lt39": "Wet cold<39",
    "daymet_dairy_weighted_wet_cold_days_lt50": "Wet cold<50",
    "daymet_dairy_weighted_wet_cold_load_lt45": "Wet-cold load",
    "storm_event_types": "Storm events",
    "storm_flood_events": "Flood events",
    "storm_any_disruptive_event": "Any storm",
    "storm_fire_events": "Fire events",
    "nass_forage_condition_index_1to5": "Forage index",
    "nass_forage_good_or_excellent_pct": "Forage good",
    "nass_forage_poor_minus_good_pct": "Forage poor-good",
    "nass_forage_poor_or_very_poor_pct": "Forage poor",
    "nass_pastureland_condition_index_1to5": "Pasture index",
    "nass_pastureland_good_or_excellent_pct": "Pasture good",
    "nass_pastureland_poor_minus_good_pct": "Pasture poor-good",
    "nass_pastureland_poor_or_very_poor_pct": "Pasture poor",
    "feed_alfalfa_hay_price_ratio": "Alfalfa/corn",
    "feed_alfalfa_hay_price_ratio_state_month_anomaly": "Alfalfa/corn anomaly",
    "feed_alfalfa_hay_price_ratio_state_month_robust_z": "Alfalfa/corn z",
    "feed_alfalfa_hay_price_spread_ton": "Alfalfa spread",
    "feed_alfalfa_hay_price_spread_ton_state_month_anomaly": "Alfalfa spread anomaly",
    "feed_corn_price_received_bu": "Corn price",
    "feed_corn_price_received_bu_index_2000_2004": "Corn index",
    "feed_corn_price_received_bu_index_2000_2004_state_month_anomaly": "Corn index anomaly",
    "feed_corn_price_received_bu_state_month_anomaly": "Corn price anomaly",
    "feed_hay_to_corn_price_ratio": "Hay/corn",
    "feed_hay_to_corn_price_ratio_state_month_anomaly": "Hay/corn anomaly",
    "feed_hay_to_corn_price_ratio_state_month_robust_z": "Hay/corn z",
    "feed_price_index_2000_2004": "Feed index",
    "feed_price_index_2000_2004_state_month_anomaly": "Feed index anomaly",
    "feed_hay_excl_alfalfa_price_received_ton_index_2000_2004_state_month_robust_z": "Hay index z",
    "feed_hay_excl_alfalfa_price_received_ton_state_month_robust_z": "Hay price z",
    "milk_price_all_classes_received_cwt": "Milk price",
    "milk_price_feed_ratio_proxy_index_2000_2004": "Milk-feed",
    "milk_price_fluid_grade_received_cwt": "Fluid price",
    "market_log_population_total": "Population",
    "market_population_change_1y": "Pop change",
    "market_population_index_2000": "Pop index",
    "market_population_share_us": "US pop share",
    "market_population_yoy_growth_pct": "Pop growth",
}

def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    sd = x.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean(skipna=True)) / sd


def slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.nanstd(y[ok]) <= 1e-12:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def short_label(exposure: str, reason: str) -> str:
    if exposure in LABELS:
        return LABELS[exposure]
    if isinstance(reason, str) and reason:
        label = reason
    else:
        label = exposure
    replacements = {
        "duration dimension: ": "",
        " threshold": "",
        " frequency": "",
        "cumulative ": "",
        "core ": "",
        "moderate ": "",
        "severe ": "",
        "daymet dairy weighted ": "",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = label.replace("maximum consecutive", "max consecutive")
    label = label.replace("atmospheric dryness", "VPD")
    label = label.replace("condition", "cond.")
    label = label.replace("population", "pop.")
    label = label.replace("price", "px")
    label = label.replace("market", "mkt")
    label = label.replace("_state_month_anomaly", "")
    label = label.replace("_dairy_weighted", "")
    label = label.replace("_rolling_3mo_sum", " 3mo sum")
    label = label.replace("_rolling_3mo_mean", " 3mo mean")
    label = label.replace("_consecutive", " consec.")
    label = label.replace("_threshold", "")
    label = label.replace("_", " ")
    label = label.replace(">=", "≥").replace("<=", "≤")
    return label[:38]

def build_table() -> pd.DataFrame:
    beta_path = NOMINAL_BETA if NOMINAL_BETA.exists() else BASE_BETA
    beta = pd.read_csv(beta_path)
    beta = beta[
        beta["outcome"].eq("per_cow")
        & beta["status"].eq("ok")
        & beta["year"].isin(YEARS)
        & beta["domain_label"].isin(DOMAIN_ORDER)
    ].copy()
    beta["abs_beta"] = beta["beta_log_per_1sd_exposure"].abs()
    if NOMINAL_SCREEN.exists() and beta_path == NOMINAL_BETA:
        screen = pd.read_csv(
            NOMINAL_SCREEN,
            usecols=[
                "exposure",
                "bonferroni_sig_2015_2024",
                "by_fdr_sig_2015_2024",
                "nominal_sig_2015_2024",
            ],
        )
        union_sig = (
            screen["bonferroni_sig_2015_2024"].fillna(False).astype(bool)
            | screen["by_fdr_sig_2015_2024"].fillna(False).astype(bool)
            | screen["nominal_sig_2015_2024"].fillna(False).astype(bool)
        )
        nominal = set(screen[union_sig]["exposure"].drop_duplicates())
        beta = beta[beta["exposure"].isin(nominal)].copy()
    panel_header = pd.read_csv(PANEL, nrows=0)
    panel_exposures = set(panel_header.columns) - {"year", "state_alpha", "month"}
    selected = (
        beta.loc[beta["exposure"].isin(panel_exposures), ["domain_label", "exposure"]]
        .drop_duplicates()
        .sort_values(["domain_label", "exposure"])
        .reset_index(drop=True)
    )

    panel_cols = ["year", "state_alpha", "month"] + selected["exposure"].tolist()
    panel = pd.read_csv(PANEL, usecols=lambda c: c in set(panel_cols))

    rows = []
    for _, row in selected.iterrows():
        exposure = row["exposure"]
        domain = row["domain_label"]
        if exposure not in panel.columns:
            continue

        x = zscore(panel[exposure])
        annual = (
            pd.DataFrame({"year": panel["year"], "exposure_z": x})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .groupby("year", as_index=False)
            .agg(
                annual_mean_exposure_z=("exposure_z", "mean"),
                n_state_months=("exposure_z", "size"),
            )
        )
        a = annual[annual["year"].isin(YEARS)].sort_values("year")
        b = beta[beta["exposure"].eq(exposure)].sort_values("year")

        intensity_slope = slope(a["year"].to_numpy(float), a["annual_mean_exposure_z"].to_numpy(float))
        abs_beta_slope = slope(b["year"].to_numpy(float), b["abs_beta"].to_numpy(float))
        rows.append(
            {
                "domain_label": domain,
                "exposure": exposure,
                "exposure_label": short_label(exposure, ""),
                "intensity_slope_z_per_year_2015_2024": intensity_slope,
                "intensity_endpoint_change_z_2015_2024": (
                    float(a["annual_mean_exposure_z"].iloc[-1] - a["annual_mean_exposure_z"].iloc[0])
                    if len(a) >= 2
                    else np.nan
                ),
                "mean_abs_beta_2015_2024": float(b["abs_beta"].mean()) if len(b) else np.nan,
                "abs_beta_slope_per_year_2015_2024": abs_beta_slope,
                "abs_beta_endpoint_change_2015_2024": (
                    float(b["abs_beta"].iloc[-1] - b["abs_beta"].iloc[0]) if len(b) >= 2 else np.nan
                ),
                "n_intensity_years": int(len(a)),
                "n_beta_years": int(len(b)),
                "mean_n_state_months_per_year": float(a["n_state_months"].mean()) if len(a) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["domain_label"] = pd.Categorical(out["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    return out.sort_values(["domain_label", "exposure"])

def main() -> int:
    table = build_table()
    table.to_csv(OUT_TABLE, index=False)
    print(f"Wrote {OUT_TABLE}")
    print("\nQuadrant counts")
    q = table.dropna(subset=["intensity_slope_z_per_year_2015_2024", "abs_beta_slope_per_year_2015_2024"]).copy()
    q["intensity_trend"] = np.where(q["intensity_slope_z_per_year_2015_2024"] >= 0, "exposure_up", "exposure_down")
    q["abs_beta_trend"] = np.where(q["abs_beta_slope_per_year_2015_2024"] >= 0, "|beta|_up", "|beta|_down")
    print(q.groupby(["intensity_trend", "abs_beta_trend", "domain_label"], observed=True).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
