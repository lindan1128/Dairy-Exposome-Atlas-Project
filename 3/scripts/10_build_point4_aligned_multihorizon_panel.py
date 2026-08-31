#!/usr/bin/env python3
"""Build a Point-4-aligned, 50-state monthly panel for 1-6 month forecasts.

All 186 original Point 4 exposures are read from the monthly source table.
Every engineered predictor only uses the feature month and earlier months.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[3]
TAB, LOG = HERE / "tables", HERE / "logs"
POINT2_META = ROOT / "analysis/statistics/50/2/tables/point2_tree_hgb_feature_dictionary.csv"
MILK = ROOT / "data/us_milk/tables/analysis_1_2_4_state_month_milk_phenotypes.csv"
EXPOSURE = ROOT / "data/us_expose_new/processed/exposure_state_month_expanded.csv"
HORIZONS = range(1, 13)


def add_temporal_features(panel: pd.DataFrame, exposure: str, source_domain: str) -> dict[str, pd.Series]:
    """Return only feature-time or backward-looking transforms for one exposure."""
    value = pd.to_numeric(panel[exposure], errors="coerce").astype(float)
    by_state = value.groupby(panel.state_alpha, sort=False)
    features = {f"x__current__{exposure}": value}
    if source_domain == "Nature and climate":
        features[f"x__roll3__{exposure}"] = by_state.transform(
            lambda series: series.rolling(3, min_periods=2).mean()
        )
        features[f"x__delta12__{exposure}"] = value - by_state.shift(12)
    else:
        features[f"x__yoy_change__{exposure}"] = value - by_state.shift(12)
    return features


def build_derived_milk_panel_50(states: set[str]) -> pd.DataFrame:
    """Apply the same quarterly-total-to-monthly-equivalent rule used in 50/0-4."""
    raw = pd.read_csv(
        MILK,
        usecols=["state_alpha", "year", "month", "milk_per_cow_lb", "milk_production_lb", "milk_cows_head"],
        low_memory=False,
    )
    raw = raw.loc[raw.state_alpha.isin(states) & raw.year.between(2000, 2025)].copy()
    raw["official_per_cow_qc"] = raw.milk_per_cow_lb
    derived_raw = raw.milk_production_lb / raw.milk_cows_head
    ratio = derived_raw / raw.official_per_cow_qc
    raw["quarterly_total_record"] = (
        raw.month.isin([1, 4, 7, 10])
        & derived_raw.notna()
        & (raw.official_per_cow_qc.isna() | ratio.gt(1.5))
    )
    raw["milk_production_monthly_equiv_lb"] = raw.milk_production_lb.where(
        ~raw.quarterly_total_record, raw.milk_production_lb / 3.0
    )
    raw["derived_per_cow_lb"] = raw.milk_production_monthly_equiv_lb / raw.milk_cows_head

    rows: list[pd.Series] = []
    for _, year_data in raw.sort_values(["state_alpha", "year", "month"]).groupby(["state_alpha", "year"], sort=False):
        observed_months = year_data.loc[year_data.derived_per_cow_lb.notna(), "month"].astype(int).tolist()
        quarterly_year = bool(observed_months) and len(observed_months) <= 4 and set(observed_months).issubset({1, 4, 7, 10})
        if quarterly_year:
            for row in year_data.loc[year_data.derived_per_cow_lb.notna()].itertuples(index=False):
                for month in range(int(row.month), min(int(row.month) + 3, 13)):
                    copy = pd.Series(row._asdict())
                    copy["month"] = month
                    rows.append(copy)
        else:
            rows.extend(pd.Series(row._asdict()) for row in year_data.itertuples(index=False))
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["derived_per_cow_lb", "milk_cows_head"])
    out = out.loc[(out.milk_production_monthly_equiv_lb > 0) & (out.milk_cows_head > 0)].copy()
    out = out.sort_values(["state_alpha", "year", "month"]).drop_duplicates(["state_alpha", "year", "month"], keep="first")
    out["milk_per_cow_lb"] = out.derived_per_cow_lb
    return out[["state_alpha", "year", "month", "milk_per_cow_lb", "milk_cows_head"]]


def main() -> int:
    for directory in (TAB, LOG):
        directory.mkdir(parents=True, exist_ok=True)

    states = pd.read_csv(TAB / "point3_50_state_percow_state_list.csv")
    region = pd.read_csv(TAB / "point3_state_region_lookup.csv")
    meta_raw = pd.read_csv(POINT2_META)
    meta_raw = meta_raw.loc[meta_raw.feature_group.eq("exposure_node")].drop_duplicates("exposure").copy()
    meta_raw = meta_raw.rename(
        columns={
            "source_class": "source_domain",
            "subclass_label": "subclass_label",
            "mechanistic_domain_short": "mechanistic_subclass_short",
        }
    )
    if len(meta_raw) != 186:
        raise RuntimeError(f"Expected the fixed 186-variable Point 4 universe, found {len(meta_raw)}.")
    exposures = meta_raw.exposure.astype(str).tolist()

    source_header = pd.read_csv(EXPOSURE, nrows=0).columns
    unavailable = sorted(set(exposures).difference(source_header))
    if unavailable:
        raise RuntimeError(f"Point 4 exposures unavailable in monthly source: {unavailable}")

    milk = build_derived_milk_panel_50(set(states.state_alpha))
    exposure = pd.read_csv(
        EXPOSURE,
        usecols=["state_alpha", "year", "month", *exposures],
        low_memory=False,
    )
    panel = milk.merge(exposure, on=["state_alpha", "year", "month"], how="left", validate="one_to_one")
    panel = panel.merge(region, on="state_alpha", how="left", validate="many_to_one")
    if panel.region.isna().any():
        raise RuntimeError("Every modeled state must have a region.")
    panel = panel.sort_values(["state_alpha", "year", "month"]).reset_index(drop=True)
    panel["date_index"] = panel.year * 12 + panel.month
    panel["log_milk_per_cow"] = np.log(panel.milk_per_cow_lb.clip(lower=1.0))
    panel["log_milk_per_cow_lag1"] = panel.groupby("state_alpha").log_milk_per_cow.shift(1)
    panel["log_milk_per_cow_lag2"] = panel.groupby("state_alpha").log_milk_per_cow.shift(2)
    panel["milk_loss_lag1"] = -100.0 * (panel.log_milk_per_cow - panel.log_milk_per_cow_lag1)
    panel["log_milk_cows"] = np.log(panel.milk_cows_head.clip(lower=1.0))
    panel["year_centered"] = panel.year - 2012.5

    by_state = panel.groupby("state_alpha", sort=False)
    target_month_features: dict[str, pd.Series] = {}
    for horizon in HORIZONS:
        future_log = by_state.log_milk_per_cow.shift(-horizon)
        future_milk = by_state.milk_per_cow_lb.shift(-horizon)
        target_year = by_state.year.shift(-horizon)
        target_month = by_state.month.shift(-horizon)
        target_date = by_state.date_index.shift(-horizon)
        continuous = target_date.sub(panel.date_index).eq(horizon)
        panel[f"target_year_h{horizon}"] = target_year.where(continuous)
        panel[f"target_month_h{horizon}"] = target_month.where(continuous)
        panel[f"target_milk_h{horizon}_lb_per_cow"] = future_milk.where(continuous)
        panel[f"target_loss_h{horizon}_pct"] = (-100.0 * (future_log - panel.log_milk_per_cow)).where(continuous)
        for month in range(1, 13):
            target_month_features[f"h{horizon}__target_month_{month:02d}"] = (
                target_month.eq(month).where(continuous, False).astype(float)
            )
    panel = pd.concat([panel, pd.DataFrame(target_month_features, index=panel.index)], axis=1)

    feature_rows: list[dict[str, object]] = []
    feature_data: dict[str, pd.Series] = {}
    for row in meta_raw.itertuples(index=False):
        exposure_name = str(row.exposure)
        for feature, values in add_temporal_features(panel, exposure_name, str(row.source_domain)).items():
            feature_data[feature] = values
            feature_rows.append(
                {
                    "feature": feature,
                    "exposure": exposure_name,
                    "feature_group": "exposure_node",
                    "temporal_transform": feature.split("__", 2)[1],
                    "source_domain": row.source_domain,
                    "class_label": row.class_label,
                    "subclass_label": row.subclass_label,
                    "mechanistic_subclass_short": row.mechanistic_subclass_short,
                    "exposure_zh": row.exposure_zh,
                }
            )
    panel = pd.concat([panel, pd.DataFrame(feature_data, index=panel.index)], axis=1)
    feature_meta = pd.DataFrame(feature_rows)
    if feature_meta.feature.nunique() != 459 or feature_meta.exposure.nunique() != 186:
        raise RuntimeError(
            f"Expected 459 derived features from 186 raw exposures, found "
            f"{feature_meta.feature.nunique()} from {feature_meta.exposure.nunique()}."
        )

    state_dummies = pd.get_dummies(panel.state_alpha, prefix="state", dtype=float)
    panel = pd.concat([panel, state_dummies], axis=1)
    history_rows = []
    base_history = [
        "log_milk_per_cow", "log_milk_per_cow_lag1", "log_milk_per_cow_lag2",
        "milk_loss_lag1", "year_centered", *state_dummies.columns,
    ]
    for horizon in HORIZONS:
        for feature in [*base_history, *[f"h{horizon}__target_month_{month:02d}" for month in range(1, 13)]]:
            history_rows.append(
                {
                    "horizon_months": horizon,
                    "feature": feature,
                    "feature_group": "history_baseline",
                    "temporal_transform": "baseline",
                }
            )

    target_cols = [
        column for horizon in HORIZONS for column in (
            f"target_year_h{horizon}", f"target_month_h{horizon}",
            f"target_milk_h{horizon}_lb_per_cow", f"target_loss_h{horizon}_pct",
            *[f"h{horizon}__target_month_{month:02d}" for month in range(1, 13)],
        )
    ]
    core_cols = [
        "state_alpha", "region", "year", "month", "date_index", "milk_cows_head",
        "milk_per_cow_lb", "log_milk_per_cow", "log_milk_per_cow_lag1",
        "log_milk_per_cow_lag2", "milk_loss_lag1", "log_milk_cows", "year_centered",
    ]
    keep = [*core_cols, *target_cols, *state_dummies.columns, *feature_meta.feature.tolist()]
    panel[keep].to_csv(TAB / "point3_point4aligned_multihorizon_model_panel.csv", index=False)
    feature_meta.to_csv(TAB / "point3_point4aligned_multihorizon_exposure_dictionary.csv", index=False)
    pd.DataFrame(history_rows).to_csv(TAB / "point3_point4aligned_multihorizon_history_dictionary.csv", index=False)
    manifest = {
        "raw_exposure_dictionary": str(POINT2_META),
        "n_original_exposures": 186,
        "n_forecast_features": 459,
        "temporal_policy": "current; climate also roll3 and delta12; non-climate also yoy_change; all transforms use feature month or earlier only",
        "horizons_months": list(HORIZONS),
        "states": int(panel.state_alpha.nunique()),
        "rows": int(len(panel)),
    }
    (LOG / "point3_point4aligned_multihorizon_panel_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
