#!/usr/bin/env python3
"""State-level forecasts using each region's best Point 5 feature combination.

This analysis is a bridge between the regional Point 5 forecast and the
state-specific Point 6 benchmark. Each state inherits the feature combination
that performed best for its geographic region in the Point 5 regional subset
scan. Rolling ridge models are then trained on all preceding years and
evaluated within each state from 2015 to 2025.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
TAB5 = ROOT / "analysis" / "statistics" / "5" / "tables"
TAB6 = ROOT / "analysis" / "statistics" / "6" / "tables"
TAB6.mkdir(parents=True, exist_ok=True)

TEST_YEARS = list(range(2015, 2026))

OUT_REGION_BEST = TAB6 / "point6_region_best_subdomain_sets.csv"
OUT_PRED = TAB6 / "point6_region_best_forecast_by_state_predictions.csv"
OUT_PERF = TAB6 / "point6_region_best_forecast_by_state_performance.csv"


def one_hot(s: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(s.astype(str), prefix=prefix, drop_first=True, dtype=float)


def standardize_train_test(train: pd.DataFrame, test: pd.DataFrame):
    tr = train.astype(float).copy()
    te = test.astype(float).copy()
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0).replace(0, 1).fillna(1)
    return ((tr - mu) / sd).fillna(0), ((te - mu) / sd).fillna(0)


def ridge_fit_predict(xtr, ytr, xte, alpha: float):
    x = np.asarray(xtr, float)
    y = np.asarray(ytr, float)
    xt = np.asarray(xte, float)
    x = np.column_stack([np.ones(len(x)), x])
    xt = np.column_stack([np.ones(len(xt)), xt])
    pen = np.eye(x.shape[1]) * alpha
    pen[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + pen, x.T @ y)
    return xt @ beta


def rmse(y, pred, w=None) -> float:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() == 0:
        return np.nan
    if w is None:
        return float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))
    ww = np.asarray(w, float)[ok]
    return float(np.sqrt(np.average((y[ok] - pred[ok]) ** 2, weights=ww)))


def build_design(panel: pd.DataFrame, feature_cols: list[str]):
    base = panel[
        [
            "state_alpha",
            "region",
            "year",
            "month",
            "next_year",
            "next_month",
            "milk_cows_head",
            "log_milk_per_cow",
            "log_milk_per_cow_lag1",
            "next_loss_pct",
        ]
    ].copy()
    base["year_centered"] = base["year"] - 2012.5
    base["log_milk_cows"] = np.log(base["milk_cows_head"].clip(lower=1))
    dummies = pd.concat(
        [one_hot(base["state_alpha"], "state"), one_hot(base["next_month"].astype(int), "month")],
        axis=1,
    )
    design = pd.concat(
        [base.reset_index(drop=True), dummies.reset_index(drop=True), panel[feature_cols].reset_index(drop=True)],
        axis=1,
    )
    base_cols = ["log_milk_per_cow", "log_milk_per_cow_lag1", "year_centered", "log_milk_cows"] + list(
        dummies.columns
    )
    return design, base_cols


def subset_features(blocks: str, meta: pd.DataFrame, all_features: set[str]) -> list[str]:
    block_list = [b.strip() for b in str(blocks).split("|") if b.strip()]
    feats = meta.loc[meta["mechanistic_domain_en"].isin(block_list), "feature"].drop_duplicates().tolist()
    return [f for f in feats if f in all_features]


def fit_predictions_for_subset(design: pd.DataFrame, base_cols: list[str], features: list[str], alpha: float):
    rows = []
    cols = base_cols + features
    for test_year in TEST_YEARS:
        train = design[design["year"] < test_year]
        test = design[design["year"].eq(test_year)]
        if len(test) == 0:
            continue
        xtr, xte = standardize_train_test(train[cols], test[cols])
        pred = ridge_fit_predict(xtr, train["next_loss_pct"].to_numpy(float), xte, alpha)
        out = test[
            [
                "state_alpha",
                "region",
                "year",
                "month",
                "next_year",
                "next_month",
                "milk_cows_head",
                "next_loss_pct",
            ]
        ].copy()
        out["predicted_loss_pct"] = pred
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def fit_baseline_predictions(design: pd.DataFrame, base_cols: list[str]):
    return fit_predictions_for_subset(design, base_cols, [], alpha=10.0).rename(
        columns={"predicted_loss_pct": "baseline_predicted_loss_pct"}
    )


def main() -> int:
    panel = pd.read_csv(TAB5 / "point5_forecast_state_month_feature_panel.csv", low_memory=False)
    meta = pd.read_csv(TAB5 / "point5_forecast_feature_dictionary.csv", low_memory=False)
    bench = pd.read_csv(TAB5 / "point5_region_subdomain_subset_benchmark_all4095.csv", low_memory=False)

    meta = meta[meta["feature_group"].eq("exposure_node")].copy()
    feature_cols = sorted(set(meta["feature"]) & set(panel.columns))
    meta = meta[meta["feature"].isin(feature_cols)].copy()
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["next_loss_pct", "log_milk_per_cow", "log_milk_per_cow_lag1", "milk_cows_head", "region"]
    ).copy()
    panel[feature_cols] = panel[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    design, base_cols = build_design(panel, feature_cols)
    all_features = set(feature_cols)

    region_rows = []
    for region in ["South", "West", "Midwest", "Northeast"]:
        rmse_col = f"{region}_rmse"
        idx = bench[rmse_col].astype(float).idxmin()
        row = bench.loc[idx]
        region_rows.append(
            {
                "region": region,
                "subset_id": int(row["subset_id"]),
                "n_blocks": int(row["n_blocks"]),
                "n_features": int(row["n_features"]),
                "blocks": row["blocks"],
                "blocks_short": row["blocks_short"],
                "alpha_median": float(row["alpha_median"]),
                "region_rmse": float(row[rmse_col]),
                "region_improvement_pct": float(row[f"{region}_improvement_pct"]),
            }
        )
    region_best = pd.DataFrame(region_rows)
    region_best.to_csv(OUT_REGION_BEST, index=False)

    baseline = fit_baseline_predictions(design, base_cols)
    baseline_key = baseline[["state_alpha", "year", "month", "baseline_predicted_loss_pct"]]

    pred_parts = []
    for row in region_best.itertuples(index=False):
        feats = subset_features(row.blocks, meta, all_features)
        pred = fit_predictions_for_subset(design, base_cols, feats, float(row.alpha_median))
        pred = pred[pred["region"].eq(row.region)].copy()
        pred = pred.merge(baseline_key, on=["state_alpha", "year", "month"], how="left")
        pred["region_best_subset_id"] = int(row.subset_id)
        pred["region_best_blocks"] = row.blocks
        pred["region_best_alpha"] = float(row.alpha_median)
        pred_parts.append(pred)

    pred_out = pd.concat(pred_parts, ignore_index=True)
    pred_out.to_csv(OUT_PRED, index=False)

    perf_rows = []
    for state, g in pred_out.groupby("state_alpha", sort=True):
        base = rmse(g["next_loss_pct"], g["baseline_predicted_loss_pct"], g["milk_cows_head"])
        expo = rmse(g["next_loss_pct"], g["predicted_loss_pct"], g["milk_cows_head"])
        perf_rows.append(
            {
                "state_alpha": state,
                "region": g["region"].iloc[0],
                "baseline_rmse": base,
                "region_best_exposome_rmse": expo,
                "delta_rmse": base - expo,
                "improvement_pct": (base - expo) / base * 100 if base > 0 else np.nan,
                "n_state_months": len(g),
                "n_test_years": g["year"].nunique(),
                "region_best_subset_id": int(g["region_best_subset_id"].iloc[0]),
                "region_best_blocks": g["region_best_blocks"].iloc[0],
            }
        )
    perf = pd.DataFrame(perf_rows).sort_values(["region", "state_alpha"]).reset_index(drop=True)
    perf.to_csv(OUT_PERF, index=False)

    print("Region best subsets")
    print(region_best.round(4).to_string(index=False))
    print("\nState performance using region-best combinations")
    print(perf.round(4).sort_values(["region", "state_alpha"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
