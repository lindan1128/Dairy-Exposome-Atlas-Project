#!/usr/bin/env python3
"""Regional observed and predicted next-month milk-loss trajectories."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
TAB5 = ROOT / "analysis/statistics/5/tables"
FIG5 = ROOT / "analysis/statistics/5/figures"
FIT_SCRIPT = SCRIPT_DIR / "02_fit_exposome_ridge_risk_model.py"
PLOT_SCRIPT = SCRIPT_DIR / "17_make_ridge_regional_milk_loss_trajectory.R"
TEST_YEARS = list(range(2015, 2026))

REGION = {
    "AZ": "West", "CA": "West", "CO": "West", "ID": "West",
    "NM": "West", "OR": "West", "UT": "West", "WA": "West",
    "IA": "Midwest", "IL": "Midwest", "IN": "Midwest", "KS": "Midwest",
    "MI": "Midwest", "MN": "Midwest", "MO": "Midwest", "OH": "Midwest",
    "SD": "Midwest", "WI": "Midwest",
    "NY": "Northeast", "PA": "Northeast", "VT": "Northeast",
    "FL": "South", "GA": "South", "KY": "South", "TX": "South", "VA": "South",
}
REGION_ORDER = ["South", "West", "Midwest", "Northeast"]


def load_fit_module():
    spec = importlib.util.spec_from_file_location("point5_ridge_fit", FIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def build_design(panel: pd.DataFrame, feature_cols: list[str], fit) -> tuple[pd.DataFrame, list[str]]:
    base = panel[
        [
            "state_alpha", "year", "month", "next_year", "next_month", "milk_cows_head",
            "log_milk_per_cow", "log_milk_per_cow_lag1", "next_loss_pct",
        ]
    ].copy()
    base["year_centered"] = base["year"] - 2012.5
    base["log_milk_cows"] = np.log(base["milk_cows_head"].clip(lower=1))
    dummies = pd.concat(
        [fit.one_hot(base["state_alpha"], "state"), fit.one_hot(base["next_month"].astype(int), "month")],
        axis=1,
    )
    design = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True), panel[feature_cols].reset_index(drop=True)], axis=1)
    base_cols = ["log_milk_per_cow", "log_milk_per_cow_lag1", "year_centered", "log_milk_cows"] + list(dummies.columns)
    return design, base_cols


def ensure_prediction_table() -> None:
    pred_path = TAB5 / "point5_exposome_milk_loss_risk_predictions.csv"
    if pred_path.exists():
        return
    fit = load_fit_module()
    panel = pd.read_csv(TAB5 / "point5_forecast_state_month_feature_panel.csv", low_memory=False)
    meta = pd.read_csv(TAB5 / "point5_forecast_feature_dictionary.csv", low_memory=False)
    temporal_cols = meta.loc[meta["feature_group"].eq("exposure_node"), "feature"].tolist()
    panel = (
        panel.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["next_loss_pct", "log_milk_per_cow", "log_milk_per_cow_lag1", "milk_cows_head"])
        .copy()
    )
    design, base_cols = build_design(panel, temporal_cols, fit)
    rows = []
    for test_year in TEST_YEARS:
        train = design[design["year"] < test_year].copy()
        test = design[design["year"] == test_year].copy()
        ytr = train["next_loss_pct"].to_numpy(float)
        xb_tr, xb_te, _, _ = fit.standardize_train_test(train[base_cols], test[base_cols])
        baseline_pred, _ = fit.ridge_fit_predict(xb_tr, ytr, xb_te, alpha=10.0)
        alpha = fit.choose_alpha(train, temporal_cols, base_cols, "next_loss_pct")
        model_cols = base_cols + temporal_cols
        xtr, xte, _, _ = fit.standardize_train_test(train[model_cols], test[model_cols])
        exposome_pred, _ = fit.ridge_fit_predict(xtr, ytr, xte, alpha=alpha)
        out = test[["state_alpha", "year", "month", "next_year", "next_month", "milk_cows_head", "next_loss_pct"]].copy()
        out["baseline_predicted_loss_pct"] = baseline_pred
        out["exposome_predicted_loss_pct"] = exposome_pred
        rows.append(out)
    pd.concat(rows, ignore_index=True).to_csv(pred_path, index=False)


def rmse(y: pd.Series, pred: pd.Series) -> float:
    ok = np.isfinite(y) & np.isfinite(pred)
    return float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2))) if ok.any() else np.nan


def weighted_annual(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["obs_num"] = pred["next_loss_pct"] * pred["milk_cows_head"]
    pred["base_num"] = pred["baseline_predicted_loss_pct"] * pred["milk_cows_head"]
    pred["exposome_num"] = pred["exposome_predicted_loss_pct"] * pred["milk_cows_head"]
    out = pred.groupby(["region", "year"], as_index=False).agg(
        weight=("milk_cows_head", "sum"),
        observed_num=("obs_num", "sum"),
        baseline_num=("base_num", "sum"),
        exposome_num=("exposome_num", "sum"),
        n_state_months=("next_loss_pct", "size"),
        n_states=("state_alpha", "nunique"),
    )
    out["observed_loss_pct"] = out["observed_num"] / out["weight"]
    out["baseline_predicted_loss_pct"] = out["baseline_num"] / out["weight"]
    out["exposome_predicted_loss_pct"] = out["exposome_num"] / out["weight"]
    return out[
        [
            "region", "year", "observed_loss_pct", "baseline_predicted_loss_pct",
            "exposome_predicted_loss_pct", "n_state_months", "n_states",
        ]
    ]


def bootstrap_prediction_ci(pred: pd.DataFrame, n_boot: int = 2000, seed: int = 12345) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (region, year), dat in pred.groupby(["region", "year"], sort=True):
        states = np.array(sorted(dat["state_alpha"].dropna().unique()))
        if len(states) == 0:
            continue
        state_frames = {st: dat.loc[dat["state_alpha"].eq(st)] for st in states}
        for series, col in [
            ("Baseline", "baseline_predicted_loss_pct"),
            ("Prediction based on phenotype history and exposome", "exposome_predicted_loss_pct"),
        ]:
            vals = []
            for _ in range(n_boot):
                sampled = rng.choice(states, size=len(states), replace=True)
                boot = pd.concat([state_frames[st] for st in sampled], ignore_index=True)
                w = boot["milk_cows_head"].to_numpy(float)
                y = boot[col].to_numpy(float)
                ok = np.isfinite(w) & np.isfinite(y) & (w > 0)
                vals.append(float(np.average(y[ok], weights=w[ok])) if ok.any() else np.nan)
            vals = np.array(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            rows.append({
                "region": region,
                "year": int(year),
                "series": series,
                "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
                "n_boot": int(len(vals)),
                "n_states": int(len(states)),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ensure_prediction_table()
    pred = pd.read_csv(TAB5 / "point5_exposome_milk_loss_risk_predictions.csv")
    pred["region"] = pred["state_alpha"].map(REGION)
    pred = pred[pred["region"].isin(REGION_ORDER)].copy()

    annual = weighted_annual(pred)
    perf_rows = []
    for region, dat in annual.groupby("region"):
        base_rmse = rmse(dat["observed_loss_pct"], dat["baseline_predicted_loss_pct"])
        ridge_rmse = rmse(dat["observed_loss_pct"], dat["exposome_predicted_loss_pct"])
        perf_rows.append(
            {
                "region": region,
                "annual_baseline_rmse": base_rmse,
                "annual_exposome_rmse": ridge_rmse,
                "annual_delta_rmse_baseline_minus_exposome": base_rmse - ridge_rmse,
                "n_states": int(dat["n_states"].max()),
            }
        )
    annual_perf = pd.DataFrame(perf_rows)

    annual.to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_annual_trajectory.csv", index=False)
    annual_perf.to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_annual_trajectory_performance.csv", index=False)
    bootstrap_prediction_ci(pred).to_csv(
        TAB5 / "point5_exposome_milk_loss_risk_region_annual_prediction_ci.csv",
        index=False,
    )

    subprocess.run(["Rscript", str(PLOT_SCRIPT)], check=True)
    print(annual_perf.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
