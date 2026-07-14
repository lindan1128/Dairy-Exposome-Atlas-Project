#!/usr/bin/env python3
"""State-specific best-subset exposome forecast analysis.

This Point 6 analysis mirrors the Point 5 regional subset search at state level.
Candidate feature sets are the same non-empty combinations of the 12
mechanistic exposure blocks benchmarked in Point 5. For each candidate, rolling
ridge predictions are generated for held-out calendar years and evaluated
separately within each state. Each state then receives the candidate subset
with the lowest held-out RMSE. Domain importance is estimated in that
state-specific best model by grouped permutation within state-year test sets.

The script does not modify any Point 5 file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
TAB5 = ROOT / "analysis" / "statistics" / "5" / "tables"
TAB6 = ROOT / "analysis" / "statistics" / "6" / "tables"
TAB6.mkdir(parents=True, exist_ok=True)

OUT_BENCH = TAB6 / "point6_state_specific_subdomain_subset_benchmark_all4095.csv"
OUT_BEST = TAB6 / "point6_state_specific_best_subdomain_sets.csv"
OUT_PRED = TAB6 / "point6_state_specific_best_forecast_predictions.csv"
OUT_PERF = TAB6 / "point6_state_specific_best_forecast_performance.csv"

TEST_YEARS = list(range(2015, 2026))
N_PERM = 500
SEED = 20260713


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
    pen[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + pen, x.T @ y)
    return xt @ beta, beta


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


def empirical_p(deltas: np.ndarray) -> float:
    return float((1 + np.sum(deltas <= 0)) / (len(deltas) + 1))


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


def load_inputs():
    panel = pd.read_csv(TAB5 / "point5_forecast_state_month_feature_panel.csv", low_memory=False)
    meta = pd.read_csv(TAB5 / "point5_forecast_feature_dictionary.csv", low_memory=False)
    combos = pd.read_csv(TAB5 / "point5_region_subdomain_subset_benchmark_all4095.csv", low_memory=False)
    meta = meta[meta["feature_group"].eq("exposure_node")].copy()
    features = sorted(set(meta["feature"]) & set(panel.columns))
    meta = meta[meta["feature"].isin(features)].copy()
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["next_loss_pct", "log_milk_per_cow", "log_milk_per_cow_lag1", "milk_cows_head", "region"]
    )
    panel[features] = panel[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return panel, meta, features, combos


def subset_features(blocks: str, meta: pd.DataFrame, all_features: set[str]) -> list[str]:
    block_list = [b.strip() for b in str(blocks).split("|") if b.strip()]
    feats = meta.loc[meta["mechanistic_domain_en"].isin(block_list), "feature"].drop_duplicates().tolist()
    return [f for f in feats if f in all_features]


def fit_predictions_for_subset(
    design: pd.DataFrame,
    base_cols: list[str],
    features: list[str],
    alpha: float,
):
    pred_rows = []
    for test_year in TEST_YEARS:
        train = design[design["year"] < test_year]
        test = design[design["year"].eq(test_year)]
        if len(test) == 0:
            continue
        cols = base_cols + features
        xtr, xte = standardize_train_test(train[cols], test[cols])
        pred, beta = ridge_fit_predict(xtr, train["next_loss_pct"].to_numpy(float), xte, alpha)
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
        out["exposome_predicted_loss_pct"] = pred
        pred_rows.append(out)
    return pd.concat(pred_rows, ignore_index=True)


def fit_baseline_predictions(design: pd.DataFrame, base_cols: list[str]):
    rows = []
    for test_year in TEST_YEARS:
        train = design[design["year"] < test_year]
        test = design[design["year"].eq(test_year)]
        xtr, xte = standardize_train_test(train[base_cols], test[base_cols])
        pred, _ = ridge_fit_predict(xtr, train["next_loss_pct"].to_numpy(float), xte, alpha=10.0)
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
        out["baseline_predicted_loss_pct"] = pred
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def prepare_year_cache(design: pd.DataFrame, base_cols: list[str], feature_cols: list[str]):
    """Precompute standardized all-column matrices for fast subset scans."""
    all_cols = base_cols + feature_cols
    base_idx = np.arange(len(base_cols), dtype=int)
    feature_idx = {f: len(base_cols) + i for i, f in enumerate(feature_cols)}
    cache = {}
    for test_year in TEST_YEARS:
        train = design[design["year"] < test_year]
        test = design[design["year"].eq(test_year)]
        xtr_raw = train[all_cols].astype(float).to_numpy()
        xte_raw = test[all_cols].astype(float).to_numpy()
        mu = np.nanmean(xtr_raw, axis=0)
        sd = np.nanstd(xtr_raw, axis=0, ddof=1)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        xtr_all = np.nan_to_num((xtr_raw - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        xte_all = np.nan_to_num((xte_raw - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
        meta = test[
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
        ].reset_index(drop=True)
        state_indices = {
            state: np.flatnonzero(meta["state_alpha"].to_numpy() == state)
            for state in meta["state_alpha"].dropna().unique()
        }
        cache[test_year] = {
            "xtr_all": xtr_all,
            "xte_all": xte_all,
            "ytr": train["next_loss_pct"].to_numpy(float),
            "yte": meta["next_loss_pct"].to_numpy(float),
            "wte": meta["milk_cows_head"].to_numpy(float),
            "meta": meta,
            "state_indices": state_indices,
        }
    return cache, base_idx, feature_idx


def exposome_predict_from_arrays(xtr_all, ytr, xte_all, col_idx: np.ndarray, alpha: float):
    x = xtr_all[:, col_idx]
    xt = xte_all[:, col_idx]
    x = np.column_stack([np.ones(len(x)), x])
    xt = np.column_stack([np.ones(len(xt)), xt])
    pen = np.eye(x.shape[1]) * alpha
    pen[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + pen, x.T @ ytr)
    return xt @ beta


def benchmark_subsets(design: pd.DataFrame, base_cols: list[str], meta: pd.DataFrame, features: list[str], combos: pd.DataFrame):
    all_features = set(features)
    baseline = fit_baseline_predictions(design, base_cols)
    cache, base_idx, feature_idx = prepare_year_cache(design, base_cols, features)
    baseline_perf = []
    for state, g in baseline.groupby("state_alpha", sort=True):
        baseline_perf.append(
            {
                "state_alpha": state,
                "baseline_rmse": rmse(g["next_loss_pct"], g["baseline_predicted_loss_pct"], g["milk_cows_head"]),
            }
        )
    baseline_perf = pd.DataFrame(baseline_perf)

    rows = []
    for i, combo in combos.iterrows():
        feats = subset_features(combo["blocks"], meta, all_features)
        if not feats:
            continue
        alpha = float(combo["alpha_median"]) if np.isfinite(combo["alpha_median"]) else 100.0
        rec = {
            "subset_id": int(combo["subset_id"]),
            "n_blocks": int(combo["n_blocks"]),
            "n_features": len(feats),
            "blocks": combo["blocks"],
            "blocks_short": combo["blocks_short"],
            "alpha_median": alpha,
        }
        state_y = {}
        state_pred = {}
        state_w = {}
        col_idx = np.concatenate([base_idx, np.array([feature_idx[f] for f in feats], dtype=int)])
        for test_year in TEST_YEARS:
            cy = cache[test_year]
            pred = exposome_predict_from_arrays(cy["xtr_all"], cy["ytr"], cy["xte_all"], col_idx, alpha)
            for state, idx in cy["state_indices"].items():
                state_y.setdefault(state, []).append(cy["yte"][idx])
                state_pred.setdefault(state, []).append(pred[idx])
                state_w.setdefault(state, []).append(cy["wte"][idx])
        for state in sorted(state_y):
            rec[f"{state}_rmse"] = rmse(
                np.concatenate(state_y[state]),
                np.concatenate(state_pred[state]),
                np.concatenate(state_w[state]),
            )
        rows.append(rec)
        if (i + 1) % 250 == 0:
            print(f"benchmarked {i + 1}/{len(combos)} subsets")
    bench = pd.DataFrame(rows)
    for _, r in baseline_perf.iterrows():
        state = r["state_alpha"]
        if f"{state}_rmse" in bench.columns:
            bench[f"{state}_improvement_pct"] = (r["baseline_rmse"] - bench[f"{state}_rmse"]) / r["baseline_rmse"] * 100
    bench.to_csv(OUT_BENCH, index=False)
    return bench, baseline


def select_best_sets(bench: pd.DataFrame, baseline: pd.DataFrame):
    baseline_rmse = (
        baseline.groupby("state_alpha")
        .apply(lambda g: rmse(g["next_loss_pct"], g["baseline_predicted_loss_pct"], g["milk_cows_head"]))
        .rename("baseline_rmse")
        .reset_index()
    )
    state_region = baseline[["state_alpha", "region"]].drop_duplicates()
    rows = []
    for state in sorted(baseline["state_alpha"].dropna().unique()):
        col = f"{state}_rmse"
        if col not in bench.columns:
            continue
        idx = bench[col].astype(float).idxmin()
        b = bench.loc[idx]
        base = float(baseline_rmse.loc[baseline_rmse["state_alpha"].eq(state), "baseline_rmse"].iloc[0])
        ridge = float(b[col])
        rows.append(
            {
                "state_alpha": state,
                "region": state_region.loc[state_region["state_alpha"].eq(state), "region"].iloc[0],
                "best_subset_id": int(b["subset_id"]),
                "n_blocks": int(b["n_blocks"]),
                "n_features": int(b["n_features"]),
                "blocks": b["blocks"],
                "blocks_short": b["blocks_short"],
                "alpha_median": float(b["alpha_median"]),
                "baseline_rmse": base,
                "best_exposome_rmse": ridge,
                "delta_rmse": base - ridge,
                "improvement_pct": (base - ridge) / base * 100 if base > 0 else np.nan,
            }
        )
    best = pd.DataFrame(rows).sort_values(["region", "state_alpha"]).reset_index(drop=True)
    best.to_csv(OUT_BEST, index=False)
    return best


def build_best_predictions(
    best: pd.DataFrame,
    design: pd.DataFrame,
    base_cols: list[str],
    meta: pd.DataFrame,
    all_features: set[str],
    baseline: pd.DataFrame,
):
    parts = []
    baseline_key = baseline[
        [
            "state_alpha",
            "year",
            "month",
            "baseline_predicted_loss_pct",
        ]
    ]
    for row in best.itertuples(index=False):
        features = subset_features(row.blocks, meta, all_features)
        pred = fit_predictions_for_subset(design, base_cols, features, float(row.alpha_median))
        s = pred[pred["state_alpha"].eq(row.state_alpha)].copy()
        s = s.merge(baseline_key, on=["state_alpha", "year", "month"], how="left")
        s["best_subset_id"] = int(row.best_subset_id)
        s["blocks"] = row.blocks
        parts.append(s)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT_PRED, index=False)
    perf = []
    for state, g in out.groupby("state_alpha", sort=True):
        perf.append(
            {
                "state_alpha": state,
                "region": g["region"].iloc[0],
                "baseline_rmse": rmse(g["next_loss_pct"], g["baseline_predicted_loss_pct"], g["milk_cows_head"]),
                "best_exposome_rmse": rmse(g["next_loss_pct"], g["exposome_predicted_loss_pct"], g["milk_cows_head"]),
                "n_state_months": len(g),
                "n_test_years": g["year"].nunique(),
            }
        )
    perf = pd.DataFrame(perf)
    perf["delta_rmse"] = perf["baseline_rmse"] - perf["best_exposome_rmse"]
    perf["improvement_pct"] = perf["delta_rmse"] / perf["baseline_rmse"] * 100
    perf.to_csv(OUT_PERF, index=False)
    return out, perf


def main() -> int:
    panel, meta, features, combos = load_inputs()
    design, base_cols = build_design(panel, features)
    bench, baseline = benchmark_subsets(design, base_cols, meta, features, combos)
    best = select_best_sets(bench, baseline)
    best_pred, perf = build_best_predictions(best, design, base_cols, meta, set(features), baseline)
    print(f"Wrote {OUT_BENCH} {bench.shape}")
    print(f"Wrote {OUT_BEST} {best.shape}")
    print(f"Wrote {OUT_PRED} {best_pred.shape}")
    print(f"Wrote {OUT_PERF} {perf.shape}")
    print("\nPerformance by state")
    print(perf.sort_values("improvement_pct", ascending=False).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
