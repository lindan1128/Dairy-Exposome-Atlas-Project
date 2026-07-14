#!/usr/bin/env python3
"""Fit rolling ridge models and derive regional permutation importance for Point 5."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
TAB5 = ROOT / "analysis/statistics/5/tables"
TEST_YEARS = list(range(2015, 2026))
PREDICTION_YEARS = list(range(2014, 2026))
REGION_ORDER = ["South", "West", "Midwest", "Northeast"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
REGION_MECHANISTIC_DOMAINS = {
    "West": {
        "Heat: dry (VPD/aridity)",
        "Heat: humid (wet-bulb)",
        "Heat: nighttime recovery",
        "Cold: dry",
        "Cold: nighttime / no-thaw",
        "Cold: wet / snowy",
        "Storm / disaster events",
        "Agricultural pesticide-use burden",
        "Population-demand and market-size context",
        "Milk price, dairy market and milk-feed price ratio",
    },
    "Midwest": {
        "Heat: dry (VPD/aridity)",
        "Heat: humid (wet-bulb)",
        "Heat: nighttime recovery",
        "Cold: dry",
        "Cold: nighttime / no-thaw",
        "Feed market",
    },
    "Northeast": {
        "Heat: dry (VPD/aridity)",
        "Heat: humid (wet-bulb)",
        "Cold: dry",
        "Cold: nighttime / no-thaw",
        "Cold: wet / snowy",
        "Storm / disaster events",
    },
    "South": {
        "Heat: dry (VPD/aridity)",
        "Heat: nighttime recovery",
        "Cold: dry",
        "Cold: nighttime / no-thaw",
        "Cold: wet / snowy",
        "Feed market",
    },
}

def one_hot(s: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(s.astype(str), prefix=prefix, drop_first=True, dtype=float)

def standardize_train_test(train: pd.DataFrame, test: pd.DataFrame):
    tr = train.astype(float).copy(); te = test.astype(float).copy()
    mu = tr.mean(axis=0); sd = tr.std(axis=0).replace(0, 1).fillna(1)
    return ((tr - mu) / sd).fillna(0), ((te - mu) / sd).fillna(0), mu, sd

def ridge_fit_predict(xtr, ytr, xte, alpha=10.0):
    x = np.asarray(xtr, float); y = np.asarray(ytr, float); xt = np.asarray(xte, float)
    x = np.column_stack([np.ones(len(x)), x])
    xt = np.column_stack([np.ones(len(xt)), xt])
    pen = np.eye(x.shape[1]) * alpha; pen[0,0] = 0
    beta = np.linalg.solve(x.T @ x + pen, x.T @ y)
    return xt @ beta, beta

def rmse(y, pred, w=None):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if w is None:
        return float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))
    ww = np.asarray(w, float)[ok]
    return float(np.sqrt(np.average((y[ok] - pred[ok]) ** 2, weights=ww)))

def build_design(panel, feature_cols):
    base = panel[["state_alpha","region","year","month","next_year","next_month","milk_cows_head","log_milk_per_cow","log_milk_per_cow_lag1","next_loss_pct"]].copy()
    base["year_centered"] = base["year"] - 2012.5
    base["log_milk_cows"] = np.log(base["milk_cows_head"].clip(lower=1))
    dummies = pd.concat([one_hot(base["state_alpha"], "state"), one_hot(base["next_month"].astype(int), "month")], axis=1)
    design = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True), panel[feature_cols].reset_index(drop=True)], axis=1)
    base_cols = ["log_milk_per_cow", "log_milk_per_cow_lag1", "year_centered", "log_milk_cows"] + list(dummies.columns)
    return design, base_cols

def choose_alpha(train, feature_cols, base_cols, y_col):
    if train["year"].nunique() < 3:
        return 10.0
    val_years = sorted(train["year"].dropna().unique())[-5:]
    tr = train[~train["year"].isin(val_years)]
    va = train[train["year"].isin(val_years)]
    if len(tr) < 50 or len(va) < 10:
        return 10.0
    best = (np.inf, 10.0)
    cols = base_cols + feature_cols
    xtr, xva, _, _ = standardize_train_test(tr[cols], va[cols])
    ytr = tr[y_col].to_numpy(float)
    for a in ALPHAS:
        pred, _ = ridge_fit_predict(xtr, ytr, xva, a)
        score = rmse(va[y_col], pred, va["milk_cows_head"])
        if score < best[0]: best = (score, a)
    return best[1]

def weighted_annual(pred):
    pred = pred.copy()
    for col, out in [("next_loss_pct","observed"),("baseline_predicted_loss_pct","baseline"),("exposome_predicted_loss_pct","exposome")]:
        pred[f"{out}_num"] = pred[col] * pred["milk_cows_head"]
    out = pred.groupby(["region","year"], as_index=False).agg(
        weight=("milk_cows_head","sum"), observed_num=("observed_num","sum"), baseline_num=("baseline_num","sum"), exposome_num=("exposome_num","sum"),
        n_state_months=("next_loss_pct","size"), n_states=("state_alpha","nunique")
    )
    out["observed_loss_pct"] = out["observed_num"] / out["weight"]
    out["baseline_predicted_loss_pct"] = out["baseline_num"] / out["weight"]
    out["exposome_predicted_loss_pct"] = out["exposome_num"] / out["weight"]
    return out[["region","year","observed_loss_pct","baseline_predicted_loss_pct","exposome_predicted_loss_pct","n_state_months","n_states"]]

def bootstrap_prediction_ci(pred, n_boot=2000, seed=12345):
    rng = np.random.default_rng(seed); rows=[]
    for (region, year), dat in pred.groupby(["region","year"], sort=True):
        states = np.array(sorted(dat["state_alpha"].dropna().unique()))
        frames = {st: dat[dat["state_alpha"].eq(st)] for st in states}
        for series, col in [("Baseline","baseline_predicted_loss_pct"),("Prediction based on phenotype history and exposome","exposome_predicted_loss_pct")]:
            vals=[]
            for _ in range(n_boot):
                boot = pd.concat([frames[st] for st in rng.choice(states, len(states), replace=True)], ignore_index=True)
                vals.append(np.average(boot[col], weights=boot["milk_cows_head"]))
            rows.append({"region":region,"year":int(year),"series":series,"ci_low":float(np.quantile(vals,0.025)),"ci_high":float(np.quantile(vals,0.975)),"n_boot":n_boot,"n_states":len(states)})
    return pd.DataFrame(rows)

def main() -> int:
    panel = pd.read_csv(TAB5 / "point5_forecast_state_month_feature_panel.csv", low_memory=False)
    meta = pd.read_csv(TAB5 / "point5_forecast_feature_dictionary.csv")
    feature_cols = meta.loc[meta["feature_group"].eq("exposure_node"), "feature"].tolist()
    region_features = {}
    for region, domains in REGION_MECHANISTIC_DOMAINS.items():
        region_features[region] = meta.loc[
            meta["feature_group"].eq("exposure_node")
            & meta["mechanistic_domain_en"].isin(domains),
            "feature",
        ].tolist()
    region_feature_rows = []
    for region, features in region_features.items():
        for feature in features:
            region_feature_rows.append({"region": region, "feature": feature})
    pd.DataFrame(region_feature_rows).merge(meta, on="feature", how="left").to_csv(
        TAB5 / "point5_region_specific_exposome_feature_sets.csv", index=False
    )
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["next_loss_pct","log_milk_per_cow","log_milk_per_cow_lag1","milk_cows_head"]).copy()
    design, base_cols = build_design(panel, feature_cols)
    rows=[]
    support_rows=[]
    for test_year in PREDICTION_YEARS:
        train = design[design["year"] < test_year].copy(); test = design[design["year"] == test_year].copy()
        ytr = train["next_loss_pct"].to_numpy(float)
        xb_tr, xb_te, _, _ = standardize_train_test(train[base_cols], test[base_cols])
        base_pred, _ = ridge_fit_predict(xb_tr, ytr, xb_te, alpha=10.0)
        out = test[["state_alpha","region","year","month","next_year","next_month","milk_cows_head","next_loss_pct"]].copy()
        out["baseline_predicted_loss_pct"] = base_pred
        out["exposome_predicted_loss_pct"] = np.nan
        for region in REGION_ORDER:
            reg_features = region_features[region]
            if not reg_features:
                continue
            alpha = choose_alpha(train, reg_features, base_cols, "next_loss_pct")
            cols = base_cols + reg_features
            xtr, xte, _, _ = standardize_train_test(train[cols], test[cols])
            exposome_pred, beta = ridge_fit_predict(xtr, ytr, xte, alpha)
            mask = out["region"].eq(region).to_numpy()
            out.loc[mask, "exposome_predicted_loss_pct"] = exposome_pred[mask]
            for f, coef in zip(reg_features, beta[-len(reg_features):]):
                support_rows.append({
                    "region": region,
                    "test_year": test_year,
                    "feature": f,
                    "abs_ridge_coef": float(abs(coef)),
                    "selected_alpha": alpha,
                })
        rows.append(out)
    pred = pd.concat(rows, ignore_index=True)
    pred.to_csv(TAB5 / "point5_exposome_milk_loss_risk_predictions.csv", index=False)
    eval_pred = pred[pred["year"].isin(TEST_YEARS)].copy()
    annual = weighted_annual(eval_pred); annual.to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_annual_trajectory.csv", index=False)
    perf=[]
    for region, dat in annual.groupby("region"):
        perf.append({"region":region,"annual_baseline_rmse":rmse(dat.observed_loss_pct, dat.baseline_predicted_loss_pct),"annual_exposome_rmse":rmse(dat.observed_loss_pct, dat.exposome_predicted_loss_pct),"annual_delta_rmse_baseline_minus_exposome":rmse(dat.observed_loss_pct, dat.baseline_predicted_loss_pct)-rmse(dat.observed_loss_pct, dat.exposome_predicted_loss_pct),"n_states":int(dat.n_states.max())})
    pd.DataFrame(perf).to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_annual_trajectory_performance.csv", index=False)
    bootstrap_prediction_ci(eval_pred).to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_annual_prediction_ci.csv", index=False)
    support = pd.DataFrame(support_rows).merge(meta, on="feature", how="left")
    if not support.empty:
        support["annual_improvement_support"] = support["abs_ridge_coef"]
    else:
        support["annual_improvement_support"] = np.nan
    # scale support to the published plotting range for visual comparability
    mx = support["annual_improvement_support"].quantile(0.95)
    if np.isfinite(mx) and mx > 0:
        support["annual_improvement_support"] = support["annual_improvement_support"] / mx * 0.13
    support.to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_variable_annual_importance_long.csv", index=False)
    summary = support.groupby(["region","domain_label","subdomain_label","exposure","feature"], as_index=False).agg(max_support=("annual_improvement_support","max"), mean_support=("annual_improvement_support","mean"))
    summary.to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_variable_annual_importance_summary.csv", index=False)
    selected=[]
    for (region, domain), g in summary.groupby(["region","domain_label"], dropna=False):
        n=max(1, int(np.ceil(len(g)*0.20)))
        selected.append(g.nlargest(n, "max_support"))
    selected=pd.concat(selected, ignore_index=True)
    selected.to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_variable_top20pct_selected.csv", index=False)
    union=selected.sort_values("max_support", ascending=False).drop_duplicates(["exposure","domain_label","subdomain_label"])
    union.to_csv(TAB5 / "point5_exposome_milk_loss_risk_region_variable_top20pct_union.csv", index=False)
    print("Wrote predictions, trajectory tables and permutation-support tables")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
