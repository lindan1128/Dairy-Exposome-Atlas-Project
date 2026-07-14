#!/usr/bin/env python3
"""State-level raw-feature permutation using the Point 5 regional model logic.

This sensitivity analysis keeps the Point 5 region-specific feature sets and
ridge-fitting workflow, but evaluates permutation importance within each
state-year test set. It is intended to check whether state-level maps remain
consistent with the regional Point 5 fan plot.
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
ALPHAS = [0.1, 1.0, 10.0, 100.0]
N_PERM = 500
SEED = 20260713

OUT_FEATURE = TAB6 / "point6_point5consistent_state_raw_feature_permutation_rawp30_feature_long.csv"
OUT_EXPOSURE = TAB6 / "point6_point5consistent_state_raw_feature_permutation_rawp30_exposure_long.csv"
OUT_DOMAIN = TAB6 / "point6_point5consistent_state_raw_feature_permutation_rawp30_domain_summary.csv"
OUT_MODEL = TAB6 / "point6_point5consistent_state_raw_feature_permutation_model_log.csv"

FORECAST_MECHANISTIC_DOMAINS = [
    "Heat: dry (VPD/aridity)",
    "Heat: humid (wet-bulb)",
    "Heat: nighttime recovery",
    "Cold: dry",
    "Cold: nighttime / no-thaw",
    "Cold: wet / snowy",
    "Storm / disaster events",
    "Agricultural pesticide-use burden",
    "Forage condition",
    "Feed market",
    "Population-demand and market-size context",
    "Milk price, dairy market and milk-feed price ratio",
]

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
    tr = train.astype(float).copy()
    te = test.astype(float).copy()
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0).replace(0, 1).fillna(1)
    return ((tr - mu) / sd).fillna(0), ((te - mu) / sd).fillna(0)


def ridge_fit_predict(xtr, ytr, xte, alpha=10.0):
    x = np.asarray(xtr, float)
    y = np.asarray(ytr, float)
    xt = np.asarray(xte, float)
    x = np.column_stack([np.ones(len(x)), x])
    xt = np.column_stack([np.ones(len(xt)), xt])
    pen = np.eye(x.shape[1]) * alpha
    pen[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + pen, x.T @ y)
    return xt @ beta, beta


def rmse(y, pred, w=None):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() == 0:
        return np.nan
    if w is None:
        return float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))
    ww = np.asarray(w, float)[ok]
    return float(np.sqrt(np.average((y[ok] - pred[ok]) ** 2, weights=ww)))


def build_design(panel, feature_cols):
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


def choose_alpha(train, feature_cols, base_cols, y_col):
    if train["year"].nunique() < 6:
        return 100.0
    val_years = sorted(train["year"].dropna().unique())[-5:]
    tr = train[~train["year"].isin(val_years)]
    va = train[train["year"].isin(val_years)]
    if len(tr) < 100 or len(va) < 20:
        return 100.0
    cols = base_cols + feature_cols
    xtr, xva = standardize_train_test(tr[cols], va[cols])
    ytr = tr[y_col].to_numpy(float)
    best_score = np.inf
    best_alpha = 100.0
    for alpha in ALPHAS:
        pred, _ = ridge_fit_predict(xtr, ytr, xva, alpha)
        score = rmse(va[y_col], pred, va["milk_cows_head"])
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def feature_permutation_state_year(
    test,
    xte,
    pred,
    beta,
    model_features,
    candidate_features,
    base_cols,
    state,
    year,
    model_context,
    added_domain,
    rng,
):
    dat_mask = test["state_alpha"].eq(state).to_numpy()
    if dat_mask.sum() < 6:
        return []
    y = test.loc[dat_mask, "next_loss_pct"].to_numpy(float)
    w = test.loc[dat_mask, "milk_cows_head"].to_numpy(float)
    pred_state = pred[dat_mask]
    base_rmse = rmse(y, pred_state, w)
    if not np.isfinite(base_rmse):
        return []

    rows = []
    feature_to_pos = {f: j for j, f in enumerate(model_features)}
    offset = 1 + len(base_cols)
    for feature in candidate_features:
        j = feature_to_pos.get(feature)
        if j is None:
            continue
        xj = xte.loc[dat_mask, feature].to_numpy(float)
        if np.nanstd(xj) <= 1e-12:
            rows.append((state, year, model_context, added_domain, feature, 0.0, 1.0, 0.0, int(dat_mask.sum())))
            continue
        bj = beta[offset + j]
        if not np.isfinite(bj) or abs(bj) <= 1e-12:
            rows.append((state, year, model_context, added_domain, feature, 0.0, 1.0, 0.0, int(dat_mask.sum())))
            continue
        deltas = np.empty(N_PERM, dtype=float)
        for b in range(N_PERM):
            xp = rng.permutation(xj)
            pp = pred_state + bj * (xp - xj)
            deltas[b] = rmse(y, pp, w) - base_rmse
        importance = float(np.mean(deltas))
        p_emp = float((1 + np.sum(deltas <= 0)) / (N_PERM + 1))
        rows.append((state, year, model_context, added_domain, feature, max(importance, 0.0), p_emp, importance, int(dat_mask.sum())))
    return rows


def collapse_temporal_features(feature_rows: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    x = feature_rows.merge(meta, on="feature", how="left")
    x = x.sort_values(
        ["state_alpha", "test_year", "exposure", "p_value", "permutation_importance_rmse"],
        ascending=[True, True, True, True, False],
    )
    out = x.groupby(["state_alpha", "test_year", "exposure"], as_index=False).first()
    out["rawp30_importance"] = np.where(
        (out["p_value"] < 0.30) & (out["permutation_importance_rmse"] > 0),
        out["permutation_importance_rmse"],
        0.0,
    )
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(TAB5 / "point5_forecast_state_month_feature_panel.csv", low_memory=False)
    meta = pd.read_csv(TAB5 / "point5_forecast_feature_dictionary.csv", low_memory=False)
    meta = meta[
        meta["feature_group"].eq("exposure_node")
        & meta["mechanistic_domain_en"].isin(FORECAST_MECHANISTIC_DOMAINS)
    ].copy()
    feature_cols = [f for f in meta["feature"].tolist() if f in panel.columns]
    meta = meta[meta["feature"].isin(feature_cols)].copy()
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["next_loss_pct", "log_milk_per_cow", "log_milk_per_cow_lag1", "milk_cows_head", "region"]
    ).copy()
    panel[feature_cols] = panel[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    design, base_cols = build_design(panel, feature_cols)
    domain_features = {
        domain: meta.loc[meta["mechanistic_domain_en"].eq(domain), "feature"].tolist()
        for domain in FORECAST_MECHANISTIC_DOMAINS
    }
    state_region = panel[["state_alpha", "region"]].drop_duplicates().sort_values(["region", "state_alpha"])

    all_rows = []
    model_rows = []
    for test_year in TEST_YEARS:
        train = design[design["year"] < test_year].copy()
        test = design[design["year"] == test_year].copy()
        ytr = train["next_loss_pct"].to_numpy(float)
        for region, sr in state_region.groupby("region", sort=False):
            if region not in REGION_MECHANISTIC_DOMAINS:
                continue
            selected_domains = set(REGION_MECHANISTIC_DOMAINS[region])
            selected_features = []
            for domain in FORECAST_MECHANISTIC_DOMAINS:
                if domain in selected_domains:
                    selected_features.extend(domain_features[domain])
            model_specs = [("base_selected", None, selected_features, selected_features)]
            for domain in FORECAST_MECHANISTIC_DOMAINS:
                if domain in selected_domains:
                    continue
                added_features = domain_features[domain]
                model_specs.append((f"add_one::{domain}", domain, selected_features + added_features, added_features))

            for context, added_domain, model_features, candidate_features in model_specs:
                alpha = choose_alpha(train, model_features, base_cols, "next_loss_pct")
                cols = base_cols + model_features
                xtr, xte = standardize_train_test(train[cols], test[cols])
                pred, beta = ridge_fit_predict(xtr, ytr, xte, alpha)
                for state in sr["state_alpha"].tolist():
                    all_rows.extend(
                        feature_permutation_state_year(
                            test,
                            xte,
                            pred,
                            beta,
                            model_features,
                            candidate_features,
                            base_cols,
                            state,
                            test_year,
                            context,
                            added_domain,
                            rng,
                        )
                    )
                model_rows.append(
                    {
                        "region": region,
                        "test_year": test_year,
                        "model_context": context,
                        "added_domain": added_domain,
                        "n_model_features": len(model_features),
                        "n_permuted_features": len(candidate_features),
                        "selected_alpha": alpha,
                    }
                )
        print(f"Finished {test_year}: feature rows={len(all_rows)}")

    feature_out = pd.DataFrame(
        all_rows,
        columns=[
            "state_alpha",
            "test_year",
            "model_context",
            "added_domain",
            "feature",
            "permutation_importance_rmse",
            "p_value",
            "mean_delta_rmse",
            "n_state_months",
        ],
    ).merge(state_region, on="state_alpha", how="left")
    feature_out = feature_out.merge(meta, on="feature", how="left")
    feature_out.to_csv(OUT_FEATURE, index=False)
    pd.DataFrame(model_rows).to_csv(OUT_MODEL, index=False)

    exposure_out = collapse_temporal_features(
        feature_out[
            [
                "state_alpha",
                "test_year",
                "model_context",
                "added_domain",
                "feature",
                "permutation_importance_rmse",
                "p_value",
                "mean_delta_rmse",
                "n_state_months",
            ]
        ],
        meta,
    ).merge(state_region, on="state_alpha", how="left")
    exposure_out.to_csv(OUT_EXPOSURE, index=False)

    domain_summary = exposure_out.groupby(
        ["state_alpha", "region", "source_class", "domain_label"],
        as_index=False,
    ).agg(
        mean_importance_rmse=("rawp30_importance", "mean"),
        max_importance_rmse=("rawp30_importance", "max"),
        min_p_value=("p_value", "min"),
        n_rawp30_exposures=("rawp30_importance", lambda x: int(np.sum(np.asarray(x) > 0))),
    )
    domain_lookup = meta[["source_class", "domain_label"]].drop_duplicates()
    complete_grid = state_region.merge(domain_lookup, how="cross")
    domain_summary = complete_grid.merge(
        domain_summary,
        on=["state_alpha", "region", "source_class", "domain_label"],
        how="left",
    )
    domain_summary["mean_importance_rmse"] = domain_summary["mean_importance_rmse"].fillna(0.0)
    domain_summary["max_importance_rmse"] = domain_summary["max_importance_rmse"].fillna(0.0)
    domain_summary["min_p_value"] = domain_summary["min_p_value"].fillna(1.0)
    domain_summary["n_rawp30_exposures"] = domain_summary["n_rawp30_exposures"].fillna(0).astype(int)
    domain_summary.to_csv(OUT_DOMAIN, index=False)
    print("Wrote Point 5-consistent state-level raw-feature permutation tables")
    print("feature rows:", len(feature_out))
    print("state-domain rows:", len(domain_summary))
    print("shown state-domain signals:", int((domain_summary["mean_importance_rmse"] > 0).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
