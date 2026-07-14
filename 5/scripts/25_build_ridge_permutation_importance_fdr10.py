#!/usr/bin/env python3
"""True permutation importance tests for regional milk-loss risk ridge models."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
TAB5 = ROOT / 'analysis/statistics/5/tables'
TEST_YEARS = list(range(2015, 2026))
REGION_ORDER = ['South', 'West', 'Midwest', 'Northeast']
ALPHAS = [0.1, 1.0, 10.0, 100.0]
N_PERM = 500
SEED = 20260709

REGION = {
    'AZ':'West','CA':'West','CO':'West','ID':'West','NM':'West','OR':'West','UT':'West','WA':'West',
    'IA':'Midwest','IL':'Midwest','IN':'Midwest','KS':'Midwest','MI':'Midwest','MN':'Midwest','MO':'Midwest','OH':'Midwest','SD':'Midwest','WI':'Midwest',
    'NY':'Northeast','PA':'Northeast','VT':'Northeast',
    'FL':'South','GA':'South','KY':'South','TX':'South','VA':'South'
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
    pen[0, 0] = 0
    beta = np.linalg.solve(x.T @ x + pen, x.T @ y)
    return xt @ beta, beta

def rmse(y, pred, w=None):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if w is None:
        return float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))
    ww = np.asarray(w, float)[ok]
    return float(np.sqrt(np.average((y[ok] - pred[ok]) ** 2, weights=ww)))

def build_design(panel, feature_cols):
    base = panel[[
        'state_alpha','region','year','month','next_year','next_month','milk_cows_head',
        'log_milk_per_cow','log_milk_per_cow_lag1','next_loss_pct'
    ]].copy()
    base['year_centered'] = base['year'] - 2012.5
    base['log_milk_cows'] = np.log(base['milk_cows_head'].clip(lower=1))
    dummies = pd.concat([
        one_hot(base['state_alpha'], 'state'),
        one_hot(base['next_month'].astype(int), 'month')
    ], axis=1)
    design = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True), panel[feature_cols].reset_index(drop=True)], axis=1)
    base_cols = ['log_milk_per_cow', 'log_milk_per_cow_lag1', 'year_centered', 'log_milk_cows'] + list(dummies.columns)
    return design, base_cols

def choose_alpha(train, feature_cols, base_cols, y_col):
    if train['year'].nunique() < 3:
        return 10.0
    val_year = int(train['year'].max())
    tr = train[train['year'] < val_year]
    va = train[train['year'] == val_year]
    if len(tr) < 50 or len(va) < 10:
        return 10.0
    cols = base_cols + feature_cols
    xtr, xva = standardize_train_test(tr[cols], va[cols])
    ytr = tr[y_col].to_numpy(float)
    best_score = np.inf
    best_alpha = 10.0
    for alpha in ALPHAS:
        pred, _ = ridge_fit_predict(xtr, ytr, xva, alpha)
        score = rmse(va[y_col], pred, va['milk_cows_head'])
        if score < best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = len(order)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q

def permutation_importance_region_year(test, xte, pred, beta, feature_cols, base_cols, region, year, rng):
    dat_mask = test['region'].eq(region).to_numpy()
    if dat_mask.sum() < 6:
        return []
    y = test.loc[dat_mask, 'next_loss_pct'].to_numpy(float)
    w = test.loc[dat_mask, 'milk_cows_head'].to_numpy(float)
    pred_region = pred[dat_mask]
    base_rmse = rmse(y, pred_region, w)
    if not np.isfinite(base_rmse):
        return []
    x_region = xte.loc[dat_mask, feature_cols]
    rows = []
    offset = 1 + len(base_cols)
    for j, feature in enumerate(feature_cols):
        xj = x_region.iloc[:, j].to_numpy(float)
        if np.nanstd(xj) <= 1e-12:
            rows.append((region, year, feature, 0.0, 1.0, 0.0, int(dat_mask.sum())))
            continue
        bj = beta[offset + j]
        if not np.isfinite(bj) or abs(bj) <= 1e-12:
            rows.append((region, year, feature, 0.0, 1.0, 0.0, int(dat_mask.sum())))
            continue
        deltas = np.empty(N_PERM, dtype=float)
        for b in range(N_PERM):
            xp = rng.permutation(xj)
            pp = pred_region + bj * (xp - xj)
            deltas[b] = rmse(y, pp, w) - base_rmse
        importance = float(np.mean(deltas))
        p_emp = float((1 + np.sum(deltas <= 0)) / (N_PERM + 1))
        rows.append((region, year, feature, max(importance, 0.0), p_emp, float(np.mean(deltas)), int(dat_mask.sum())))
    return rows

def main() -> int:
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(TAB5 / 'point5_forecast_state_month_feature_panel.csv', low_memory=False)
    meta = pd.read_csv(TAB5 / 'point5_forecast_feature_dictionary.csv', low_memory=False)
    feature_cols = meta.loc[meta['feature_group'].eq('exposure_node'), 'feature'].tolist()
    feature_cols = [f for f in feature_cols if f in panel.columns]
    panel['region'] = panel['state_alpha'].map(REGION)
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['next_loss_pct','log_milk_per_cow','log_milk_per_cow_lag1','milk_cows_head','region']
    ).copy()
    design, base_cols = build_design(panel, feature_cols)
    all_rows = []
    for test_year in TEST_YEARS:
        train = design[design['year'] < test_year].copy()
        test = design[design['year'] == test_year].copy()
        alpha = choose_alpha(train, feature_cols, base_cols, 'next_loss_pct')
        cols = base_cols + feature_cols
        xtr, xte = standardize_train_test(train[cols], test[cols])
        ytr = train['next_loss_pct'].to_numpy(float)
        pred, beta = ridge_fit_predict(xtr, ytr, xte, alpha)
        for region in REGION_ORDER:
            all_rows.extend(permutation_importance_region_year(test, xte, pred, beta, feature_cols, base_cols, region, test_year, rng))
        print(f'Finished {test_year} alpha={alpha} rows={len(all_rows)}')
    out = pd.DataFrame(all_rows, columns=[
        'region','test_year','feature','permutation_importance_rmse','p_value','mean_delta_rmse','n_state_months'
    ])
    out['fdr_p_value'] = np.nan
    for (region, year), idx in out.groupby(['region','test_year']).groups.items():
        out.loc[idx, 'fdr_p_value'] = bh_fdr(out.loc[idx, 'p_value'].to_numpy(float))
    out['significant_importance'] = np.where((out['fdr_p_value'] < 0.10) & (out['permutation_importance_rmse'] > 0), out['permutation_importance_rmse'], 0.0)
    out = out.merge(meta, on='feature', how='left')
    out.to_csv(TAB5 / 'point5_exposome_milk_loss_risk_region_node_permutation_fdr10_importance_long.csv', index=False)
    summary = out.groupby(['region','domain_label','subdomain_label','exposure','feature'], as_index=False).agg(
        max_significant_importance=('significant_importance','max'),
        mean_significant_importance=('significant_importance','mean'),
        min_fdr_p_value=('fdr_p_value','min'),
        n_significant_years=('significant_importance', lambda x: int(np.sum(np.asarray(x) > 0)))
    )
    summary.to_csv(TAB5 / 'point5_exposome_milk_loss_risk_region_node_permutation_fdr10_importance_summary.csv', index=False)
    union = summary[summary['max_significant_importance'] > 0].copy()
    union = union.sort_values('max_significant_importance', ascending=False).drop_duplicates(['exposure','domain_label','subdomain_label'])
    union.to_csv(TAB5 / 'point5_exposome_milk_loss_risk_region_node_permutation_fdr10_union.csv', index=False)
    print('Wrote permutation FDR10 importance tables')
    print('significant region-year-feature cells:', int((out['significant_importance'] > 0).sum()))
    print('union nodes:', len(union))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
