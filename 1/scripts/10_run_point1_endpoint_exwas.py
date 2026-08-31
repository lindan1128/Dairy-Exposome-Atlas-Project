#!/usr/bin/env python3
"""Run the Point 1 endpoint exposure-wide association screen.

This script implements the manuscript Methods endpoint ExWAS: each curated
numeric exposure is tested independently against total production and milk
production per cow using log outcomes, standardized exposures, state and
year-month fixed effects, milk-cow-inventory WLS weights and state-clustered
standard errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
ROOT = POINT.parents[3]
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

TAB = POINT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

CLEAN_DICT = (
    ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
OUT = TAB / "point1_endpoint_exwas_associations.csv"
AUDIT = TAB / "point1_endpoint_exwas_audit.csv"

ENDPOINTS = {
    "per_cow_50": ("milk_per_cow_lb", "Milk production per cow"),
    "total_same_50": ("milk_production_lb", "Total milk production"),
}
EXCLUDE_FLAGS = {"no", "n", "false", "0", "exclude", "excluded", "unused"}
EXCLUDE_CLASSES = {"Dairy scale", "Herd structure / scale"}
BONFERRONI_N = 204


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / sd if np.isfinite(sd) and sd > 0 else np.zeros_like(x, dtype=float)


def weighted_r2_from_design(data: pd.DataFrame, y_col: str, x_col: str) -> tuple[float, float, float, float, int]:
    needed = ["state_alpha", "year", "month", "milk_cows_head", y_col, x_col]
    d = data.loc[:, needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d.loc[d.milk_cows_head.gt(0)].copy()
    if len(d) < 60 or d.state_alpha.nunique() < 3:
        return (np.nan, np.nan, np.nan, np.nan, len(d))
    y = np.log(d[y_col].to_numpy(float))
    x = standardize(d[x_col].to_numpy(float))
    w = d.milk_cows_head.to_numpy(float)
    sw = np.sqrt(w / np.nanmean(w))
    fe = L.fixed_effects(d, spec="twoway").to_numpy(float)
    z_full = np.column_stack([fe, x])
    z_reduced = fe
    y_w = y * sw
    z_full_w = z_full * sw[:, None]
    z_reduced_w = z_reduced * sw[:, None]
    beta_full, *_ = np.linalg.lstsq(z_full_w, y_w, rcond=None)
    beta_reduced, *_ = np.linalg.lstsq(z_reduced_w, y_w, rcond=None)
    yhat_full = z_full @ beta_full
    yhat_reduced = z_reduced @ beta_reduced
    ybar = np.average(y, weights=w)
    sst = np.sum(w * (y - ybar) ** 2)
    rss_full = np.sum(w * (y - yhat_full) ** 2)
    rss_reduced = np.sum(w * (y - yhat_reduced) ** 2)
    if not np.isfinite(sst) or sst <= 0:
        return (np.nan, np.nan, np.nan, np.nan, len(d))
    r2 = 1 - rss_full / sst
    reduced_r2 = 1 - rss_reduced / sst
    p_full = z_full.shape[1]
    p_reduced = z_reduced.shape[1]
    n = len(d)
    adj = 1 - (1 - r2) * (n - 1) / (n - p_full - 1) if n > p_full + 1 else np.nan
    reduced_adj = 1 - (1 - reduced_r2) * (n - 1) / (n - p_reduced - 1) if n > p_reduced + 1 else np.nan
    return (float(r2), float(adj), float(reduced_r2), float(reduced_adj), n)


def read_clean_dictionary() -> pd.DataFrame:
    meta = pd.read_excel(CLEAN_DICT)
    lower = {c.lower(): c for c in meta.columns}
    exposure_col = lower.get("exposure") or lower.get("variable") or lower.get("variable_name")
    if exposure_col is None:
        raise RuntimeError("Clean exposure dictionary must contain an exposure/variable column.")
    meta = meta.rename(columns={exposure_col: "exposure"})
    if "use" in lower:
        use_col = lower["use"]
        meta = meta.loc[~meta[use_col].astype(str).str.lower().isin(EXCLUDE_FLAGS)].copy()
    for col in ("class_label", "domain", "class"):
        if col in meta.columns:
            meta = meta.loc[~meta[col].astype(str).isin(EXCLUDE_CLASSES)].copy()
    meta = meta.drop_duplicates("exposure").copy()
    return meta


def harmonize_meta_columns(meta: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "domain" not in meta.columns and "source_domain" in meta.columns:
        rename["source_domain"] = "domain"
    if "class_label" not in meta.columns and "class" in meta.columns:
        rename["class"] = "class_label"
    if "subclass_label" not in meta.columns and "subdomain" in meta.columns:
        rename["subdomain"] = "subclass_label"
    meta = meta.rename(columns=rename)
    for col in ["domain", "class_label", "subclass_label", "exposure_zh", "definition_en"]:
        if col not in meta.columns:
            meta[col] = np.nan
    return meta


def main() -> int:
    panel = L.load_panel()
    panel["year_month"] = panel.year.astype(int).astype(str) + "_" + panel.month.astype(int).astype(str).str.zfill(2)
    meta = harmonize_meta_columns(read_clean_dictionary())
    exposures = [x for x in meta.exposure.astype(str) if x in panel.columns]
    rows = []
    for phenotype_scope, (outcome_col, phenotype_label) in ENDPOINTS.items():
        for exposure in exposures:
            fit = L.fit_exposure(
                panel,
                y_col=outcome_col,
                exposures=exposure,
                spec="twoway",
                weight_col="milk_cows_head",
                standardize=True,
            )
            info = meta.loc[meta.exposure.astype(str).eq(exposure)].iloc[0].to_dict()
            result = fit.get("results", {}).get(exposure, {})
            beta = result.get("beta", np.nan)
            se = result.get("se_cluster", np.nan)
            p = result.get("p_cluster", np.nan)
            r2, adj_r2, reduced_r2, reduced_adj_r2, n_r2 = weighted_r2_from_design(panel, outcome_col, exposure)
            rows.append({
                "phenotype_scope": phenotype_scope,
                "phenotype_label": phenotype_label,
                "outcome_col": outcome_col,
                "exposure": exposure,
                "exposure_zh": info.get("exposure_zh"),
                "domain": info.get("domain"),
                "class_label": info.get("class_label"),
                "subclass_label": info.get("subclass_label"),
                "definition_en": info.get("definition_en"),
                "window": "native",
                "beta": beta,
                "se_cluster": se,
                "ci_low": beta - 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan,
                "ci_high": beta + 1.96 * se if np.isfinite(beta) and np.isfinite(se) else np.nan,
                "plot_p": p,
                "p_nominal": p,
                "r2": r2,
                "adjusted_r2": adj_r2,
                "reduced_r2": reduced_r2,
                "reduced_adjusted_r2": reduced_adj_r2,
                "incr_r2": fit.get("incr_r2", np.nan),
                "adjusted_incremental_r2": adj_r2 - reduced_adj_r2 if np.isfinite(adj_r2) and np.isfinite(reduced_adj_r2) else np.nan,
                "n": fit.get("n", n_r2),
                "n_states": fit.get("n_clusters", np.nan),
                "status": fit.get("status", "failed"),
            })
    out = pd.DataFrame(rows)
    out["by_fdr_q"] = np.nan
    out["bonferroni_threshold"] = 0.05 / BONFERRONI_N
    for phenotype_scope, idx in out.groupby("phenotype_scope").groups.items():
        pvals = out.loc[idx, "p_nominal"].to_numpy(float)
        finite = np.isfinite(pvals)
        q = np.full(len(pvals), np.nan)
        if finite.any():
            m = finite.sum()
            harmonic = np.sum(1 / np.arange(1, m + 1))
            order = np.argsort(pvals[finite])
            ranked = pvals[finite][order] * m * harmonic / np.arange(1, m + 1)
            ranked = np.minimum.accumulate(ranked[::-1])[::-1]
            ranked = np.clip(ranked, 0, 1)
            q_finite = np.empty(m)
            q_finite[order] = ranked
            q[finite] = q_finite
        out.loc[idx, "by_fdr_q"] = q
    out["by_fdr_significant"] = out.by_fdr_q < 0.05
    out["bonferroni_significant"] = out.p_nominal < out.bonferroni_threshold
    out.to_csv(OUT, index=False)
    pd.DataFrame({
        "metric": ["dictionary_exposures", "available_exposures", "bonferroni_n", "phenotypes"],
        "value": [len(meta), len(exposures), BONFERRONI_N, ",".join(ENDPOINTS.keys())],
    }).to_csv(AUDIT, index=False)
    print(f"Wrote {OUT} rows={len(out)} exposures={len(exposures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
