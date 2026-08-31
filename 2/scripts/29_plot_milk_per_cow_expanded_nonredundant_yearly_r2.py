#!/usr/bin/env python3
"""Milk-per-cow yearly R2-family metrics for the expanded exposure set."""

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


SELECTION = TAB / "point2_common_sense_expanded_kept_variables.csv"
THREE_METRIC_VALUES = TAB / "point2_beta_std_two_stage_three_metric_by_variable.csv"
THREE_METRIC_OUTLIERS = TAB / "point2_beta_std_two_stage_three_metric_boxplots_by_class_outliers_removed.csv"
OUT_LONG = TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_r2.csv"
OUT_TABLE = TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_partial_r2_point2_style_by_year.csv"
OUT_N = TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_partial_r2_point2_style_n_exposures.csv"
METRIC_SPECS = {
    "partial_r2": {
        "table": OUT_TABLE,
    },
    "adjusted_partial_r2": {
        "table": TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_adjusted_partial_r2_point2_style_by_year.csv",
    },
    "incremental_r2": {
        "table": TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_incremental_r2_point2_style_by_year.csv",
    },
    "adjusted_incremental_r2": {
        "table": TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_adjusted_incremental_r2_point2_style_by_year.csv",
    },
}

KEY = ["state_alpha", "year", "month"]
YEARS = list(range(2000, 2026))
LB_TO_KG = 0.45359237
DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]


def retained_exposures() -> set[str]:
    """Variables retained in the mean |standardized beta| supplement boxplot."""
    if THREE_METRIC_VALUES.exists():
        values = pd.read_csv(THREE_METRIC_VALUES)
        paired = values.dropna(subset=["mean_abs_beta_2000_2014", "mean_abs_beta_2015_2024"]).copy()
        if THREE_METRIC_OUTLIERS.exists():
            outliers = pd.read_csv(THREE_METRIC_OUTLIERS)
            beta_outliers = set(outliers.loc[outliers["metric"].eq("mean_abs_beta_std"), "exposure"].dropna())
        else:
            beta_outliers = {"market_log_population_total", "storm_event_types"}
        return set(paired["exposure"]) - beta_outliers
    selected = pd.read_csv(SELECTION)
    selected = selected[selected["expanded_selection_status"].str.startswith("kept_expanded", na=False)].copy()
    return set(selected["exposure"]) - {"market_log_population_total", "storm_event_types"}


def residualize(m: np.ndarray, fe: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sw = np.sqrt(weights / np.nanmean(weights))
    mw = m * sw[:, None]
    few = fe * sw[:, None]
    coef, *_ = np.linalg.lstsq(few, mw, rcond=None)
    return mw - few @ coef


def prepare_panel() -> pd.DataFrame:
    panel = L.load_panel().copy()
    panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * LB_TO_KG
    panel["log_per_cow"] = np.log(panel["milk_per_cow_kg"].where(panel["milk_per_cow_kg"] > 0))
    return panel


def adjusted_r2(sse: float, sst: float, df_resid: int, df_total: int) -> float:
    if sst <= 1e-12 or df_resid <= 0 or df_total <= 0:
        return np.nan
    return 1.0 - (sse / df_resid) / (sst / df_total)


def fit_yearly_r2_metrics(panel: pd.DataFrame, x_col: str) -> list[dict]:
    needed = KEY + ["log_per_cow", x_col, "milk_cows_head"]
    d = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d[d["milk_cows_head"] > 0]
    if len(d) < 300 or d["state_alpha"].nunique() < 6 or d[x_col].nunique(dropna=True) <= 1:
        return [{"year": y, "status": "too_few"} for y in YEARS]

    y = d["log_per_cow"].to_numpy(float)
    x = L._standardize(d[x_col].to_numpy(float))
    year_arr = d["year"].to_numpy(int)
    x_year = np.column_stack([x * (year_arr == yv) for yv in YEARS])
    fe = pd.concat(
        [
            pd.Series(1.0, index=d.index, name="intercept"),
            pd.get_dummies(d["state_alpha"].astype(str), prefix="state", drop_first=True, dtype=float),
            pd.get_dummies(d["month"].astype(int), prefix="month", drop_first=True, dtype=float),
            pd.get_dummies(d["year"].astype(int), prefix="year", drop_first=True, dtype=float),
        ],
        axis=1,
    ).to_numpy(float)
    w = d["milk_cows_head"].to_numpy(float)
    resid = residualize(np.column_stack([y, x_year]), fe, w)
    y_r = resid[:, 0]
    design = resid[:, 1:]
    sst = float(np.sum(y_r**2))
    fe_rank = int(np.linalg.matrix_rank(fe))
    df_total = int(len(d) - fe_rank)
    x_cols = len(YEARS)
    keep_x = np.nanstd(design[:, :x_cols], axis=0) > 1e-10
    keep = keep_x
    if not keep_x.any():
        return [{"year": yv, "status": "collinear"} for yv in YEARS]

    xh = design[:, keep]
    kept_cols = np.where(keep)[0]
    beta, *_ = np.linalg.lstsq(xh, y_r, rcond=None)
    resid_full = y_r - xh @ beta
    sse_full = float(np.sum(resid_full**2))
    df_full = int(len(d) - fe_rank - xh.shape[1])
    r2_full = 1.0 - sse_full / sst if sst > 1e-12 else np.nan
    adj_r2_full = adjusted_r2(sse_full, sst, df_full, df_total)

    out = {
        yv: {
            "year": yv,
            "status": "collinear",
            "n": len(d),
            "n_states": d["state_alpha"].nunique(),
            "partial_r2": np.nan,
            "adjusted_partial_r2": np.nan,
            "incremental_r2": np.nan,
            "adjusted_incremental_r2": np.nan,
            "full_r2": r2_full,
            "full_adjusted_r2": adj_r2_full,
        }
        for yv in YEARS
    }
    for i, yv in enumerate(YEARS):
        if not keep_x[i]:
            continue
        pos = int(np.where(kept_cols == i)[0][0])
        reduced = np.delete(xh, pos, axis=1)
        beta_reduced, *_ = np.linalg.lstsq(reduced, y_r, rcond=None)
        resid_reduced = y_r - reduced @ beta_reduced
        sse_reduced = float(np.sum(resid_reduced**2))
        df_reduced = int(len(d) - fe_rank - reduced.shape[1])
        r2_reduced = 1.0 - sse_reduced / sst if sst > 1e-12 else np.nan
        adj_r2_reduced = adjusted_r2(sse_reduced, sst, df_reduced, df_total)
        partial = (sse_reduced - sse_full) / sse_reduced if sse_reduced > 1e-12 else np.nan
        incremental = r2_full - r2_reduced if np.isfinite(r2_full) and np.isfinite(r2_reduced) else np.nan
        adjusted_incremental = (
            adj_r2_full - adj_r2_reduced
            if np.isfinite(adj_r2_full) and np.isfinite(adj_r2_reduced)
            else np.nan
        )
        adjusted_partial = (
            1.0 - (sse_full / df_full) / (sse_reduced / df_reduced)
            if sse_reduced > 1e-12 and df_full > 0 and df_reduced > 0
            else np.nan
        )
        out[yv] = {
            "year": yv,
            "status": "ok",
            "n": int(len(d)),
            "n_states": int(d["state_alpha"].nunique()),
            "partial_r2": float(max(0.0, partial)) if np.isfinite(partial) else np.nan,
            "adjusted_partial_r2": float(adjusted_partial) if np.isfinite(adjusted_partial) else np.nan,
            "incremental_r2": float(max(0.0, incremental)) if np.isfinite(incremental) else np.nan,
            "adjusted_incremental_r2": float(adjusted_incremental) if np.isfinite(adjusted_incremental) else np.nan,
            "full_r2": float(r2_full) if np.isfinite(r2_full) else np.nan,
            "full_adjusted_r2": float(adj_r2_full) if np.isfinite(adj_r2_full) else np.nan,
            "reduced_r2": float(r2_reduced) if np.isfinite(r2_reduced) else np.nan,
            "reduced_adjusted_r2": float(adj_r2_reduced) if np.isfinite(adj_r2_reduced) else np.nan,
        }
    return [out[yv] for yv in YEARS]


def build() -> pd.DataFrame:
    selected = pd.read_csv(SELECTION)
    selected = selected[selected["expanded_selection_status"].str.startswith("kept_expanded", na=False)].copy()
    selected = selected[selected["exposure"].isin(retained_exposures())].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = prepare_panel()
    rows = []
    for _, r in selected.iterrows():
        x = r["exposure"]
        if x not in panel.columns:
            continue
        for fit in fit_yearly_r2_metrics(panel, x):
            rows.append(
                {
                    "outcome": "per_cow",
                    "outcome_label": "Milk per cow (kg)",
                    "domain_label": r["domain_label"],
                    "exposure": x,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = long[
        long["status"].eq("ok")
        & long["year"].between(2000, 2024)
        & long["domain_label"].isin(DOMAIN_ORDER)
    ].copy()
    summary = (
        d.groupby(["year", "domain_label"], as_index=False)
        .agg(median_metric=(metric, "median"))
        .pivot(index="year", columns="domain_label", values="median_metric")
        .reindex(index=range(2000, 2025), columns=DOMAIN_ORDER)
        .sort_index()
    )
    n_exp = (
        d.groupby(["year", "domain_label"], as_index=False)
        .agg(n_exposures=("exposure", "nunique"))
        .pivot(index="year", columns="domain_label", values="n_exposures")
        .reindex(index=range(2000, 2025), columns=DOMAIN_ORDER)
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    return summary, n_exp


def main() -> int:
    long = build()
    long.to_csv(OUT_LONG, index=False)
    summaries = {}
    n_exp = None
    for metric, spec in METRIC_SPECS.items():
        summary, metric_n_exp = summarize(long, metric)
        summaries[metric] = summary
        if n_exp is None:
            n_exp = metric_n_exp
        summary.reset_index(names="year").to_csv(spec["table"], index=False)
        print(f"Wrote {spec['table']}")
    n_exp.reset_index(names="year").to_csv(OUT_N, index=False)
    print(f"Wrote {OUT_LONG}")
    print(f"Wrote {OUT_N}")
    print("\nMedian R2-family table heads")
    for metric, summary in summaries.items():
        print(f"\n{metric}")
        print((summary.head() * 100).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
