#!/usr/bin/env python3
"""Yearly per-cow response trajectories for the all-clean Point 2 pool.

This is a sensitivity companion to the chord-signal yearly trajectory. It uses
the same year-varying-slope model as `30_build_chord_signal_yearly_associations.py`,
but fits every retained clean native variable. The resulting domain summaries
allow exposure-to-response translation to be compared for:

1. variables that enter the chord signal set; and
2. all clean retained native variables in the same domains.
"""

from __future__ import annotations

import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

TAB = POINT / "tables"
US_EXPOSE_NEW = ROOT / "data" / "us_expose_new" / "processed"
KEY = ["state_alpha", "year", "month"]
OUT_ROWS = TAB / "point2_all_clean_yearly_variable_associations.csv"
OUT_SUMMARY = TAB / "point2_all_clean_yearly_domain_summary.csv"
OUT_EXPOSURE_ALL_CLEAN = TAB / "point2_response_attenuation_domain_year_all_clean.csv"
OUT_EXPOSURE_CHORD = TAB / "point2_response_attenuation_domain_year.csv"

DOMAIN_RENAME = {"Herd structure / scale": "Dairy scale"}
DOMAINS = [
    "Heat",
    "Cold",
    "Severe weather",
    "Forage condition",
    "Agricultural pesticides",
    "Feed market",
    "Milk price / dairy market",
    "Market demand",
    "Dairy scale",
]


def canonical_domain(series: pd.Series) -> pd.Series:
    return series.replace(DOMAIN_RENAME)


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * 0.0


def _p_from_t(t: float, df: int) -> float:
    if not np.isfinite(t):
        return np.nan
    try:
        from scipy import stats

        return float(2 * stats.t.sf(abs(t), max(df, 1)))
    except ModuleNotFoundError:
        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def load_panel() -> pd.DataFrame:
    milk = pd.read_csv(L.US_MILK / "state_month_panel.csv", low_memory=False)
    milk["implied_milk_per_cow_lb"] = np.where(
        (pd.to_numeric(milk.get("milk_cows_head"), errors="coerce") > 0)
        & pd.to_numeric(milk.get("milk_production_lb"), errors="coerce").notna(),
        pd.to_numeric(milk.get("milk_production_lb"), errors="coerce")
        / pd.to_numeric(milk.get("milk_cows_head"), errors="coerce"),
        pd.to_numeric(milk.get("milk_per_cow_lb"), errors="coerce"),
    )
    exp = pd.read_csv(US_EXPOSE_NEW / "exposure_state_month_expanded.csv", low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    return milk.merge(exp, on=KEY, how="left")


def load_all_clean_variables() -> pd.DataFrame:
    assoc = pd.read_csv(TAB / "point2_native_only_endpoint_exwas_associations.csv", low_memory=False)
    assoc = assoc.copy()
    assoc["domain"] = canonical_domain(assoc["domain"])
    keep = (
        assoc["phenotype_scope"].eq("per_cow_26")
        & assoc["window"].eq("native")
        & assoc["domain"].isin(DOMAINS)
    )
    return (
        assoc.loc[keep, ["domain", "source_class", "exposure"]]
        .drop_duplicates()
        .sort_values(["domain", "exposure"])
        .reset_index(drop=True)
    )


def fit_year_interactions(panel: pd.DataFrame, years: list[int], x_col: str, y_col: str) -> list[dict]:
    if x_col not in panel.columns:
        return [{"year": y, "status": "missing"} for y in years]

    d = panel.loc[
        panel["year"].isin(years),
        ["state_alpha", "year", "month", "milk_cows_head", y_col, x_col],
    ].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["milk_cows_head"] > 0]
    if len(d) < 300 or d["state_alpha"].nunique() < 6 or d[x_col].nunique(dropna=True) <= 1:
        return [
            {"year": y, "status": "too_few", "n": len(d), "n_states": d["state_alpha"].nunique()}
            for y in years
        ]

    y_raw = d[y_col].to_numpy(float)
    y = np.log(y_raw) if np.nanmin(y_raw) > 0 else y_raw
    y = _standardize(y)
    x = _standardize(d[x_col].to_numpy(float))
    year_values = d["year"].to_numpy(int)
    xmat = np.column_stack([x * (year_values == yv) for yv in years])

    fe = pd.concat(
        [
            pd.Series(1.0, index=d.index, name="intercept"),
            pd.get_dummies(d["state_alpha"].astype(str), prefix="state", drop_first=True, dtype=float),
            pd.get_dummies(d["month"].astype(int), prefix="month", drop_first=True, dtype=float),
            pd.get_dummies(d["year"].astype(int), prefix="year", drop_first=True, dtype=float),
        ],
        axis=1,
    ).to_numpy()
    clusters = d["state_alpha"].to_numpy()
    w = d["milk_cows_head"].to_numpy(float)
    sw = np.sqrt(w / np.nanmean(w))

    y_r = L.absorb((y * sw).reshape(-1, 1), fe * sw[:, None]).ravel()
    x_r = L.absorb(xmat * sw[:, None], fe * sw[:, None])
    keep = np.nanstd(x_r, axis=0) > 1e-10
    if not keep.any():
        return [
            {"year": yv, "status": "collinear", "n": len(d), "n_states": d["state_alpha"].nunique()}
            for yv in years
        ]

    kept_years = [years[i] for i, ok in enumerate(keep) if ok]
    fit = L.cluster_robust_ols(y_r, x_r[:, keep], clusters, absorbed_params=fe.shape[1])
    out = {
        yv: {"year": yv, "status": "collinear", "n": len(d), "n_states": d["state_alpha"].nunique()}
        for yv in years
    }
    for i, yv in enumerate(kept_years):
        se = float(fit["se_cluster"][i])
        beta = float(fit["beta"][i])
        t = beta / se if se > 0 else np.nan
        out[yv] = {
            "year": yv,
            "status": "ok",
            "n": int(fit["n"]),
            "n_states": int(fit["n_clusters"]),
            "beta": beta,
            "se": se,
            "t": t,
            "p": _p_from_t(t, fit["n_clusters"] - 1),
        }
    return [out[yv] for yv in years]


def exposure_burden_summary(panel: pd.DataFrame, variables: pd.DataFrame, years: list[int], pool: str) -> pd.DataFrame:
    rows = []
    for _, r in variables.drop_duplicates(["domain", "exposure"]).iterrows():
        v = r["exposure"]
        if v not in panel.columns:
            continue
        x = pd.to_numeric(panel[v], errors="coerce")
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - x.mean()) / sd
        tmp = pd.DataFrame(
            {
                "domain": r["domain"],
                "year": panel["year"],
                "exposure": v,
                "exposure_z": z,
                "abs_exposure_z": z.abs(),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    d = pd.concat(rows, ignore_index=True)
    out = (
        d[d["year"].isin(years)]
        .groupby(["domain", "year"], dropna=False)
        .agg(
            n_exposure_variables=("exposure", "nunique"),
            n_state_months=("exposure_z", "size"),
            mean_abs_exposure_z=("abs_exposure_z", "mean"),
            mean_exposure_z=("exposure_z", "mean"),
            exposure_contrast_sd_z=("exposure_z", "std"),
            exposure_p90_abs_z=("abs_exposure_z", lambda x: float(np.nanpercentile(x, 90))),
        )
        .reset_index()
    )
    out["exposure_pool"] = pool
    return out


def main() -> int:
    panel = load_panel()
    variables = load_all_clean_variables()
    years = list(range(2000, 2026))

    rows = []
    phenotypes = [
        ("per_cow_26", "milk_per_cow_lb"),
        ("per_cow_50_implied", "implied_milk_per_cow_lb"),
    ]
    for scope, y_col in phenotypes:
        for _, var in variables.iterrows():
            for fit in fit_year_interactions(panel, years, var["exposure"], y_col):
                rows.append({**var.to_dict(), "phenotype_scope": scope, **fit})
    yearly = pd.DataFrame(rows)
    yearly.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")

    ok = yearly[yearly["status"].eq("ok")].copy()
    ok["neglogp"] = -np.log10(ok["p"].clip(lower=1e-300))
    ok["abs_beta"] = ok["beta"].abs()
    summary = (
        ok.groupby(["phenotype_scope", "source_class", "domain", "year"], dropna=False)
        .agg(
            n_variables=("exposure", "nunique"),
            median_beta=("beta", "median"),
            median_abs_beta=("abs_beta", "median"),
            mean_abs_beta=("abs_beta", "mean"),
            median_neglogp=("neglogp", "median"),
            max_neglogp=("neglogp", "max"),
            n_negative=("beta", lambda x: int((x < 0).sum())),
            n_positive=("beta", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    all_clean_exposure = exposure_burden_summary(
        panel,
        variables[["domain", "source_class", "exposure"]],
        years,
        "all_clean_native_variables",
    )
    all_clean_exposure.to_csv(OUT_EXPOSURE_ALL_CLEAN, index=False, encoding="utf-8-sig")

    chord_vars = pd.read_csv(TAB / "point2_chord_signal_yearly_variable_associations.csv", low_memory=False)
    chord_vars = chord_vars.copy()
    chord_vars["domain"] = canonical_domain(chord_vars["domain"])
    chord_vars = (
        chord_vars[chord_vars["domain"].isin(DOMAINS)][["domain", "source_class", "exposure"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    chord_exposure = exposure_burden_summary(
        panel,
        chord_vars,
        years,
        "chord_signal_variables",
    )
    chord_exposure.to_csv(OUT_EXPOSURE_CHORD, index=False, encoding="utf-8-sig")

    print(f"All-clean variables: {variables['exposure'].nunique()}")
    print(f"Yearly variable fits: {len(yearly)} rows; ok={int(yearly['status'].eq('ok').sum())}")
    print(f"Wrote {OUT_ROWS}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_EXPOSURE_ALL_CLEAN}")
    print(f"Wrote {OUT_EXPOSURE_CHORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
