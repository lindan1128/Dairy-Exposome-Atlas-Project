#!/usr/bin/env python3
"""Year-specific association trajectories for the Point 1 chord signal sets.

For each endpoint, domain, and year, this script refits the variables that enter
the paired chord plot using year-varying slopes:

    standardized log(milk endpoint) ~ state FE + month FE + year FE
                                      + standardized exposure x year

The output is summarized by domain so the figure can ask whether the signal
sets discovered in the full 2000-2025 ExWAS become stronger or weaker over
calendar time.
"""

from __future__ import annotations

import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

TAB = POINT / "tables"
KEY = ["state_alpha", "year", "month"]
US_EXPOSE_NEW = Path(__file__).resolve().parents[5] / "data" / "us_expose_new" / "processed"
OUT_ROWS = TAB / "point1_chord_signal_yearly_variable_associations.csv"
OUT_SUMMARY = TAB / "point1_chord_signal_yearly_domain_summary.csv"
OUT_AUDIT = TAB / "point1_chord_signal_yearly_domain_method_audit.csv"

EXCLUDED_DOMAINS = {
    "Drought",
    "Wildfire smoke",
    "Air pollution",
    "Industrial chemicals",
    "Production system context",
}
ENDPOINTS = {
    "total_26": "milk_production_lb_total26",
    "per_cow_26": "milk_per_cow_lb",
}


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
        # Normal fallback; used only if scipy is unavailable.
        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def load_panel() -> pd.DataFrame:
    milk = pd.read_csv(L.US_MILK / "state_month_panel.csv", low_memory=False)
    expanded_path = US_EXPOSE_NEW / "exposure_state_month_expanded.csv"
    exp = pd.read_csv(expanded_path, low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    panel = milk.merge(exp, on=KEY, how="left")

    per_cow_states = sorted(panel.loc[panel["milk_per_cow_lb"].notna(), "state_alpha"].dropna().unique())
    panel["milk_production_lb_total26"] = np.where(
        panel["state_alpha"].isin(per_cow_states), panel["milk_production_lb"], np.nan
    )
    return panel


def load_chord_signal_union() -> pd.DataFrame:
    assoc = pd.read_csv(TAB / "point1_native_only_endpoint_exwas_associations.csv", low_memory=False)
    # Use the clean dictionary values for modelling, but keep the plotting
    # domains consistent with the existing chord layout for event and dairy
    # market domains.
    assoc["domain"] = np.where(
        (assoc["domain"].eq("Pandemic shock")) & (assoc["mechanistic_domain_en"].eq("COVID")),
        "COVID",
        np.where(
            (assoc["domain"].eq("Pandemic shock")) & (assoc["mechanistic_domain_en"].eq("HPAI")),
            "HPAI",
            np.where(assoc["domain"].eq("Dairy market"), "Milk price / dairy market", assoc["domain"]),
        ),
    )
    assoc["plot_p"] = pd.to_numeric(assoc["plot_p"], errors="coerce")
    assoc["n_specs_same_sign"] = pd.to_numeric(assoc["n_specs_same_sign"], errors="coerce")
    keep = (
        assoc["window"].eq("native")
        & assoc["phenotype_scope"].isin(ENDPOINTS)
        & ~assoc["domain"].isin(EXCLUDED_DOMAINS)
        & (assoc["plot_p"] < 0.05)
        & (assoc["n_specs_same_sign"] >= 3)
        & assoc["effect_direction"].isin(["negative", "positive"])
    )
    endpoint_specific = assoc.loc[
        keep,
        [
            "phenotype_scope",
            "domain",
            "source_class",
            "exposure",
            "effect_direction",
            "native_signal_tier",
            "plot_p",
            "plot_incr_r2",
        ],
    ].drop_duplicates()
    # Use the union of variables that enter either endpoint's chord plot, then
    # refit that identical variable set for both endpoints. This makes the
    # endpoint comparison fair: differences reflect association strength, not
    # endpoint-specific feature selection.
    union = (
        endpoint_specific.groupby(["domain", "source_class", "exposure"], dropna=False)
        .agg(
            chord_endpoint_sources=("phenotype_scope", lambda x: ";".join(sorted(set(map(str, x))))),
            source_effect_directions=("effect_direction", lambda x: ";".join(sorted(set(map(str, x))))),
            best_source_plot_p=("plot_p", "min"),
            max_source_plot_incr_r2=("plot_incr_r2", "max"),
            source_signal_tiers=("native_signal_tier", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    rows = []
    for endpoint in ENDPOINTS:
        tmp = union.copy()
        tmp["phenotype_scope"] = endpoint
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def fit_year_interactions(panel: pd.DataFrame, years: list[int], y_col: str, x_col: str) -> list[dict]:
    if x_col not in panel.columns or y_col not in panel.columns:
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
    year_cols = []
    year_names = []
    for yv in years:
        col = x * (d["year"].to_numpy(int) == yv)
        year_cols.append(col)
        year_names.append(yv)
    xmat = np.column_stack(year_cols)

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
    y_w = y * sw
    x_w = xmat * sw[:, None]
    fe_w = fe * sw[:, None]

    y_r = L.absorb(y_w.reshape(-1, 1), fe_w).ravel()
    x_r = L.absorb(x_w, fe_w)
    keep = np.nanstd(x_r, axis=0) > 1e-10
    if not keep.any():
        return [
            {"year": yv, "status": "collinear", "n": len(d), "n_states": d["state_alpha"].nunique()}
            for yv in years
        ]
    x_fit = x_r[:, keep]
    kept_years = [year_names[i] for i, ok in enumerate(keep) if ok]
    # Refit after dropping collinear year-slope columns.
    fit = L.cluster_robust_ols(y_r, x_fit, clusters, absorbed_params=fe_w.shape[1])

    out = {
        yv: {"year": yv, "status": "collinear", "n": len(d), "n_states": d["state_alpha"].nunique()}
        for yv in years
    }
    for i, yv in enumerate(kept_years):
        se = float(fit["se_cluster"][i])
        beta = float(fit["beta"][i])
        t = beta / se if se > 0 else np.nan
        p = _p_from_t(t, fit["n_clusters"] - 1)
        out[yv] = {
            "year": yv,
            "status": "ok",
            "n": int(fit["n"]),
            "n_states": int(fit["n_clusters"]),
            "beta": beta,
            "se": se,
            "t": t,
            "p": p,
            "incr_r2": np.nan,
        }
    return [out[yv] for yv in years]


def main() -> int:
    panel = load_panel()
    signals = load_chord_signal_union()
    years = list(range(2000, 2026))

    rows = []
    for _, sig in signals.iterrows():
        y_col = ENDPOINTS[sig["phenotype_scope"]]
        fits = fit_year_interactions(panel, years, y_col, sig["exposure"])
        for fit in fits:
            row = {
                "phenotype_scope": sig["phenotype_scope"],
                "domain": sig["domain"],
                "source_class": sig["source_class"],
                "exposure": sig["exposure"],
                "chord_endpoint_sources": sig["chord_endpoint_sources"],
                "source_effect_directions": sig["source_effect_directions"],
                "source_signal_tiers": sig["source_signal_tiers"],
                "best_source_plot_p": sig["best_source_plot_p"],
                "max_source_plot_incr_r2": sig["max_source_plot_incr_r2"],
                **fit,
            }
            rows.append(row)
    yearly = pd.DataFrame(rows)
    yearly.to_csv(OUT_ROWS, index=False, encoding="utf-8-sig")

    ok = yearly[yearly["status"].eq("ok")].copy()
    ok["neglogp"] = -np.log10(ok["p"].clip(lower=1e-300))
    ok["abs_beta"] = ok["beta"].abs()
    ok["signed_abs_beta"] = np.where(ok["beta"] < 0, -ok["abs_beta"], ok["abs_beta"])

    summary = (
        ok.groupby(["phenotype_scope", "source_class", "domain", "year"], dropna=False)
        .agg(
            n_signal_variables=("exposure", "nunique"),
            median_beta=("beta", "median"),
            median_abs_beta=("abs_beta", "median"),
            mean_abs_beta=("abs_beta", "mean"),
            median_neglogp=("neglogp", "median"),
            max_neglogp=("neglogp", "max"),
            median_incr_r2=("incr_r2", "median"),
            mean_incr_r2=("incr_r2", "mean"),
            n_negative=("beta", lambda x: int((x < 0).sum())),
            n_positive=("beta", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    summary["net_negative_share"] = (
        (summary["n_negative"] - summary["n_positive"]) / summary["n_signal_variables"]
    )
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    audit_base = (
        yearly.groupby(["phenotype_scope", "source_class", "domain"], dropna=False)
        .agg(
            n_signal_variables=("exposure", "nunique"),
            n_year_rows=("year", "size"),
            n_ok_year_rows=("status", lambda x: int((x == "ok").sum())),
            n_collinear_year_rows=("status", lambda x: int((x == "collinear").sum())),
            n_too_few_year_rows=("status", lambda x: int((x == "too_few").sum())),
        )
        .reset_index()
    )
    ok_years = (
        yearly.loc[yearly["status"].eq("ok")]
        .groupby(["phenotype_scope", "domain"], dropna=False)
        .agg(first_ok_year=("year", "min"), last_ok_year=("year", "max"), n_ok_years=("year", "nunique"))
        .reset_index()
    )
    audit = audit_base.merge(ok_years, on=["phenotype_scope", "domain"], how="left")

    def method_note(row: pd.Series) -> pd.Series:
        domain = row["domain"]
        n_years = row.get("n_ok_years", 0)
        first = row.get("first_ok_year", np.nan)
        last = row.get("last_ok_year", np.nan)
        if domain in {"COVID", "HPAI"}:
            model_class = "event shock"
            suitable = "No"
            interpretation = "Use event-study effect; do not interpret as a 2000-2025 association trajectory."
        elif n_years >= 20:
            model_class = "longitudinal exposure/context"
            suitable = "Yes"
            interpretation = "Suitable for a descriptive year-varying association trajectory within the observed window."
        elif n_years >= 10:
            model_class = "partial-window background/context"
            suitable = "Partial"
            interpretation = (
                f"Interpret only for observed years {int(first)}-{int(last)}; not a full 2000-2025 trend."
                if np.isfinite(first) and np.isfinite(last)
                else "Interpret only for observed years; not a full 2000-2025 trend."
            )
        else:
            model_class = "sparse/short-window"
            suitable = "No"
            interpretation = "Too sparse for yearly trend interpretation; keep as endpoint/domain signal only."
        return pd.Series(
            {
                "model_class": model_class,
                "suitable_for_2000_2025_yearly_trend": suitable,
                "interpretation_note": interpretation,
            }
        )

    audit = pd.concat([audit, audit.apply(method_note, axis=1)], axis=1)
    audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")

    print(f"Chord signal rows: {len(signals)}")
    print(f"Yearly variable fits: {len(yearly)} rows; ok={int(yearly['status'].eq('ok').sum())}")
    print(f"Wrote {OUT_ROWS}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_AUDIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
