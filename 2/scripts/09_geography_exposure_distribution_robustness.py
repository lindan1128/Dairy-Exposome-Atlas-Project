#!/usr/bin/env python3
"""Robustness checks for geographic redistribution and exposure distribution shifts."""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
TAB = POINT / "tables"
FIG = POINT / "figures"
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

DECOMP = TAB / "point2_total_insensitivity_log_decomposition.csv"
CANDIDATES = TAB / "point2_common_sense_directional_variable_selection.csv"
OUT_YEARLY = TAB / "point2_geo_exposure_robustness_yearly_sensitivity.csv"
OUT_DOMAIN = TAB / "point2_geo_exposure_robustness_domain_medians.csv"
OUT_TREND = TAB / "point2_geo_exposure_robustness_trend_summary.csv"
OUT_EXPOSURE_DIAG = TAB / "point2_exposure_distribution_shift_diagnostics.csv"
OUT_EXPOSURE_DIAG_TREND = TAB / "point2_exposure_distribution_shift_trends.csv"

KEY = ["state_alpha", "year", "month"]
OUTCOMES = {"per_cow": "Milk per cow (kg)"}
LB_TO_KG = 0.45359237
DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
COLORS = {
    "baseline_dynamic_cow_weight": "#2f6f73",
    "state_equal_weight": "#b64b3a",
    "fixed_2000_cow_weight": "#6f6a8f",
    "region_year_fe": "#d9995b",
    "year_specific_exposure_z": "#4f78a8",
}
VARIANT_LABELS = {
    "baseline_dynamic_cow_weight": "Baseline cow weight",
    "state_equal_weight": "State equal",
    "fixed_2000_cow_weight": "Fixed 2000 cow weight",
    "region_year_fe": "Region-year FE",
    "year_specific_exposure_z": "Year-specific exposure z",
}
VARIANTS = [
    "baseline_dynamic_cow_weight",
    "state_equal_weight",
    "fixed_2000_cow_weight",
    "region_year_fe",
    "year_specific_exposure_z",
]


def p_from_t(t: float, df: int) -> float:
    if not np.isfinite(t):
        return np.nan
    try:
        from scipy import stats

        return float(2 * stats.t.sf(abs(t), max(df, 1)))
    except ModuleNotFoundError:
        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x, ddof=0))
    return (x - float(np.nanmean(x))) / sd if np.isfinite(sd) and sd > 1e-12 else np.zeros_like(x)


def zscore_by_year(d: pd.DataFrame, col: str) -> np.ndarray:
    return d.groupby("year", group_keys=False)[col].transform(lambda s: zscore(s.to_numpy(float))).to_numpy(float)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    vals = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, vals))


def residualize(m: np.ndarray, fe: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is not None:
        sw = np.sqrt(weights / np.nanmean(weights))
        m = m * sw[:, None]
        fe = fe * sw[:, None]
    coef, *_ = np.linalg.lstsq(fe, m, rcond=None)
    return m - fe @ coef


def prepare_panel() -> pd.DataFrame:
    panel = L.load_panel().copy()
    panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * LB_TO_KG
    panel["log_per_cow"] = np.log(panel["milk_per_cow_kg"].where(panel["milk_per_cow_kg"] > 0))
    panel["log_cows"] = np.log(panel["milk_cows_head"].where(panel["milk_cows_head"] > 0))
    base_w = (
        panel.loc[panel.year.eq(2000)]
        .groupby("state_alpha")["milk_cows_head"]
        .mean()
        .rename("fixed_2000_cow_weight")
    )
    panel = panel.merge(base_w, on="state_alpha", how="left")
    panel["region_year"] = panel["region"].astype(str) + "_" + panel["year"].astype(int).astype(str)
    return panel


def fixed_effects(d: pd.DataFrame, variant: str) -> np.ndarray:
    pieces = [
        pd.Series(1.0, index=d.index, name="intercept"),
        pd.get_dummies(d["state_alpha"].astype(str), prefix="state", drop_first=True, dtype=float),
        pd.get_dummies(d["month"].astype(int), prefix="month", drop_first=True, dtype=float),
    ]
    if variant == "region_year_fe":
        pieces.append(pd.get_dummies(d["region_year"].astype(str), prefix="region_year", drop_first=True, dtype=float))
    else:
        pieces.append(pd.get_dummies(d["year"].astype(int), prefix="year", drop_first=True, dtype=float))
    return pd.concat(pieces, axis=1).to_numpy(float)


def weights(d: pd.DataFrame, variant: str) -> np.ndarray | None:
    if variant in {"baseline_dynamic_cow_weight", "region_year_fe", "year_specific_exposure_z"}:
        return d["milk_cows_head"].to_numpy(float)
    if variant == "state_equal_weight":
        return None
    if variant == "fixed_2000_cow_weight":
        return d["fixed_2000_cow_weight"].to_numpy(float)
    raise ValueError(variant)


def fit_yearly(panel: pd.DataFrame, y_col: str, x_col: str, variant: str) -> list[dict]:
    needed = KEY + ["region", "region_year", y_col, x_col, "milk_cows_head", "fixed_2000_cow_weight"]
    d = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d[d["milk_cows_head"] > 0]
    years = list(range(2000, 2026))
    if len(d) < 300 or d["state_alpha"].nunique() < 6 or d[x_col].nunique(dropna=True) <= 1:
        return [{"year": y, "status": "too_few"} for y in years]
    y = d[y_col].to_numpy(float)
    x = zscore_by_year(d, x_col) if variant == "year_specific_exposure_z" else zscore(d[x_col].to_numpy(float))
    x_year = np.column_stack([x * (d["year"].to_numpy(int) == yv) for yv in years])
    fe = fixed_effects(d, variant)
    w = weights(d, variant)
    resid = residualize(np.column_stack([y, x_year]), fe, w)
    y_r = resid[:, 0]
    x_r = resid[:, 1:]
    keep = np.nanstd(x_r, axis=0) > 1e-10
    kept_years = [years[i] for i, ok in enumerate(keep) if ok]
    if not kept_years:
        return [{"year": yv, "status": "collinear"} for yv in years]
    fit = L.cluster_robust_ols(y_r, x_r[:, keep], d["state_alpha"].to_numpy(), absorbed_params=fe.shape[1])
    out = {yv: {"year": yv, "status": "collinear", "n": len(d), "n_states": d["state_alpha"].nunique()} for yv in years}
    for i, yv in enumerate(kept_years):
        beta = float(fit["beta"][i])
        se = float(fit["se_cluster"][i])
        out[yv] = {
            "year": yv,
            "status": "ok",
            "n": int(fit["n"]),
            "n_states": int(fit["n_clusters"]),
            "beta_log_per_1sd_exposure": beta,
            "se": se,
            "p": p_from_t(beta / se if se > 0 else np.nan, fit["n_clusters"] - 1),
        }
    return [out[yv] for yv in years]


def selected_exposures() -> pd.DataFrame:
    if DECOMP.exists():
        dec = pd.read_csv(DECOMP)
        dec["domain_label"] = dec["domain_plot"].replace({"Forage condition": "Forage"})
        return dec[
            dec["domain_label"].isin(DOMAIN_ORDER)
            & ((dec["total_p"] < 0.05) | (dec["per_cow_p"] < 0.05) | (dec["cows_p"] < 0.05))
        ].copy()
    keep = pd.read_csv(CANDIDATES)
    keep = keep[keep["domain_label"].isin(DOMAIN_ORDER)].copy()
    keep["domain_plot"] = keep["domain_label"]
    keep["source_class"] = "candidate signal pool"
    return keep


def build_yearly(panel: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        print(f"variant {variant}", flush=True)
        for _, r in exposures.iterrows():
            x = r["exposure"]
            if x not in panel.columns:
                continue
            for outcome, label in OUTCOMES.items():
                for fit in fit_yearly(panel, f"log_{outcome}", x, variant):
                    rows.append(
                        {
                            "variant": variant,
                            "variant_label": VARIANT_LABELS[variant],
                            "outcome": outcome,
                            "outcome_label": label,
                            "domain_label": r["domain_label"],
                            "domain_plot": r["domain_plot"],
                            "exposure": x,
                            **fit,
                        }
                    )
    return pd.DataFrame(rows)


def domain_medians(yearly: pd.DataFrame) -> pd.DataFrame:
    ok = yearly[yearly["status"].eq("ok")].copy()
    ok["abs_beta_log"] = ok["beta_log_per_1sd_exposure"].abs()
    return (
        ok.groupby(["variant", "variant_label", "outcome", "outcome_label", "domain_label", "year"], as_index=False)
        .agg(
            median_abs_beta=("abs_beta_log", "median"),
            mean_abs_beta=("abs_beta_log", "mean"),
            n_exposures=("exposure", "nunique"),
        )
        .sort_values(["variant", "outcome", "domain_label", "year"])
    )


def trend_summary(domain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, s in domain.groupby(["variant", "variant_label", "outcome", "outcome_label", "domain_label"]):
        s = s.sort_values("year")
        if s["year"].nunique() < 8:
            continue
        x = s["year"].to_numpy(float)
        y = s["median_abs_beta"].to_numpy(float)
        coef = np.polyfit(x, y, 1)
        yhat = coef[0] * x + coef[1]
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        se = np.sqrt(ss_res / max(len(x) - 2, 1) / float(((x - x.mean()) ** 2).sum()))
        rows.append(
            {
                "variant": keys[0],
                "variant_label": keys[1],
                "outcome": keys[2],
                "outcome_label": keys[3],
                "domain_label": keys[4],
                "n_years": s["year"].nunique(),
                "mean_annual_median_abs_beta": float(y.mean()),
                "slope_median_abs_beta_per_year": float(coef[0]),
                "trend_r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                "trend_p": p_from_t(coef[0] / se if se > 0 else np.nan, len(x) - 2),
                "fitted_median_abs_beta_2000": float(yhat[0]),
                "fitted_median_abs_beta_2025": float(yhat[-1]),
                "percent_change_2000_to_2025": float((yhat[-1] / yhat[0] - 1) * 100) if yhat[0] != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def exposure_distribution_diagnostics(panel: pd.DataFrame, exposures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in exposures.iterrows():
        x = r["exposure"]
        if x not in panel.columns:
            continue
        for year, s in panel[[x, "year", "milk_cows_head", "state_alpha"]].replace([np.inf, -np.inf], np.nan).dropna().groupby("year"):
            vals = s[x].to_numpy(float)
            w = s["milk_cows_head"].to_numpy(float)
            rows.append(
                {
                    "exposure": x,
                    "domain_label": r["domain_label"],
                    "year": int(year),
                    "n_rows": len(s),
                    "n_states": s["state_alpha"].nunique(),
                    "unweighted_mean": float(np.mean(vals)),
                    "unweighted_sd": float(np.std(vals, ddof=0)),
                    "cow_weighted_mean": float(np.average(vals, weights=w)),
                    "cow_weighted_sd": float(np.sqrt(np.average((vals - np.average(vals, weights=w)) ** 2, weights=w))),
                    "cow_weighted_p05": weighted_quantile(vals, w, 0.05),
                    "cow_weighted_p95": weighted_quantile(vals, w, 0.95),
                }
            )
    annual = pd.DataFrame(rows)
    trend_rows = []
    for keys, s in annual.groupby(["exposure", "domain_label"]):
        for metric in ["unweighted_sd", "cow_weighted_sd", "cow_weighted_mean"]:
            d = s[["year", metric]].dropna().sort_values("year")
            if d["year"].nunique() < 8:
                continue
            x = d["year"].to_numpy(float)
            y = d[metric].to_numpy(float)
            coef = np.polyfit(x, y, 1)
            yhat = coef[0] * x + coef[1]
            trend_rows.append(
                {
                    "exposure": keys[0],
                    "domain_label": keys[1],
                    "metric": metric,
                    "slope_per_year": float(coef[0]),
                    "fitted_2000": float(yhat[0]),
                    "fitted_2025": float(yhat[-1]),
                    "percent_change_2000_to_2025": float((yhat[-1] / yhat[0] - 1) * 100) if yhat[0] != 0 else np.nan,
                }
            )
    return annual, pd.DataFrame(trend_rows)


def plot_variant_comparison(domain: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    s = (
        domain[domain["outcome"].eq("per_cow")]
        .groupby(["variant", "variant_label", "year"], as_index=False)
        .agg(median_across_domains=("median_abs_beta", "median"))
    )
    for variant in VARIANTS:
        d = s[s["variant"].eq(variant)].sort_values("year")
        ax.plot(d["year"], d["median_across_domains"] * 100, lw=1.9, color=COLORS[variant], label=VARIANT_LABELS[variant])
    ax.set_title("Milk per cow")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median domain |log beta| (%)")
    ax.set_xlim(2000, 2025)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    plt.savefig(FIG / "point2_geo_exposure_robustness_variant_comparison.svg", bbox_inches="tight")
    plt.close()


def plot_exposure_sd_diagnostics(trends: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    s = trends[trends["metric"].eq("cow_weighted_sd")].copy()
    summary = (
        s.groupby("domain_label", as_index=False)
        .agg(median_percent_change=("percent_change_2000_to_2025", "median"), n_exposures=("exposure", "nunique"))
        .sort_values("median_percent_change")
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    colors = ["#b64b3a" if v < 0 else "#2f6f73" for v in summary["median_percent_change"]]
    ax.barh(summary["domain_label"], summary["median_percent_change"], color=colors)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Median fitted change in cow-weighted exposure SD, 2000-2025 (%)")
    ax.set_ylabel("")
    ax.set_title("Did exposure distributions narrow or widen over time?")
    ax.spines[["top", "right"]].set_visible(False)
    plt.savefig(FIG / "point2_exposure_distribution_sd_change_by_domain.svg", bbox_inches="tight")
    plt.close()


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    exposures = selected_exposures()
    yearly = build_yearly(panel, exposures)
    yearly.to_csv(OUT_YEARLY, index=False)
    domain = domain_medians(yearly)
    domain.to_csv(OUT_DOMAIN, index=False)
    trends = trend_summary(domain)
    trends.to_csv(OUT_TREND, index=False)
    exposure_diag, exposure_diag_trends = exposure_distribution_diagnostics(panel, exposures)
    exposure_diag.to_csv(OUT_EXPOSURE_DIAG, index=False)
    exposure_diag_trends.to_csv(OUT_EXPOSURE_DIAG_TREND, index=False)
    plot_variant_comparison(domain)
    plot_exposure_sd_diagnostics(exposure_diag_trends)
    print(f"Wrote {OUT_YEARLY}")
    print(f"Wrote {OUT_DOMAIN}")
    print(f"Wrote {OUT_TREND}")
    print(f"Wrote {OUT_EXPOSURE_DIAG}")
    print(f"Wrote {OUT_EXPOSURE_DIAG_TREND}")
    print(
        trends.groupby(["variant_label", "outcome_label"])
        .agg(
            n_domains=("domain_label", "nunique"),
            median_slope=("slope_median_abs_beta_per_year", "median"),
            n_declining=("slope_median_abs_beta_per_year", lambda x: int((x < 0).sum())),
            median_pct=("percent_change_2000_to_2025", "median"),
        )
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
