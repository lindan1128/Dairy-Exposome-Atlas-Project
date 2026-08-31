#!/usr/bin/env python3
"""Yearly exposure-association trends for milk production per cow."""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path
import re
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
OUT_YEARLY = TAB / "point2_herd_adjusted_yearly_sensitivity.csv"
OUT_DOMAIN = TAB / "point2_herd_adjusted_yearly_domain_medians.csv"
OUT_TREND = TAB / "point2_herd_adjusted_yearly_trend_summary.csv"

KEY = ["state_alpha", "year", "month"]
OUTCOMES = {"per_cow": "Milk per cow (kg)"}
LB_TO_KG = 0.45359237
DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
COLORS = {
    "Heat": "#32a4b4",
    "Cold": "#33c5b2",
    "Severe weather": "#d5eada",
    "Forage": "#1E7A8D",
    "Feed market": "#fbc4ab",
    "Dairy market": "#E47666",
    "Market demand": "#f09d51",
}


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "svg.fonttype": "none",
    }
)


def p_from_t(t: float, df: int) -> float:
    if not np.isfinite(t):
        return np.nan
    try:
        from scipy import stats

        return float(2 * stats.t.sf(abs(t), max(df, 1)))
    except ModuleNotFoundError:
        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


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


def fit_yearly_herd_adjusted(panel: pd.DataFrame, y_col: str, x_col: str) -> list[dict]:
    needed = KEY + [y_col, x_col, "milk_cows_head"]
    d = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d[d["milk_cows_head"] > 0]
    years = list(range(2000, 2026))
    if len(d) < 300 or d["state_alpha"].nunique() < 6 or d[x_col].nunique(dropna=True) <= 1:
        return [{"year": y, "status": "too_few"} for y in years]

    y = d[y_col].to_numpy(float)
    x = L._standardize(d[x_col].to_numpy(float))
    year_arr = d["year"].to_numpy(int)
    x_year = np.column_stack([x * (year_arr == yv) for yv in years])
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
    xh_r = resid[:, 1:]
    x_cols = len(years)
    keep_x = np.nanstd(xh_r[:, :x_cols], axis=0) > 1e-10
    keep = keep_x
    kept_x_years = [years[i] for i, ok in enumerate(keep_x) if ok]
    if not kept_x_years:
        return [{"year": yv, "status": "collinear"} for yv in years]

    fit = L.cluster_robust_ols(y_r, xh_r[:, keep], d["state_alpha"].to_numpy(), absorbed_params=fe.shape[1])
    kept_cols = np.where(keep)[0]
    x_position_by_year = {years[i]: int(np.where(kept_cols == i)[0][0]) for i, ok in enumerate(keep_x) if ok}
    out = {yv: {"year": yv, "status": "collinear", "n": len(d), "n_states": d["state_alpha"].nunique()} for yv in years}
    for yv, pos in x_position_by_year.items():
        beta = float(fit["beta"][pos])
        se = float(fit["se_cluster"][pos])
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


def build_yearly(panel: pd.DataFrame) -> pd.DataFrame:
    if DECOMP.exists():
        dec = pd.read_csv(DECOMP)
        dec["domain_label"] = dec["domain_plot"].replace({"Forage condition": "Forage"})
        keep = dec[
            dec["domain_label"].isin(DOMAIN_ORDER)
            & ((dec["total_p"] < 0.05) | (dec["per_cow_p"] < 0.05) | (dec["cows_p"] < 0.05))
        ].copy()
    else:
        keep = pd.read_csv(CANDIDATES)
        keep = keep[keep["domain_label"].isin(DOMAIN_ORDER)].copy()
        keep["domain_plot"] = keep["domain_label"]
        keep["source_class"] = "candidate signal pool"
    rows = []
    for _, r in keep.iterrows():
        x = r["exposure"]
        if x not in panel.columns:
            continue
        for outcome, label in OUTCOMES.items():
            for fit in fit_yearly_herd_adjusted(panel, f"log_{outcome}", x):
                rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": label,
                        "domain_plot": r["domain_plot"],
                        "domain_label": r["domain_label"],
                        "source_class": r["source_class"],
                        "exposure": x,
                        **fit,
                    }
                )
    return pd.DataFrame(rows)


def domain_medians(yearly: pd.DataFrame) -> pd.DataFrame:
    ok = yearly[yearly["status"].eq("ok")].copy()
    ok["abs_beta_log"] = ok["beta_log_per_1sd_exposure"].abs()
    return (
        ok.groupby(["outcome", "outcome_label", "domain_plot", "domain_label", "year"], as_index=False)
        .agg(
            median_abs_beta=("abs_beta_log", "median"),
            mean_abs_beta=("abs_beta_log", "mean"),
            n_exposures=("exposure", "nunique"),
        )
        .sort_values(["outcome", "domain_label", "year"])
    )


def trend_summary(domain: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, s in domain.groupby(["outcome", "outcome_label", "domain_label"], dropna=False):
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
                "outcome": keys[0],
                "outcome_label": keys[1],
                "domain_label": keys[2],
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


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    svg = svg.replace("font-size: 9px;", "font-size: 9.00px;")
    path.write_text(svg, encoding="utf-8")


def point2_style_table(domain: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = domain.loc[domain["outcome"].eq(outcome) & domain["year"].between(2000, 2024)].copy()
    summary = (
        d.pivot(index="year", columns="domain_label", values="median_abs_beta")
        .reindex(columns=DOMAIN_ORDER)
        .sort_index()
    )
    expected_years = list(range(2000, 2025))
    if summary.index.tolist() != expected_years or summary.isna().any().any():
        raise RuntimeError(f"Incomplete point2-style herd-adjusted table for {outcome}")
    return summary


def make_point2_style_plot(summary: pd.DataFrame, outcome: str, show_legend: bool) -> None:
    height = 3.9 if show_legend else 3.45
    fig, ax = plt.subplots(figsize=(8.25, height), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plot_values = summary * 100
    years = plot_values.index.to_numpy()
    base = pd.Series(0.0, index=plot_values.index)
    for domain in reversed(DOMAIN_ORDER):
        positive = plot_values[domain].clip(lower=0)
        ax.fill_between(years, base.to_numpy(), (base + positive).to_numpy(), label=domain, color=COLORS[domain], linewidth=0)
        base += positive
    ax.set_xlim(2000, 2025)
    ax.set_ylim(0, float(base.max()) * 1.08)
    ax.axhline(0, color="#444444", linewidth=0.55)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.set_xticklabels(["2000", "2005", "2010", "2015", "2020", "2025"])
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Stacked median |β| (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles))
        ax.legend(
            [handle_by_label[domain] for domain in DOMAIN_ORDER],
            DOMAIN_ORDER,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            frameon=False,
            fontsize=9,
            columnspacing=1.4,
            handlelength=1.5,
        )
    stem = f"point2_{outcome}_herd_adjusted_yearly_point2_style"
    suffix = "" if show_legend else "_wo_legend"
    svg = FIG / f"{stem}{suffix}.svg"
    fig.savefig(svg, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    normalize_svg_text_style(svg)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    yearly = build_yearly(panel)
    yearly.to_csv(OUT_YEARLY, index=False)
    domain = domain_medians(yearly)
    domain.to_csv(OUT_DOMAIN, index=False)
    trends = trend_summary(domain)
    trends.to_csv(OUT_TREND, index=False)
    for outcome in OUTCOMES:
        summary = point2_style_table(domain, outcome)
        summary.reset_index().to_csv(TAB / f"point2_{outcome}_herd_adjusted_yearly_sensitivity_point2_style_by_year.csv", index=False)
        make_point2_style_plot(summary, outcome, show_legend=True)
        make_point2_style_plot(summary, outcome, show_legend=False)
    print(f"Wrote {OUT_YEARLY}")
    print(f"Wrote {OUT_DOMAIN}")
    print(f"Wrote {OUT_TREND}")
    print(trends.groupby("outcome_label").agg(n_domains=("domain_label", "nunique"), median_slope=("slope_median_abs_beta_per_year", "median"), n_declining=("slope_median_abs_beta_per_year", lambda x: int((x < 0).sum()))).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
