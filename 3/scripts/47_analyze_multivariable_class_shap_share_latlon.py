#!/usr/bin/env python3
"""Latitude/longitude gradients in state-level domain SHAP shares."""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parents[1]
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
CENTERS = TABLES / "point3_us_state_geographic_centers.csv"

DOMAINS = (
    "Heat",
    "Cold",
    "Severe weather",
    "Forage",
    "Feed market",
    "Dairy market",
    "Market demand",
)
DOMAIN_COLORS = {
    "Heat": "#32a4b4",
    "Cold": "#33c5b2",
    "Severe weather": "#d5eada",
    "Forage": "#1E7A8D",
    "Feed market": "#fbc4ab",
    "Dairy market": "#E47666",
    "Market demand": "#f09d51",
}


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    path.write_text(svg, encoding="utf-8")


def read_domain_shap() -> pd.DataFrame:
    blocks = []
    for horizon in range(1, 10):
        path = TABLES / f"point3_point4aligned_h{horizon}_shap_state_month_class.csv"
        data = pd.read_csv(path)
        data = data.loc[data.class_label.isin(DOMAINS)].copy()
        data["horizon_months"] = horizon
        blocks.append(data)
    return pd.concat(blocks, ignore_index=True)


def state_class_share(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["state_alpha", "region", "year", "month", "horizon_months"]
    data = raw.copy()
    data["class_abs_shap"] = data["class_abs_signed_shap"].astype(float)
    total = (
        data.groupby(keys, as_index=False)
        .agg(total_abs_shap=("class_abs_shap", "sum"))
    )
    data = data.merge(total, on=keys, how="left", validate="many_to_one")
    data["class_share"] = np.where(
        data.total_abs_shap > 0,
        data.class_abs_shap / data.total_abs_shap,
        np.nan,
    )
    state_domain = (
        data.groupby(["state_alpha", "region", "class_label"], as_index=False)
        .agg(
            median_class_share=("class_share", "median"),
            mean_class_share=("class_share", "mean"),
            n_state_month_horizon=("class_share", "count"),
        )
    )
    centers = pd.read_csv(CENTERS)
    out = state_domain.merge(centers, on="state_alpha", how="left", validate="many_to_one")
    if out[["longitude", "latitude"]].isna().any().any():
        missing = out.loc[out.latitude.isna(), "state_alpha"].unique().tolist()
        raise RuntimeError(f"Missing geographic centers for: {missing}")
    return out


def fit_latlon(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        data = values.loc[values.class_label.eq(domain)].copy()
        for coordinate in ("latitude", "longitude"):
            clean = data.dropna(subset=[coordinate, "mean_class_share"])
            if len(clean) < 3 or clean.mean_class_share.nunique() <= 1:
                fit = None
            else:
                fit = stats.linregress(
                    clean[coordinate].to_numpy(float),
                    clean.mean_class_share.to_numpy(float),
                )
            rows.append(
                {
                    "class_label": domain,
                    "coordinate": coordinate,
                    "n_states": len(clean),
                    "slope_share_per_degree": fit.slope if fit is not None else np.nan,
                    "intercept": fit.intercept if fit is not None else np.nan,
                    "r": fit.rvalue if fit is not None else np.nan,
                    "r2": fit.rvalue**2 if fit is not None else np.nan,
                    "p": fit.pvalue if fit is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot(values: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
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
    line_rows = []
    for coordinate, coordinate_label in (("latitude", "Latitude (°)"), ("longitude", "Longitude (°)")):
        active = fits.loc[fits.coordinate.eq(coordinate) & fits.p.lt(0.05)].copy()
        domains = [domain for domain in DOMAINS if domain in set(active.class_label)]
        data = values.loc[values.class_label.isin(domains)].copy()
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=DOMAIN_COLORS[domain],
                markeredgecolor="#222222",
                markeredgewidth=0.45,
                markersize=5.5,
                label=domain,
            )
            for domain in domains
        ]
        for show_legend in (True, False):
            fig, axis = plt.subplots(figsize=(3.8, 4.0))
            for domain in domains:
                group = data.loc[data.class_label.eq(domain)].sort_values(coordinate)
                x = group[coordinate].to_numpy(float)
                y = group.mean_class_share.to_numpy(float) * 100.0
                fit = stats.linregress(x, y)
                if not show_legend:
                    line_rows.append(
                        {
                            "coordinate": coordinate,
                            "class_label": domain,
                            "n_points": len(group),
                            "slope_share_pct_per_degree": fit.slope,
                            "intercept": fit.intercept,
                            "r": fit.rvalue,
                            "r2": fit.rvalue**2,
                            "p": fit.pvalue,
                        }
                    )
                axis.scatter(
                    x,
                    y,
                    s=24,
                    marker="o",
                    color=DOMAIN_COLORS[domain],
                    alpha=1.0,
                    edgecolors="#222222",
                    linewidths=0.45,
                    zorder=2,
                )
                axis.plot(
                    x,
                    fit.intercept + fit.slope * x,
                    color=DOMAIN_COLORS[domain],
                    linestyle="solid",
                    linewidth=3.0,
                    alpha=1.0,
                    zorder=1,
                )
            axis.set_xlabel(coordinate_label)
            axis.set_ylabel("Mean class SHAP (%)")
            axis.grid(axis="both", color="#d9d9d9", linestyle=":", linewidth=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(axis="both", labelsize=9, length=3, width=0.7)
            if not domains:
                axis.text(
                    0.5,
                    0.5,
                    "No domain-share association with OLS p < 0.05",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#222222",
                )
            if show_legend and domains:
                fig.legend(
                    handles,
                    domains,
                    ncol=len(domains),
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    frameon=False,
                    fontsize=9,
                    columnspacing=0.9,
                    handletextpad=0.35,
                )
                fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.82)
            else:
                fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.96)
            suffix = "" if show_legend else "_wo_legend"
            path = FIGURES / f"main_point3_multivariable_class_share_{coordinate}_scatter{suffix}.svg"
            fig.savefig(path, dpi=300, transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight")
            normalize_svg_text_style(path)
            plt.close(fig)
    return pd.DataFrame(line_rows)


def main() -> None:
    raw = read_domain_shap()
    values = state_class_share(raw)
    fits = fit_latlon(values)
    lines = plot(values, fits)
    values.to_csv(TABLES / "main_point3_multivariable_class_share_state.csv", index=False)
    fits.to_csv(TABLES / "main_point3_multivariable_class_share_latlon_ols.csv", index=False)
    lines.to_csv(TABLES / "main_point3_multivariable_class_share_latlon_plot_ols.csv", index=False)
    print(fits.sort_values(["coordinate", "p"]).to_string(index=False))
    print(lines.sort_values(["coordinate", "class_label"]).to_string(index=False))


if __name__ == "__main__":
    main()
