#!/usr/bin/env python3
"""Test state-level latitude/longitude gradients in multivariable domain SHAP.

This analysis uses exactly the held-out, class-level SHAP outputs underlying
``main_point3_regional_class_contribution_by_horizon``. For each state and
horizon, it averages the absolute net SHAP of a domain across held-out months;
the three forecast phases are then equal-horizon averages. Spearman tests are
reported for all 50 states and, as a geographic sensitivity analysis, the
contiguous 48 states (excluding Alaska and Hawaii).
"""
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
TABLES, FIGURES = HERE / "tables", HERE / "figures"
CENTERS = TABLES / "point3_us_state_geographic_centers.csv"
DOMAINS = (
    "Heat", "Cold", "Severe weather", "Forage", "Feed market",
    "Dairy market", "Market demand",
)
PHASES = {"H1-H3": (1, 2, 3), "H4-H6": (4, 5, 6), "H7-H9": (7, 8, 9)}
SCOPES = {"All 50 states": (), "Contiguous 48": ("AK", "HI")}
DOMAIN_COLORS = {
    "Heat": "#32a4b4", "Cold": "#33c5b2", "Severe weather": "#d5eada",
    "Forage": "#1E7A8D", "Feed market": "#fbc4ab", "Dairy market": "#E47666",
    "Market demand": "#f09d51",
}


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    path.write_text(svg, encoding="utf-8")


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    adjusted = ranked * len(valid_values) / np.arange(1, len(valid_values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    valid_result = np.empty_like(adjusted)
    valid_result[order] = np.clip(adjusted, 0.0, 1.0)
    result[valid] = valid_result
    return result


def stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def feature_phase_contribution() -> pd.DataFrame:
    blocks = []
    for horizon in range(1, 10):
        data = pd.read_csv(
            TABLES / f"point3_point4aligned_h{horizon}_shap_state_feature_summary.csv",
            usecols=[
                "state_alpha", "region", "feature", "exposure", "class_label",
                "subclass_label", "mechanistic_subclass_short", "temporal_transform",
                "mean_abs_shap",
            ],
        )
        data = data.loc[data.class_label.isin(DOMAINS)].copy()
        data["horizon_months"] = horizon
        blocks.append(data)
    data = pd.concat(blocks, ignore_index=True)
    rows = []
    for phase, horizons in PHASES.items():
        phase_data = data.loc[data.horizon_months.isin(horizons)]
        summary = (
            phase_data.groupby([
                "state_alpha", "region", "feature", "exposure", "class_label",
                "subclass_label", "mechanistic_subclass_short", "temporal_transform",
            ], as_index=False)
            .agg(mean_abs_shap=("mean_abs_shap", "mean"), n_horizons=("horizon_months", "nunique"))
        )
        summary["phase"] = phase
        rows.append(summary)
    centers = pd.read_csv(CENTERS)
    result = pd.concat(rows, ignore_index=True).merge(centers, on="state_alpha", how="left", validate="many_to_one")
    if result[["longitude", "latitude"]].isna().any().any():
        missing = result.loc[result.latitude.isna(), "state_alpha"].unique().tolist()
        raise RuntimeError(f"Missing geographic centers for: {missing}")
    return result


def domain_median_contribution(values: pd.DataFrame) -> pd.DataFrame:
    summary = (
        values.groupby(["state_alpha", "region", "phase", "class_label"], as_index=False)
        .agg(
            median_abs_shap=("mean_abs_shap", "median"),
            n_features_in_domain=("feature", "nunique"),
        )
    )
    centers = pd.read_csv(CENTERS)
    result = summary.merge(centers, on="state_alpha", how="left", validate="many_to_one")
    if result[["longitude", "latitude"]].isna().any().any():
        missing = result.loc[result.latitude.isna(), "state_alpha"].unique().tolist()
        raise RuntimeError(f"Missing geographic centers for: {missing}")
    return result


def domain_median_ols(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = values.copy()
    work["standardized_abs_shap"] = work.groupby("class_label")["median_abs_shap"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    for domain, data in work.groupby("class_label", sort=False):
        for coordinate in ("latitude", "longitude"):
            clean = data.dropna(subset=[coordinate, "standardized_abs_shap"])
            if len(clean) < 3 or clean.standardized_abs_shap.nunique() <= 1:
                fit = None
            else:
                fit = stats.linregress(clean[coordinate].to_numpy(float), clean.standardized_abs_shap.to_numpy(float))
            rows.append({
                "class_label": domain,
                "coordinate": coordinate,
                "n_state_phase_points": len(clean),
                "median_n_features_per_state_phase": clean.n_features_in_domain.median(),
                "slope": fit.slope if fit is not None else np.nan,
                "intercept": fit.intercept if fit is not None else np.nan,
                "r": fit.rvalue if fit is not None else np.nan,
                "r2": fit.rvalue ** 2 if fit is not None else np.nan,
                "p": fit.pvalue if fit is not None else np.nan,
            })
    return pd.DataFrame(rows)


def plot_pooled_scatter(values: pd.DataFrame, domain_results: pd.DataFrame) -> pd.DataFrame:
    """Make one scatter per coordinate using class-median feature SHAP."""
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "text.color": "#222222",
        "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
        "svg.fonttype": "none",
    })
    display = values.copy()
    display["standardized_abs_shap"] = display.groupby("class_label")["median_abs_shap"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    domain_line_rows = []
    for coordinate, coordinate_label in (("latitude", "Latitude (°)"), ("longitude", "Longitude (°)")):
        active_domains = domain_results.loc[
            domain_results.coordinate.eq(coordinate) & domain_results.p.lt(0.05)
        ].copy()
        domains = [domain for domain in DOMAINS if domain in set(active_domains.class_label)]
        data = display.loc[display.class_label.isin(domains)]
        domain_handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=DOMAIN_COLORS[domain],
                       markeredgecolor="#222222", markeredgewidth=0.45, markersize=5.5, label=domain)
            for domain in domains
        ]
        for show_legend in (True, False):
            fig, axis = plt.subplots(figsize=(3.8, 4.0))
            for domain in domains:
                group = data.loc[data.class_label.eq(domain)].sort_values(coordinate)
                x = group[coordinate].to_numpy(float)
                y = group.standardized_abs_shap.to_numpy(float)
                axis.scatter(
                    x, y, s=24, marker="o", color=DOMAIN_COLORS[domain],
                    alpha=1.0, edgecolors="#222222", linewidths=0.45, zorder=2,
                )
                fit = stats.linregress(x, y)
                if not show_legend:
                    domain_line_rows.append({
                        "coordinate": coordinate,
                        "class_label": domain,
                        "n_points": len(group),
                        "median_n_features_per_state_phase": group.n_features_in_domain.median(),
                        "slope": fit.slope,
                        "intercept": fit.intercept,
                        "r": fit.rvalue,
                        "r2": fit.rvalue ** 2,
                        "p": fit.pvalue,
                    })
                axis.plot(x, fit.intercept + fit.slope * x, color=DOMAIN_COLORS[domain],
                          linestyle="solid", linewidth=3.0, alpha=1.0, zorder=1)
            axis.set_xlabel(coordinate_label)
            axis.set_ylabel("Standardized class-median held-out |SHAP|")
            axis.grid(axis="both", color="#d9d9d9", linestyle=":", linewidth=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(axis="both", labelsize=9, length=3, width=0.7)
            if not domains:
                axis.text(
                    0.5, 0.5, "No class-median association with OLS p < 0.05",
                    transform=axis.transAxes, ha="center", va="center", fontsize=9, color="#222222",
                )
            if show_legend and domains:
                fig.legend(domain_handles, domains, ncol=len(domains), loc="upper center", bbox_to_anchor=(0.5, 1.02),
                           frameon=False, fontsize=9, columnspacing=0.9, handletextpad=0.35)
                fig.text(
                    0.5, 0.01,
                    "Shown: class medians across all variables; trend lines are OLS visual fits.",
                    ha="center", va="bottom", fontsize=9,
                )
                fig.subplots_adjust(left=0.13, right=0.985, bottom=0.15, top=0.79)
            elif show_legend:
                fig.subplots_adjust(left=0.13, right=0.985, bottom=0.18, top=0.96)
            else:
                fig.subplots_adjust(left=0.13, right=0.985, bottom=0.18, top=0.96)
            suffix = "" if show_legend else "_wo_legend"
            path = FIGURES / f"main_point3_multivariable_class_contribution_{coordinate}_scatter{suffix}.svg"
            fig.savefig(path, dpi=300, transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight")
            normalize_svg_text_style(path)
            plt.close(fig)
    return pd.DataFrame(domain_line_rows)


def main() -> None:
    feature_values = feature_phase_contribution()
    values = domain_median_contribution(feature_values)
    domain_results = domain_median_ols(values)
    feature_values.to_csv(TABLES / "point3_multivariable_feature_contribution_state_phase.csv", index=False)
    values.to_csv(TABLES / "point3_multivariable_class_median_feature_contribution_state_phase.csv", index=False)
    domain_results.to_csv(TABLES / "point3_multivariable_class_median_feature_contribution_latlon_ols.csv", index=False)
    domain_lines = plot_pooled_scatter(values, domain_results)
    domain_lines.to_csv(TABLES / "point3_multivariable_class_median_feature_contribution_latlon_plot_ols.csv", index=False)
    print(domain_results.sort_values(["coordinate", "p"]).to_string(index=False))
    print(domain_lines.sort_values(["coordinate", "class_label"]).to_string(index=False))


if __name__ == "__main__":
    main()
