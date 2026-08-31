#!/usr/bin/env python3
"""Test state-level latitude/longitude gradients in multivariable RF domain SHAP.

This analysis uses exactly the held-out, class-level SHAP outputs underlying
``supp_point3_rf_regional_class_contribution_by_horizon``. For each state and
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


def state_phase_contribution() -> pd.DataFrame:
    blocks = []
    for horizon in range(1, 10):
        data = pd.read_csv(
            TABLES / f"supp_point3_rf_h{horizon}_shap_state_month_class.csv",
            usecols=["horizon_months", "state_alpha", "class_label", "class_abs_signed_shap"],
        )
        data = data.loc[data.class_label.isin(DOMAINS)].copy()
        blocks.append(data)
    data = pd.concat(blocks, ignore_index=True)
    # Within a state and horizon, every held-out month has equal weight.
    state_horizon = (
        data.groupby(["horizon_months", "state_alpha", "class_label"], as_index=False)
        .agg(mean_abs_net_shap=("class_abs_signed_shap", "mean"), n_heldout_state_months=("class_abs_signed_shap", "size"))
    )
    rows = []
    for phase, horizons in PHASES.items():
        phase_data = state_horizon.loc[state_horizon.horizon_months.isin(horizons)]
        summary = (
            phase_data.groupby(["state_alpha", "class_label"], as_index=False)
            .agg(mean_abs_net_shap=("mean_abs_net_shap", "mean"), n_horizons=("horizon_months", "nunique"))
        )
        summary["phase"] = phase
        rows.append(summary)
    centers = pd.read_csv(CENTERS)
    result = pd.concat(rows, ignore_index=True).merge(centers, on="state_alpha", how="left", validate="many_to_one")
    if result[["longitude", "latitude"]].isna().any().any():
        missing = result.loc[result.latitude.isna(), "state_alpha"].unique().tolist()
        raise RuntimeError(f"Missing geographic centers for: {missing}")
    return result


def test_gradients(values: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, excluded in SCOPES.items():
        scoped = values.loc[~values.state_alpha.isin(excluded)]
        for phase in PHASES:
            for domain in DOMAINS:
                data = scoped.loc[(scoped.phase.eq(phase)) & (scoped.class_label.eq(domain))]
                for coordinate in ("latitude", "longitude"):
                    test = (
                        stats.spearmanr(data[coordinate], data.mean_abs_net_shap)
                        if data.mean_abs_net_shap.nunique() > 1
                        else None
                    )
                    rows.append({
                        "scope": scope, "phase": phase, "class_label": domain,
                        "coordinate": coordinate, "n_states": len(data),
                        "spearman_rho": float(test.statistic) if test is not None else np.nan,
                        "spearman_p": float(test.pvalue) if test is not None else np.nan,
                    })
    results = pd.DataFrame(rows)
    results["spearman_q_bh_fdr"] = results.groupby("scope")["spearman_p"].transform(
        lambda p: benjamini_hochberg(p)
    )
    results["fdr_significance"] = results.spearman_q_bh_fdr.map(stars)
    return results


def plot_pooled_scatter(values: pd.DataFrame, results: pd.DataFrame) -> None:
    """Make one latitude and one longitude scatter for robust associations."""
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "text.color": "#222222",
        "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
        "svg.fonttype": "none",
    })
    full = results.loc[results.scope.eq("All 50 states")].set_index(["phase", "class_label", "coordinate"])
    contiguous = results.loc[results.scope.eq("Contiguous 48")].set_index(["phase", "class_label", "coordinate"])
    candidates = []
    for phase in PHASES:
        for domain in DOMAINS:
            for coordinate in ("latitude", "longitude"):
                key = (phase, domain, coordinate)
                if full.loc[key, "spearman_q_bh_fdr"] < 0.05 and contiguous.loc[key, "spearman_q_bh_fdr"] < 0.05:
                    candidates.append(key)
    if not candidates:
        raise RuntimeError("No associations are robust to the contiguous-48 sensitivity.")
    selected = pd.DataFrame(candidates, columns=["phase", "class_label", "coordinate"])
    display = values.merge(selected[["phase", "class_label"]].drop_duplicates(), on=["phase", "class_label"], how="inner")
    display["standardized_abs_shap"] = display.groupby(["phase", "class_label"])["mean_abs_net_shap"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )
    phase_markers = {"H1-H3": "o", "H4-H6": "s", "H7-H9": "^"}
    phase_lines = {"H1-H3": "solid", "H4-H6": "dashed", "H7-H9": "dotted"}
    domain_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=DOMAIN_COLORS[domain],
                   markeredgecolor="none", markersize=5.5, label=domain)
        for domain in DOMAINS
    ]
    phase_handles = [
        plt.Line2D([0], [0], marker=phase_markers[phase], linestyle="", color="#222222",
                   markerfacecolor="#222222", markersize=5.0, label=phase)
        for phase in PHASES
    ]
    for coordinate, coordinate_label in (("latitude", "Latitude (degrees)"), ("longitude", "Longitude (degrees)")):
        active = selected.loc[selected.coordinate.eq(coordinate), ["phase", "class_label"]]
        data = display.merge(active, on=["phase", "class_label"], how="inner")
        fig, axis = plt.subplots(figsize=(7.2, 4.5))
        for row in active.itertuples(index=False):
            group = data.loc[(data.phase.eq(row.phase)) & (data.class_label.eq(row.class_label))].sort_values(coordinate)
            x = group[coordinate].to_numpy(float)
            y = group.standardized_abs_shap.to_numpy(float)
            axis.scatter(
                x, y, s=24, marker=phase_markers[row.phase], color=DOMAIN_COLORS[row.class_label],
                alpha=1.0, linewidths=0, zorder=2,
            )
            slope, intercept, _, _, _ = stats.linregress(x, y)
            axis.plot(x, intercept + slope * x, color=DOMAIN_COLORS[row.class_label],
                      linestyle=phase_lines[row.phase], linewidth=1.0, alpha=1.0, zorder=1)
        axis.set_xlabel(coordinate_label)
        axis.set_ylabel("Within domain-phase standardized held-out |SHAP|")
        axis.grid(axis="both", color="#d9d9d9", linestyle=":", linewidth=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=8, length=3, width=0.7)
        fig.legend(domain_handles, DOMAINS, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   frameon=False, fontsize=8, columnspacing=0.9, handletextpad=0.35)
        fig.legend(phase_handles, list(PHASES), ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   frameon=False, fontsize=8, columnspacing=1.0, handletextpad=0.35)
        fig.text(
            0.5, 0.01,
            "Shown: associations with q < 0.05 in both all-50 and contiguous-48 analyses; trend lines are OLS visual fits.",
            ha="center", va="bottom", fontsize=8,
        )
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.15, top=0.79)
        for suffix in ("png", "svg"):
            path = FIGURES / f"supp_point3_rf_class_contribution_{coordinate}_scatter.{suffix}"
            fig.savefig(path, dpi=300, transparent=True, facecolor="none", edgecolor="none", bbox_inches="tight")
            if suffix == "svg":
                normalize_svg_text_style(path)
        plt.close(fig)


def main() -> None:
    values = state_phase_contribution()
    results = test_gradients(values)
    values.to_csv(TABLES / "supp_point3_rf_class_contribution_state_phase.csv", index=False)
    results.to_csv(TABLES / "supp_point3_rf_class_contribution_latlon_spearman.csv", index=False)
    print(results.sort_values(["scope", "phase", "coordinate", "spearman_q_bh_fdr"]).to_string(index=False))


if __name__ == "__main__":
    main()
