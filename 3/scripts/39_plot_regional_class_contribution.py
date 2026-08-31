#!/usr/bin/env python3
"""Plot regional exposure-class contributions from Point 3 held-out SHAP.

The input tables are produced by the same nested, rolling-origin multivariable
HGB models used for ``main_point3_point4aligned_multihorizon_model_comparison``.
For every held-out state-month, each class contribution is the signed sum of
its feature SHAP values.  We use the absolute value of that sum as the domain's
net contribution strength, average it within region and horizon, then express
it as a share of the seven retained exposure domains.
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
TABLES, FIGURES = HERE / "tables", HERE / "figures"
DOMAINS = (
    "Heat", "Cold", "Severe weather", "Forage", "Feed market",
    "Dairy market", "Market demand",
)
REGIONS = ("South", "West", "Midwest", "Northeast")
PHASES = {
    "Short-term (1-3 months ahead)": (1, 2, 3),
    "Mid-term (4-6 months ahead)": (4, 5, 6),
    "Long-term (7-9 months ahead)": (7, 8, 9),
}
COLORS = {
    "Heat": "#32a4b4", "Cold": "#33c5b2", "Severe weather": "#d5eada",
    "Forage": "#1E7A8D", "Feed market": "#fbc4ab", "Dairy market": "#E47666",
    "Market demand": "#f09d51",
}


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    path.write_text(svg, encoding="utf-8")


def summarize() -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks = []
    for horizon in range(1, 10):
        path = TABLES / f"point3_point4aligned_h{horizon}_shap_state_month_class.csv"
        data = pd.read_csv(path)
        data = data.loc[data["class_label"].isin(DOMAINS)].copy()
        if not data["horizon_months"].eq(horizon).all():
            raise RuntimeError(f"Unexpected horizon values in {path.name}")
        blocks.append(data)
    data = pd.concat(blocks, ignore_index=True)

    # Each input row represents one held-out state-month and one domain; zero
    # contributions are retained, so unselected domain features count as zero.
    by_horizon = (
        data.groupby(["horizon_months", "region", "class_label"], as_index=False)
        .agg(
            mean_abs_net_shap=("class_abs_signed_shap", "mean"),
            n_heldout_state_months=("class_abs_signed_shap", "size"),
        )
    )
    expected = by_horizon.groupby(["horizon_months", "region"])["n_heldout_state_months"].nunique()
    if not expected.eq(1).all():
        raise RuntimeError("Class tables do not contain aligned held-out state-month counts.")

    phase_rows = []
    for phase, horizons in PHASES.items():
        subset = by_horizon.loc[by_horizon.horizon_months.isin(horizons)]
        summary = (
            subset.groupby(["region", "class_label"], as_index=False)["mean_abs_net_shap"]
            .mean()
            .rename(columns={"mean_abs_net_shap": "mean_horizon_abs_net_shap"})
        )
        summary["phase"] = phase
        summary["share_pct"] = 100 * summary["mean_horizon_abs_net_shap"] / summary.groupby("region")["mean_horizon_abs_net_shap"].transform("sum")
        summary["rank_within_region"] = summary.groupby("region")["mean_horizon_abs_net_shap"].rank(method="min", ascending=False).astype(int)
        phase_rows.append(summary)
    phase_summary = pd.concat(phase_rows, ignore_index=True)
    return by_horizon, phase_summary


def plot(phase_summary: pd.DataFrame, legend: bool) -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 11, "text.color": "#222222",
        "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.3), sharey=True)
    handles = []
    for axis, (phase, _) in zip(axes, PHASES.items()):
        subset = phase_summary.loc[phase_summary.phase.eq(phase)]
        for y, region in enumerate(REGIONS):
            left = 0.0
            for domain in DOMAINS:
                value = float(subset.loc[(subset.region.eq(region)) & (subset.class_label.eq(domain)), "share_pct"].iloc[0])
                bar = axis.barh(y, value, left=left, height=0.64, color=COLORS[domain], edgecolor="#222222", linewidth=0.35)
                if phase == next(iter(PHASES)) and y == 0:
                    handles.append(bar[0])
                if domain in {"Heat", "Cold"} and value >= 10:
                    axis.text(left + value / 2, y, f"{value:.0f}%", ha="center", va="center", fontsize=11,
                              color="white" if domain in {"Heat", "Cold"} else "#222222")
                left += value
        axis.set_title(phase, fontsize=11, pad=8)
        axis.set_xlim(0, 100)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlabel("Mean held-out class contribution (%)", labelpad=4)
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#222222")
        axis.spines["bottom"].set_color("#222222")
        axis.tick_params(axis="x", length=3, width=0.7)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(REGIONS)), REGIONS)
    axes[0].invert_yaxis()
    if legend:
        fig.legend(handles, DOMAINS, loc="upper center", ncol=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.98), columnspacing=0.85, handlelength=1.0,
                   handletextpad=0.35, fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.76 if legend else 0.93, wspace=0.15)
    stem = "main_point3_regional_class_contribution_by_horizon"
    if not legend:
        stem += "_wo_legend"
    output = FIGURES / f"{stem}.svg"
    fig.savefig(output, dpi=300, transparent=True, facecolor="none", edgecolor="none")
    normalize_svg_text_style(output)
    plt.close(fig)


def main() -> None:
    by_horizon, phase_summary = summarize()
    by_horizon.to_csv(TABLES / "point3_point4aligned_regional_class_net_shap_by_horizon.csv", index=False)
    phase_summary.to_csv(TABLES / "point3_point4aligned_regional_class_net_shap_by_phase.csv", index=False)
    plot(phase_summary, legend=True)
    plot(phase_summary, legend=False)


if __name__ == "__main__":
    main()
