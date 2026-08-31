#!/usr/bin/env python3
"""Map state-level equal-weighted priority shares by domain."""
from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.sampledata.us_states import data as US_STATES
from matplotlib.patches import Polygon


POINT4 = Path(__file__).resolve().parents[1]
POINT50 = POINT4.parent
POINT0 = POINT50 / "0"
POINT3 = POINT50 / "3"
TAB4 = POINT4 / "tables"
FIG4 = POINT4 / "figures"
TAB0 = POINT0 / "tables"
TAB3 = POINT3 / "tables"
FIG4.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "Heat",
    "Cold",
    "Severe weather",
    "Forage",
    "Feed market",
    "Dairy market",
    "Market demand",
]
ALL_DOMAINS = [
    "Heat",
    "Cold",
    "Severe weather",
    "Forage",
    "Feed market",
    "Dairy market",
    "Market demand",
]
DOMAIN_COLORS = {
    "Heat": "#32a4b4",
    "Cold": "#33c5b2",
    "Severe weather": "#d5eada",
    "Forage": "#1E7A8D",
    "Feed market": "#fbc4ab",
    "Dairy market": "#E47666",
    "Market demand": "#f09d51",
}
EDGE = "#6F6F6F"
FILL_OTHER = "#FFFFFF"
FONT_SIZE = 9
SHADOW_FILL = "#CFCFCF"
SHADOW_OFFSET = (0.55, -0.55)


def clean_svg(path: Path) -> None:
    svg = path.read_text()
    svg = re.sub(
        r"font: ([0-9.]+)px 'DejaVu Sans'[^;]*;",
        r"font-size: \1px; font-family: 'Arial';",
        svg,
    )
    svg = re.sub(
        r"font: ([0-9.]+)px 'Arial'",
        r"font-size: \1px; font-family: 'Arial'",
        svg,
    )
    path.write_text(svg)


def transformed_coords(state: str, lons: list[float], lats: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(lons, dtype=float)
    y = np.asarray(lats, dtype=float)
    if state == "AK":
        x = (x + 179.0) * 0.35 - 124.0
        y = (y - 51.0) * 0.35 + 24.0
    elif state == "HI":
        x = x + 46.2
        y = y + 4.4
    return x, y


def percent_rank(values: pd.Series) -> pd.Series:
    valid = values.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.sum() <= 1:
        out.loc[valid] = 0.0
        return out
    out.loc[valid] = (values.loc[valid].rank(method="min") - 1) / (valid.sum() - 1)
    return out


def build_shap_share_index() -> pd.DataFrame:
    blocks = []
    for horizon in range(1, 10):
        path = TAB3 / f"point3_point4aligned_h{horizon}_shap_state_month_class.csv"
        data = pd.read_csv(path)
        data = data.loc[data.class_label.isin(ALL_DOMAINS)].copy()
        data["horizon_months"] = horizon
        blocks.append(data)
    raw = pd.concat(blocks, ignore_index=True)
    keys = ["state_alpha", "region", "year", "month", "horizon_months"]
    raw["class_abs_shap"] = raw["class_abs_signed_shap"].astype(float)
    raw["total_abs_shap"] = raw.groupby(keys)["class_abs_shap"].transform("sum")
    raw["class_shap_share"] = np.where(
        raw.total_abs_shap > 0,
        raw.class_abs_shap / raw.total_abs_shap,
        np.nan,
    )
    out = (
        raw.groupby(["state_alpha", "class_label"], as_index=False)
        .agg(forecast_contribution=("class_shap_share", "mean"))
    )
    out["forecast_index"] = out.groupby("class_label")["forecast_contribution"].transform(percent_rank)
    return out[["state_alpha", "class_label", "forecast_index"]]


def build_equal_weighted_priority_share() -> pd.DataFrame:
    priority = pd.read_csv(TAB4 / "main_point4_state_class_priority_top20_overlap.csv")
    shap = build_shap_share_index()
    data = priority.merge(shap, on=["state_alpha", "class_label"], how="left", suffixes=("", "_share_based"))
    data["forecast_index"] = data["forecast_index_share_based"].combine_first(data["forecast_index"])
    index_cols = ["beta_std_index", "adjusted_incremental_r2_index", "forecast_index"]
    data["equal_weighted_priority_raw"] = data[index_cols].mean(axis=1, skipna=True)
    data["equal_weighted_priority_share"] = (
        data["equal_weighted_priority_raw"]
        / data.groupby("state_alpha")["equal_weighted_priority_raw"].transform("sum")
    )
    combined = data[[
        "state_alpha", "region", "class_label", "equal_weighted_priority_raw", "equal_weighted_priority_share"
    ]].copy()
    combined.to_csv(TAB4 / "main_point4_state_class_equal_weighted_priority_share.csv", index=False)
    return combined


def add_state(
    ax: plt.Axes,
    state: str,
    state_data: dict,
    value_by_state: dict[str, float],
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> None:
    x, y = transformed_coords(state, state_data["lons"], state_data["lats"])
    isnan = np.isnan(x) | np.isnan(y)
    breaks = np.where(isnan)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(x)]
    value = value_by_state.get(state)
    face = cmap(norm(value)) if value is not None and np.isfinite(value) else FILL_OTHER
    for start, end in zip(starts, ends):
        if end - start < 3:
            continue
        coords = np.column_stack([x[start:end], y[start:end]])
        ax.add_patch(
            Polygon(
                coords,
                closed=True,
                facecolor=face,
                edgecolor=EDGE,
                linewidth=0.35,
                joinstyle="round",
            )
        )


def add_state_shadow(ax: plt.Axes, state: str, state_data: dict) -> None:
    x, y = transformed_coords(state, state_data["lons"], state_data["lats"])
    x = x + SHADOW_OFFSET[0]
    y = y + SHADOW_OFFSET[1]
    isnan = np.isnan(x) | np.isnan(y)
    breaks = np.where(isnan)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(x)]
    for start, end in zip(starts, ends):
        if end - start < 3:
            continue
        coords = np.column_stack([x[start:end], y[start:end]])
        ax.add_patch(
            Polygon(
                coords,
                closed=True,
                facecolor=SHADOW_FILL,
                edgecolor="none",
                linewidth=0,
                alpha=0.55,
                joinstyle="round",
                zorder=0,
            )
        )


def plot_domain_map(data: pd.DataFrame, domain: str) -> None:
    domain_data = data.loc[data.class_label.eq(domain)].copy()
    vals = domain_data.equal_weighted_priority_share.astype(float)
    vmin = float(vals.min(skipna=True))
    vmax = float(vals.max(skipna=True))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        domain_data["map_priority_scaled"] = (vals - vmin) / (vmax - vmin)
    else:
        domain_data["map_priority_scaled"] = np.where(vals.notna(), 0.0, np.nan)
    value_by_state = dict(zip(domain_data.state_alpha, domain_data.map_priority_scaled))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{domain}_priority",
        ["#FFFFFF", DOMAIN_COLORS[domain]],
    )

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for state in sorted(US_STATES):
        if state == "DC":
            continue
        add_state_shadow(ax, state, US_STATES[state])
    for state in sorted(US_STATES):
        if state == "DC":
            continue
        add_state(ax, state, US_STATES[state], value_by_state, cmap, norm)

    ax.set_xlim(-125.0, -66.0)
    ax.set_ylim(23.0, 50.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        shrink=0.48,
        aspect=22,
    )
    cbar.ax.tick_params(labelsize=FONT_SIZE, length=2.5, width=0.35, pad=1)
    cbar.outline.set_linewidth(0.35)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0", "0.5", "1.0"])

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.12)
    stem = domain.lower().replace(" ", "_")
    out = FIG4 / f"main_point4_state_{stem}_equal_weighted_priority_share_us_map.svg"
    fig.savefig(
        out,
        format="svg",
        transparent=True,
    )
    clean_svg(out)
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "svg.fonttype": "none",
        }
    )
    states = set(pd.read_csv(TAB0 / "point0_50_state_percow_state_list.csv")["state_alpha"])
    data = build_equal_weighted_priority_share()
    data = data.loc[data.state_alpha.isin(states)].copy()
    for domain in DOMAINS:
        plot_domain_map(data, domain)
    print(data.loc[data.class_label.isin(DOMAINS)].groupby("class_label").equal_weighted_priority_share.describe().to_string())


if __name__ == "__main__":
    main()
