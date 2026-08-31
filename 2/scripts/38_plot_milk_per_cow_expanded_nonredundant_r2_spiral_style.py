#!/usr/bin/env python3
"""Spiral-style R2-family ribbons paired with the point 2 nonredundant beta plot."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
TAB = POINT / "tables"
FIG = POINT / "figures"

METRIC_SPECS = {
    "adjusted_incremental_r2": {
        "label": "Adjusted Increased R²",
        "annotation": "adjusted increased R²",
        "input": TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_adjusted_incremental_r2_point2_style_by_year.csv",
        "stem": "main_point2_milk_per_cow_expanded_nonredundant_yearly_adjusted_incremental_r2_spiral_style",
        "points": TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_adjusted_incremental_r2_spiral_points.csv",
    },
}

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
START_ANGLE_DEG = 90
BACKGROUND = "none"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "text.color": "#171717",
        "axes.labelcolor": "#171717",
        "xtick.color": "#171717",
        "ytick.color": "#171717",
        "svg.fonttype": "none",
    }
)


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    svg = svg.replace("font-size: 9px;", "font-size: 9.00px;")
    svg = svg.replace("font-size: 10px;", "font-size: 10.00px;")
    path.write_text(svg, encoding="utf-8")


def smooth_interp(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    try:
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(x, y)(xi)
    except Exception:
        return np.interp(xi, x, y)


def add_text(ax: plt.Axes, x: float, y: float, text: str, **kwargs) -> None:
    defaults = {"ha": "center", "va": "center", "fontsize": 9, "color": "#171717"}
    defaults.update(kwargs)
    ax.text(x, y, text, **defaults)


def year_to_theta_deg(year: float | np.ndarray) -> float | np.ndarray:
    year_arr = np.asarray(year, dtype=float)
    theta = np.where(
        year_arr <= 2020,
        START_ANGLE_DEG - 18 * (year_arr - 2000),
        START_ANGLE_DEG - 360 - 15 * (year_arr - 2020),
    )
    if np.isscalar(year):
        return float(theta)
    return theta


def spiral_position(year: float) -> tuple[float, float]:
    theta = np.deg2rad(year_to_theta_deg(year))
    radius = 0.72 + 1.82 * (year - 2000) / (2024 - 2000)
    return theta, radius


def make_spiral_data(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = pd.read_csv(input_path)
    annual = annual.loc[annual["year"].between(2000, 2024), ["year", *DOMAIN_ORDER]].copy()
    if annual["year"].tolist() != list(range(2000, 2025)):
        raise RuntimeError(f"Expected complete 2000-2024 annual R2 table: {input_path}")
    for domain in DOMAIN_ORDER:
        annual[domain] = pd.to_numeric(annual[domain], errors="coerce").clip(lower=0)
    annual["total_positive"] = annual[DOMAIN_ORDER].sum(axis=1)
    annual["total_pct"] = annual["total_positive"] * 100

    t_year = annual["year"].to_numpy(float)
    t = np.linspace(t_year.min(), t_year.max(), 720)
    # Start near the top and run clockwise through a little over one full turn.
    theta = np.deg2rad(year_to_theta_deg(t))
    r = 0.72 + 1.82 * (t - t.min()) / (t.max() - t.min())
    center = pd.DataFrame(
        {
            "year_continuous": t,
            "theta": theta,
            "radius": r,
            "x": r * np.cos(theta),
            "y": r * np.sin(theta),
        }
    )
    for domain in DOMAIN_ORDER:
        center[domain] = smooth_interp(t_year, annual[domain].to_numpy(float), t)
    center[DOMAIN_ORDER] = center[DOMAIN_ORDER].clip(lower=0)
    center["total_positive"] = center[DOMAIN_ORDER].sum(axis=1)
    return annual, center


def draw_spiral(
    ax: plt.Axes,
    annual: pd.DataFrame,
    center: pd.DataFrame,
    annotation: str,
    show_text: bool = True,
) -> None:
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")

    x = center["x"].to_numpy(float)
    y = center["y"].to_numpy(float)
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2)
    nx = -dy / norm
    ny = dx / norm
    start_blend = np.clip((center["year_continuous"].to_numpy(float) - 2000.0) / 0.65, 0.0, 1.0)
    nx = nx * start_blend
    ny = np.sqrt(np.clip(1.0 - nx**2, 0.0, 1.0))
    # End exactly on the upper-right radial guide instead of extending beyond it.
    nx[-1] = np.cos(center["theta"].iloc[-1])
    ny[-1] = np.sin(center["theta"].iloc[-1])

    max_total = center["total_positive"].max()
    thickness_scale = 1.55 / max_total if max_total > 0 else 1.0
    values = center[DOMAIN_ORDER].to_numpy(float) * thickness_scale
    total_width = values.sum(axis=1)

    for rad in np.deg2rad(np.mod(year_to_theta_deg(np.array([2000, 2005, 2010, 2015, 2020, 2024])), 360)):
        ax.plot(
            [0, 2.78 * np.cos(rad)],
            [0, 2.78 * np.sin(rad)],
            color="#000000",
            lw=0.90,
            ls=(0, (2, 3)),
            zorder=8,
        )

    cum = -0.5 * total_width
    for domain in DOMAIN_ORDER:
        width = values[:, DOMAIN_ORDER.index(domain)]
        lower = cum
        upper = cum + width
        x_lower = x + nx * lower
        y_lower = y + ny * lower
        x_upper = x + nx * upper
        y_upper = y + ny * upper
        poly = np.column_stack(
            [
                np.r_[x_lower, x_upper[::-1]],
                np.r_[y_lower, y_upper[::-1]],
            ]
        )
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=COLORS[domain],
                edgecolor="none",
                alpha=1.0,
                zorder=2,
            )
        )
        cum = upper

    ax.plot(x, y, color="#111111", lw=1.15, zorder=4)

    if show_text:
        # Year labels are anchored to the spiral's year angle and offset radially, so 2000/2010/2020
        # remain aligned with the central vertical guide.
        year_offsets = {
            2000: (0.22, "center", "center"),
            2005: (0.18, "left", "center"),
            2010: (0.18, "center", "center"),
            2015: (0.18, "right", "center"),
            2020: (0.18, "center", "center"),
            2024: (0.22, "left", "center"),
        }
        for year, (offset, ha, va) in year_offsets.items():
            theta_label, radius_label = spiral_position(float(year))
            label_radius = radius_label + offset
            add_text(
                ax,
                label_radius * np.cos(theta_label),
                label_radius * np.sin(theta_label),
                str(year),
                ha=ha,
                va=va,
                fontsize=9,
            )

        add_text(ax, -2.72, 2.22, f"Stacked median\n{annotation}", ha="left", va="top", fontsize=9.2, fontweight="bold")
        add_text(ax, -2.72, 1.96, f"0 to {annual['total_pct'].max():.1f}%", ha="left", va="top", fontsize=7.6)

        # Compact domain legend along the bottom.
        legend_rows = [
            (["Heat", "Cold", "Severe weather", "Forage"], -2.50),
            (["Feed market", "Dairy market", "Market demand"], -2.68),
        ]
        for row_domains, legend_y in legend_rows:
            x_cursor = -2.40
            for domain in row_domains:
                ax.plot([x_cursor, x_cursor + 0.18], [legend_y, legend_y], color=COLORS[domain], lw=5, solid_capstyle="butt")
                add_text(ax, x_cursor + 0.21, legend_y, domain, ha="left", va="center", fontsize=7.0)
                x_cursor += 0.72 if domain not in {"Severe weather", "Market demand", "Dairy market"} else 1.05

    ax.set_xlim(-2.95, 3.0)
    ax.set_ylim(-2.72, 2.74)


def save_figure(fig: plt.Figure, stem: str) -> None:
    out = FIG / f"{stem}.svg"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="none", edgecolor="none", transparent=True)
    normalize_svg_text_style(out)
    print(f"Wrote {out}")


def plot_one(metric: str, spec: dict[str, str | Path]) -> None:
    annual, center = make_spiral_data(Path(spec["input"]))
    Path(spec["points"]).write_text(center.to_csv(index=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.15, 5.95), constrained_layout=False)
    fig.patch.set_facecolor("none")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.065, top=0.885)
    draw_spiral(ax, annual, center, str(spec["annotation"]), show_text=True)
    fig.text(0.035, 0.965, "Milk Per Cow Sensitivity", ha="left", va="top", fontsize=13, fontweight="bold")
    fig.text(0.19, 0.915, str(spec["label"]), ha="left", va="top", fontsize=11, fontweight="bold")
    save_figure(fig, str(spec["stem"]))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(2.575, 2.975), constrained_layout=False)
    fig.patch.set_facecolor("none")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    draw_spiral(ax, annual, center, str(spec["annotation"]), show_text=False)
    save_figure(fig, f"{spec['stem']}_wo_legend")
    print(f"Wrote {spec['points']}")
    plt.close(fig)


def plot() -> None:
    for metric, spec in METRIC_SPECS.items():
        plot_one(metric, spec)


def main() -> int:
    plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
