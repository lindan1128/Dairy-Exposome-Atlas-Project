#!/usr/bin/env python3
"""Region split of the national adjusted incremental R2 spiral."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
TAB = POINT / "tables"
FIG = POINT / "figures"
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402


SELECTION = TAB / "point2_common_sense_expanded_kept_variables.csv"
THREE_METRIC_VALUES = TAB / "point2_beta_std_two_stage_three_metric_by_variable.csv"
THREE_METRIC_OUTLIERS = TAB / "point2_beta_std_two_stage_three_metric_boxplots_by_class_outliers_removed.csv"
OUT_LONG = TAB / "point2_milk_per_cow_expanded_nonredundant_national_model_region_split_yearly_adjusted_incremental_r2.csv"
OUT_SUMMARY = TAB / "point2_milk_per_cow_expanded_nonredundant_national_model_region_split_yearly_adjusted_incremental_r2_spiral_style_by_year.csv"

KEY = ["state_alpha", "region", "year", "month"]
YEARS = list(range(2000, 2026))
PLOT_YEARS = list(range(2000, 2025))
LB_TO_KG = 0.45359237
REGION_ORDER = ["Northeast", "Midwest", "South", "West"]
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


def retained_exposures() -> set[str]:
    """Variables retained in the mean |standardized beta| supplement boxplot."""
    if THREE_METRIC_VALUES.exists():
        values = pd.read_csv(THREE_METRIC_VALUES)
        paired = values.dropna(subset=["mean_abs_beta_2000_2014", "mean_abs_beta_2015_2024"]).copy()
        if THREE_METRIC_OUTLIERS.exists():
            outliers = pd.read_csv(THREE_METRIC_OUTLIERS)
            beta_outliers = set(outliers.loc[outliers["metric"].eq("mean_abs_beta_std"), "exposure"].dropna())
        else:
            beta_outliers = {"market_log_population_total", "storm_event_types"}
        return set(paired["exposure"]) - beta_outliers
    selected = pd.read_csv(SELECTION)
    selected = selected[selected["expanded_selection_status"].str.startswith("kept_expanded", na=False)].copy()
    return set(selected["exposure"]) - {"market_log_population_total", "storm_event_types"}


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    svg = svg.replace("font-size: 7px;", "font-size: 7.00px;")
    svg = svg.replace("font-size: 9px;", "font-size: 9.00px;")
    svg = svg.replace("font-size: 10px;", "font-size: 10.00px;")
    path.write_text(svg, encoding="utf-8")


def residualize(m: np.ndarray, fe: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sw = np.sqrt(weights / np.nanmean(weights))
    mw = m * sw[:, None]
    few = fe * sw[:, None]
    coef, *_ = np.linalg.lstsq(few, mw, rcond=None)
    return mw - few @ coef


def adjusted_r2(sse: float, sst: float, df_resid: int, df_total: int) -> float:
    if sst <= 1e-12 or df_resid <= 0 or df_total <= 0:
        return np.nan
    return 1.0 - (sse / df_resid) / (sst / df_total)


def prepare_panel() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel().copy()
    panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * LB_TO_KG
    panel["log_per_cow"] = np.log(panel["milk_per_cow_kg"].where(panel["milk_per_cow_kg"] > 0))
    return panel


def fit_national_model_region_split(panel: pd.DataFrame, x_col: str) -> list[dict]:
    needed = KEY + ["log_per_cow", x_col, "milk_cows_head"]
    d = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = d[d["milk_cows_head"] > 0]
    if len(d) < 300 or d["state_alpha"].nunique() < 6 or d[x_col].nunique(dropna=True) <= 1:
        return [{"region": region, "year": year, "status": "too_few"} for region in REGION_ORDER for year in YEARS]

    y = d["log_per_cow"].to_numpy(float)
    x = L._standardize(d[x_col].to_numpy(float))
    year_arr = d["year"].to_numpy(int)
    x_year = np.column_stack([x * (year_arr == yv) for yv in YEARS])
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
    design = resid[:, 1:]
    sst = float(np.sum(y_r**2))
    fe_rank = int(np.linalg.matrix_rank(fe))
    df_total = int(len(d) - fe_rank)
    denom = sst / df_total if sst > 1e-12 and df_total > 0 else np.nan

    x_cols = len(YEARS)
    keep_x = np.nanstd(design[:, :x_cols], axis=0) > 1e-10
    keep = keep_x
    if not keep_x.any() or not np.isfinite(denom):
        return [{"region": region, "year": year, "status": "collinear"} for region in REGION_ORDER for year in YEARS]

    xh = design[:, keep]
    kept_cols = np.where(keep)[0]
    beta, *_ = np.linalg.lstsq(xh, y_r, rcond=None)
    resid_full = y_r - xh @ beta
    sse_full = float(np.sum(resid_full**2))
    df_full = int(len(d) - fe_rank - xh.shape[1])
    full_adjusted_r2 = adjusted_r2(sse_full, sst, df_full, df_total)
    region_arr = d["region"].astype(str).to_numpy()

    rows = []
    for year_i, year in enumerate(YEARS):
        if not keep_x[year_i]:
            rows.extend({"region": region, "year": year, "status": "collinear"} for region in REGION_ORDER)
            continue
        pos = int(np.where(kept_cols == year_i)[0][0])
        reduced = np.delete(xh, pos, axis=1)
        beta_reduced, *_ = np.linalg.lstsq(reduced, y_r, rcond=None)
        resid_reduced = y_r - reduced @ beta_reduced
        sse_reduced = float(np.sum(resid_reduced**2))
        df_reduced = int(len(d) - fe_rank - reduced.shape[1])
        reduced_adjusted_r2 = adjusted_r2(sse_reduced, sst, df_reduced, df_total)
        total_adjusted_incremental = (
            full_adjusted_r2 - reduced_adjusted_r2
            if np.isfinite(full_adjusted_r2) and np.isfinite(reduced_adjusted_r2)
            else np.nan
        )
        for region in REGION_ORDER:
            mask = region_arr == region
            region_sse_full = float(np.sum(resid_full[mask] ** 2))
            region_sse_reduced = float(np.sum(resid_reduced[mask] ** 2))
            split = (
                (region_sse_reduced / df_reduced - region_sse_full / df_full) / denom
                if df_full > 0 and df_reduced > 0 and np.isfinite(denom)
                else np.nan
            )
            rows.append(
                {
                    "region": region,
                    "year": year,
                    "status": "ok",
                    "n": int(len(d)),
                    "n_states": int(d["state_alpha"].nunique()),
                    "region_n": int(mask.sum()),
                    "region_n_states": int(d.loc[mask, "state_alpha"].nunique()),
                    "adjusted_incremental_r2_region_split": float(split) if np.isfinite(split) else np.nan,
                    "adjusted_incremental_r2_national": float(total_adjusted_incremental)
                    if np.isfinite(total_adjusted_incremental)
                    else np.nan,
                    "full_adjusted_r2": float(full_adjusted_r2) if np.isfinite(full_adjusted_r2) else np.nan,
                    "reduced_adjusted_r2": float(reduced_adjusted_r2) if np.isfinite(reduced_adjusted_r2) else np.nan,
                }
            )
    return rows


def build_long() -> pd.DataFrame:
    selected = pd.read_csv(SELECTION)
    selected = selected[selected["expanded_selection_status"].str.startswith("kept_expanded", na=False)].copy()
    selected = selected[selected["exposure"].isin(retained_exposures())].copy()
    panel = prepare_panel()
    rows = []
    for _, r in selected.iterrows():
        x = r["exposure"]
        if x not in panel.columns:
            continue
        for fit in fit_national_model_region_split(panel, x):
            rows.append(
                {
                    "outcome": "per_cow",
                    "outcome_label": "Milk per cow (kg)",
                    "domain_label": r["domain_label"],
                    "exposure": x,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    d = long[
        long["status"].eq("ok")
        & long["year"].isin(PLOT_YEARS)
        & long["domain_label"].isin(DOMAIN_ORDER)
    ].copy()
    d["plot_metric"] = d["adjusted_incremental_r2_region_split"].clip(lower=0)
    return (
        d.groupby(["region", "year", "domain_label"], as_index=False)
        .agg(median_metric=("plot_metric", "median"))
        .pivot(index=["region", "year"], columns="domain_label", values="median_metric")
        .reindex(columns=DOMAIN_ORDER)
        .sort_index()
        .reset_index()
    )


def smooth_interp(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    try:
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(x, y)(xi)
    except Exception:
        return np.interp(xi, x, y)


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


def make_spiral_data(annual: pd.DataFrame) -> pd.DataFrame:
    annual = annual.loc[annual["year"].isin(PLOT_YEARS), ["year", *DOMAIN_ORDER]].copy()
    annual = annual.set_index("year").reindex(PLOT_YEARS).reset_index()
    for domain in DOMAIN_ORDER:
        annual[domain] = pd.to_numeric(annual[domain], errors="coerce").fillna(0).clip(lower=0)

    t_year = annual["year"].to_numpy(float)
    t = np.linspace(t_year.min(), t_year.max(), 720)
    theta = np.deg2rad(year_to_theta_deg(t))
    r = 0.72 + 1.82 * (t - t.min()) / (t.max() - t.min())
    center = pd.DataFrame({"year_continuous": t, "theta": theta, "radius": r, "x": r * np.cos(theta), "y": r * np.sin(theta)})
    for domain in DOMAIN_ORDER:
        center[domain] = smooth_interp(t_year, annual[domain].to_numpy(float), t)
    center[DOMAIN_ORDER] = center[DOMAIN_ORDER].clip(lower=0)
    center["total_positive"] = center[DOMAIN_ORDER].sum(axis=1)
    return center


def add_text(ax: plt.Axes, x: float, y: float, text: str, **kwargs) -> None:
    defaults = {"ha": "center", "va": "center", "fontsize": 9, "color": "#171717"}
    defaults.update(kwargs)
    ax.text(x, y, text, **defaults)


def draw_spiral(
    ax: plt.Axes,
    annual: pd.DataFrame,
    center: pd.DataFrame,
    region: str,
    show_text: bool,
    global_max_total: float | None = None,
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
    nx[-1] = np.cos(center["theta"].iloc[-1])
    ny[-1] = np.sin(center["theta"].iloc[-1])

    local_max_total = center["total_positive"].max()
    max_total = global_max_total if global_max_total is not None else local_max_total
    region_boost = 5.0 if region == "Northeast" else 1.0
    thickness_scale = (2.35 * region_boost) / max_total if max_total > 0 else 1.0
    values = center[DOMAIN_ORDER].to_numpy(float) * thickness_scale
    total_width = values.sum(axis=1)

    for rad in np.deg2rad(np.mod(year_to_theta_deg(np.array([2000, 2005, 2010, 2015, 2020, 2024])), 360)):
        ax.plot([0, 2.78 * np.cos(rad)], [0, 2.78 * np.sin(rad)], color="#000000", lw=0.90, ls=(0, (2, 3)), zorder=8)
    cum = -0.5 * total_width
    for domain in DOMAIN_ORDER:
        width = values[:, DOMAIN_ORDER.index(domain)]
        lower = cum
        upper = cum + width
        poly = np.column_stack(
            [
                np.r_[x + nx * lower, (x + nx * upper)[::-1]],
                np.r_[y + ny * lower, (y + ny * upper)[::-1]],
            ]
        )
        ax.add_patch(Polygon(poly, closed=True, facecolor=COLORS[domain], edgecolor="none", alpha=1.0, zorder=2))
        cum = upper

    ax.plot(x, y, color="#111111", lw=1.15, zorder=4)

    if show_text:
        year_offsets = {2000: (0.22, "center", "center"), 2005: (0.18, "left", "center"), 2010: (0.18, "center", "center"), 2015: (0.18, "right", "center"), 2020: (0.18, "center", "center"), 2024: (0.22, "left", "center")}
        for year, (offset, ha, va) in year_offsets.items():
            theta_label, radius_label = spiral_position(float(year))
            label_radius = radius_label + offset
            add_text(ax, label_radius * np.cos(theta_label), label_radius * np.sin(theta_label), str(year), ha=ha, va=va, fontsize=9)

        annual_total_pct = (global_max_total if global_max_total is not None else annual[DOMAIN_ORDER].clip(lower=0).sum(axis=1).max()) * 100
        add_text(ax, -2.72, 2.22, f"{region}\nNational-model split", ha="left", va="top", fontsize=9.2, fontweight="bold")
        add_text(ax, -2.72, 1.90, f"0 to {annual_total_pct:.1f}%", ha="left", va="top", fontsize=7.6)
        legend_rows = [(["Heat", "Cold", "Severe weather", "Forage"], -2.50), (["Feed market", "Dairy market", "Market demand"], -2.68)]
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


def region_slug(region: str) -> str:
    return region.lower().replace(" ", "_")


def plot_region(region: str, annual: pd.DataFrame, global_max_total: float) -> None:
    center = make_spiral_data(annual)
    points = TAB / f"point2_milk_per_cow_expanded_nonredundant_{region_slug(region)}_national_model_region_split_yearly_adjusted_incremental_r2_spiral_points.csv"
    center.to_csv(points, index=False)

    stem = f"main_point2_milk_per_cow_expanded_nonredundant_{region_slug(region)}_national_model_region_split_yearly_adjusted_incremental_r2_spiral_style"
    fig, ax = plt.subplots(figsize=(3.433, 3.967), constrained_layout=False)
    fig.patch.set_facecolor("none")
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.065, top=0.885)
    draw_spiral(ax, annual, center, region, show_text=True, global_max_total=global_max_total)
    fig.text(0.035, 0.965, "Milk Per Cow Sensitivity", ha="left", va="top", fontsize=13, fontweight="bold")
    save_figure(fig, stem)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.717, 1.983), constrained_layout=False)
    fig.patch.set_facecolor("none")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    draw_spiral(ax, annual, center, region, show_text=False, global_max_total=global_max_total)
    save_figure(fig, f"{stem}_wo_legend")
    plt.close(fig)
    print(f"Wrote {points}")


def main() -> int:
    long = build_long()
    long.to_csv(OUT_LONG, index=False)
    summary = summarize(long)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote {OUT_LONG}")
    print(f"Wrote {OUT_SUMMARY}")
    global_max_total = summary[DOMAIN_ORDER].clip(lower=0).sum(axis=1).max()
    for region in REGION_ORDER:
        annual = summary.loc[summary["region"].eq(region), ["year", *DOMAIN_ORDER]].copy()
        plot_region(region, annual, float(global_max_total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
