#!/usr/bin/env python3
"""Draw one four-region US map using the Point 0 50-state map geometry."""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from bokeh.sampledata.us_states import data as US_STATES
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
from shapely.ops import unary_union


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
FIG = POINT / "figures"
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402


REGION_ORDER = ["Northeast", "Midwest", "South", "West"]
REGION_COLORS = {
    "Northeast": "#98b4ce",
    "Midwest": "#d2d0dc",
    "South": "#f2d9be",
    "West": "#f0e4d2",
}
OTHER_FILL = "#FFFFFF"
INSET_EDGE = "#B8B8B8"
US_OUTLINE_EDGE = "#B8B8B8"
HIGHLIGHT_STATE_EDGE = "#000000"
LABEL_COLOR = "#171717"
ISO_Y_SCALE = 0.66
ISO_X_SHEAR = 0.055
ISO_Y_ANCHOR = 37.0
BASE_OFFSETS = [(0.25, -1.75, "#D2D2D2", 0.78), (0.48, -2.85, "#BFBFBF", 0.42)]
REGION_OFFSETS = {
    "Northeast": (0.18, 0.06),
    "Midwest": (0.00, 0.12),
    "South": (0.02, -0.18),
    "West": (-0.20, 0.02),
}
LABEL_POSITIONS = {
    "AL": (-84.84, 32.06),
    "AR": (-93.12, 35.74),
    "CT": (-73.19, 41.27),
    "DE": (-68.89, 41.57),
    "GA": (-80.85, 31.14),
    "IL": (-89.13, 39.43),
    "IN": (-85.45, 38.51),
    "KS": (-97.41, 36.67),
    "KY": (-84.84, 38.51),
    "LA": (-93.43, 30.22),
    "MA": (-67.05, 41.88),
    "MD": (-72.57, 39.43),
    "ME": (-67.67, 46.79),
    "MO": (-93.12, 36.97),
    "MS": (-89.13, 32.06),
    "NC": (-72.57, 35.44),
    "NE": (-100.17, 40.65),
    "NH": (-71.35, 44.33),
    "NJ": (-72.57, 37.89),
    "NY": (-77.17, 42.80),
    "OH": (-81.16, 41.57),
    "PA": (-78.09, 39.73),
    "RI": (-68.59, 38.81),
    "SC": (-76.25, 33.90),
    "SD": (-100.17, 44.33),
    "TN": (-86.68, 35.13),
    "VA": (-77.48, 37.89),
    "VT": (-75.33, 46.17),
    "WV": (-81.16, 40.04),
}


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'DejaVu Sans'[^;]*;", r"font-size: \1px; font-family: 'Arial';", svg)
    svg = svg.replace("font-size: 5.4px;", "font-size: 5.40px;")
    svg = svg.replace("font-size: 9px;", "font-size: 9.00px;")
    path.write_text(svg, encoding="utf-8")


def transformed_coords(state: str, lons: list[float], lats: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(lons, dtype=float)
    y = np.asarray(lats, dtype=float)
    if state == "AK":
        x = np.where(x > 0, x - 360.0, x)
        x = (x + 179.0) * 0.35 - 124.0
        y = (y - 51.0) * 0.35 + 24.0
    elif state == "HI":
        x = x + 46.2
        y = y + 4.4
    return x + ISO_X_SHEAR * (y - ISO_Y_ANCHOR), y * ISO_Y_SCALE


def load_state_regions() -> dict[str, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel()
    state_regions = panel[["state_alpha", "region"]].drop_duplicates()
    return dict(zip(state_regions["state_alpha"], state_regions["region"]))


def region_offset(state: str, state_regions: dict[str, str]) -> np.ndarray:
    return np.asarray(REGION_OFFSETS.get(state_regions.get(state), (0.0, 0.0)), dtype=float)


def add_state(ax: plt.Axes, state: str, state_data: dict, state_regions: dict[str, str]) -> None:
    x, y = transformed_coords(state, state_data["lons"], state_data["lats"])
    isnan = np.isnan(x) | np.isnan(y)
    breaks = np.where(isnan)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(x)]
    region = state_regions.get(state)
    is_inset = state in {"AK", "HI"}
    face = REGION_COLORS.get(region, OTHER_FILL)
    offset = region_offset(state, state_regions)
    edge = INSET_EDGE if is_inset else "#2F2F2F"
    linewidth = 0.24 if is_inset else 0.20
    for start, end in zip(starts, ends):
        if end - start < 3:
            continue
        coords = np.column_stack([x[start:end], y[start:end]]) + offset
        ax.add_patch(
            MplPolygon(
                coords,
                closed=True,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                joinstyle="round",
                zorder=1 if is_inset else 2,
            )
        )


def add_floating_base(ax: plt.Axes, state_regions: dict[str, str]) -> None:
    for dx, dy, color, alpha in reversed(BASE_OFFSETS):
        for state in sorted(US_STATES):
            if state == "DC":
                continue
            region_shift = region_offset(state, state_regions)
            for coords in state_parts(state, US_STATES[state]):
                ax.add_patch(
                    MplPolygon(
                        coords + region_shift + np.asarray([dx, dy]),
                        closed=True,
                        facecolor=color,
                        edgecolor="none",
                        linewidth=0,
                        alpha=alpha,
                        joinstyle="round",
                        zorder=0,
                    )
                )


def add_region_outlines(ax: plt.Axes, state_regions: dict[str, str]) -> None:
    for region in REGION_ORDER:
        polygons = []
        for state, state_data in US_STATES.items():
            if state == "DC" or state_regions.get(state) != region:
                continue
            offset = region_offset(state, state_regions)
            for coords in state_parts(state, state_data):
                poly = ShapelyPolygon(coords + offset)
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    polygons.append(poly)
        if not polygons:
            continue
        dissolved = unary_union(polygons)
        geoms = dissolved.geoms if isinstance(dissolved, MultiPolygon) else [dissolved]
        for geom in geoms:
            x, y = geom.exterior.xy
            ax.plot(x, y, color="#000000", linewidth=0.85, solid_joinstyle="round", zorder=6)


def state_parts(state: str, state_data: dict) -> list[np.ndarray]:
    x, y = transformed_coords(state, state_data["lons"], state_data["lats"])
    isnan = np.isnan(x) | np.isnan(y)
    breaks = np.where(isnan)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(x)]
    parts = []
    for start, end in zip(starts, ends):
        if end - start < 3:
            continue
        parts.append(np.column_stack([x[start:end], y[start:end]]))
    return parts


def add_lower48_outline(ax: plt.Axes, state_regions: dict[str, str]) -> None:
    polygons = []
    for state, state_data in US_STATES.items():
        if state in {"AK", "HI", "DC"}:
            continue
        offset = region_offset(state, state_regions)
        for coords in state_parts(state, state_data):
            poly = ShapelyPolygon(coords + offset)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                polygons.append(poly)
    dissolved = unary_union(polygons)
    geoms = dissolved.geoms if isinstance(dissolved, MultiPolygon) else [dissolved]
    for geom in geoms:
        x, y = geom.exterior.xy
        ax.plot(x, y, color=US_OUTLINE_EDGE, linewidth=0.35, solid_joinstyle="round", zorder=5)


def state_geometry(state: str, state_data: dict) -> ShapelyPolygon | MultiPolygon | None:
    polygons = []
    for coords in state_parts(state, state_data):
        poly = ShapelyPolygon(coords)
        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons.append(poly)
    if not polygons:
        return None
    return unary_union(polygons)


def add_state_labels(ax: plt.Axes, state_regions: dict[str, str], target_region: str) -> None:
    for state in sorted(US_STATES):
        if state == "DC" or state_regions.get(state) != target_region:
            continue
        geom = state_geometry(state, US_STATES[state])
        if geom is None:
            continue
        if state in LABEL_POSITIONS:
            x, y = LABEL_POSITIONS[state]
        else:
            point = geom.representative_point()
            x, y = point.x, point.y
        ax.text(
            x,
            y,
            state,
            ha="center",
            va="center",
            fontsize=9,
            color=LABEL_COLOR,
            zorder=6,
        )


def draw_region_map(state_regions: dict[str, str]) -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    add_floating_base(ax, state_regions)
    for state in sorted(US_STATES):
        if state == "DC":
            continue
        add_state(ax, state, US_STATES[state], state_regions)
    add_lower48_outline(ax, state_regions)
    add_region_outlines(ax, state_regions)

    all_parts = [
        coords + region_offset(state, state_regions)
        for state, state_data in US_STATES.items()
        if state != "DC"
        for coords in state_parts(state, state_data)
    ]
    all_xy = np.vstack(all_parts + [coords + np.asarray([dx, dy]) for dx, dy, _, _ in BASE_OFFSETS for coords in all_parts])
    pad_x = (float(np.nanmax(all_xy[:, 0])) - float(np.nanmin(all_xy[:, 0]))) * 0.025
    pad_y = (float(np.nanmax(all_xy[:, 1])) - float(np.nanmin(all_xy[:, 1]))) * 0.08
    ax.set_xlim(float(np.nanmin(all_xy[:, 0])) - pad_x, float(np.nanmax(all_xy[:, 0])) + pad_x)
    ax.set_ylim(float(np.nanmin(all_xy[:, 1])) - pad_y, float(np.nanmax(all_xy[:, 1])) + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    out = FIG / "main_point2_region_highlight_us_map.svg"
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    normalize_svg_text_style(out)
    print(f"Wrote {out}")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    state_regions = load_state_regions()
    for region in REGION_ORDER:
        old = FIG / f"point2_region_highlight_{region.lower().replace(' ', '_')}_us_map.svg"
        old.unlink(missing_ok=True)
    draw_region_map(state_regions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
