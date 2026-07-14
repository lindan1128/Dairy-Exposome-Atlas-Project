#!/usr/bin/env python3
"""3D circular skyline of yearly single-variable incremental R2.

The base variables are the union of variables selected by year-specific
redundancy pruning. Each retained variable is placed once on the circular base
and grouped by subdomain. For each year from 2000 to 2025, a vertical skyline
line shows that variable's weighted cross-sectional single-variable
incremental R2 for annual per-cow milk yield.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import math
import re
import sys
import textwrap

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import art3d, proj3d  # noqa: F401  needed by 3D projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.tri import Triangulation

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "anchor_variable_network",
    SCRIPT_DIR / "68_build_anchor_year_variable_cooccurrence_network.py",
)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

TAB4 = base.TAB4
FIG4 = base.STAT / "4" / "figures"
YEARS = list(range(2000, 2026))
base.ANCHOR_YEARS = YEARS
YEAR_LABEL_THETA_OFFSET_DEG = -7.0
YEAR_LABEL_RADIUS_OFFSET = 0.0
YEAR_LABEL_CORRIDOR = True
YEAR_LABEL_CORRIDOR_LW = 8.5
ALIGN_AXES_TO_SUBDOMAIN_GAPS = True
sys.path.insert(0, str(base.STAT))
import lib_statistics_panel as L  # noqa: E402

BREED_PATH = (
    base.ROOT
    / "data"
    / "us_milk"
    / "processed"
    / "genomics"
    / "state_year_adaptive_heat_genetic_index_webconnect_enriched_2003_2020.csv"
)
BREED_BASELINE_COLS = ["cdcb_breed_heat_background_state_z"]
HERD_SCALE_BASELINE_COLS = ["log_milk_cows_head_baseline"]
SUPP2_CLEAN_EXWAS_PATH = (
    base.ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
BASELINE_SKYLINES = [
    {
        "suffix": "herd_breed_adjusted",
        "label": "Herd scale + breed context adjusted",
        "include_region": False,
        "covariates": BREED_BASELINE_COLS + HERD_SCALE_BASELINE_COLS,
    },
]

DOMAIN_ORDER = [
    "Heat",
    "Cold",
    "Severe weather",
    "Forage",
    "Pesticides",
    "Feed market",
    "Dairy market",
    "Market demand",
    "Herd scale",
    "Dairy scale",
]

DOMAIN_BASE = {
    "Heat": "#32a4b4",
    "Cold": "#33c5b2",
    "Severe weather": "#d5eada",
    "Forage": "#1D7B8D",
    "Pesticides": "#c79fa8",
    "Feed market": "#fbc4ab",
    "Dairy market": "#E47666",
    "Market demand": "#f09d51",
    "Herd scale": "#f6a04d",
    "Dairy scale": "#fec89a",
}

CLASS_BASE = {
    "Nature and climate": "#60BFA4",
    "Forage and pasture condition": "#1D7B8D",
    "Chemical and pollution exposome": "#c79fa8",
    "Market and production-system": "#F06F26",
    "Epidemic and infectious shocks": "#deab90",
}


def lighten(color: str, amount: float) -> str:
    rgb = np.array(mpl.colors.to_rgb(color))
    if amount >= 0:
        out = rgb + (1 - rgb) * amount
    else:
        out = rgb * (1 + amount)
    return mpl.colors.to_hex(np.clip(out, 0, 1))


def depth_shade(color: str, y: float, y_min: float, y_max: float) -> str:
    """Darken far/back values and lighten front values to improve 3D reading."""
    if not np.isfinite(y) or y_max <= y_min:
        return color
    depth = (y - y_min) / (y_max - y_min)
    # In this view, larger y reads as farther/back. Keep the front airy and
    # make the back saturated, like the reference skyline.
    amount = 0.44 * (1 - depth) - 0.34 * depth
    return lighten(color, float(amount))


def make_subdomain_colors(vars_df: pd.DataFrame) -> dict[str, str]:
    colors = {}
    for domain in DOMAIN_ORDER:
        subs = (
            vars_df.loc[vars_df["domain_label"].eq(domain), "subdomain_label"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not subs:
            continue
        offsets = np.linspace(-0.28, 0.34, len(subs)) if len(subs) > 1 else np.array([0.0])
        for sub, off in zip(subs, offsets):
            colors[sub] = lighten(DOMAIN_BASE[domain], float(off))
    return colors


def zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    sd = np.nanstd(arr, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(arr)
    return (arr - np.nanmean(arr)) / sd


def weighted_single_beta_p_incremental_r2(
    d: pd.DataFrame,
    exposure: str,
    include_region: bool,
    baseline_covariates: list[str],
) -> tuple[float, float, float, int, float, float]:
    cols = ["milk_per_cow_lb", "milk_cows_head", "region"] + baseline_covariates + [exposure]
    use = d[cols].replace([np.inf, -np.inf], np.nan).copy()
    use = use.dropna(subset=["milk_per_cow_lb", "milk_cows_head", "region"] + baseline_covariates)
    if len(use) < 8:
        return np.nan, np.nan, np.nan, int(len(use)), np.nan, np.nan

    y = zscore(np.log(pd.to_numeric(use["milk_per_cow_lb"], errors="coerce").to_numpy(float)))
    w = pd.to_numeric(use["milk_cows_head"], errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    use = use.loc[ok].copy()
    y = y[ok]
    w = w[ok]
    if len(use) < 8 or np.nanstd(y) <= 1e-10:
        return np.nan, np.nan, np.nan, int(len(use)), np.nan, np.nan

    pieces = [pd.Series(1.0, index=use.index, name="intercept")]
    if include_region:
        pieces.append(pd.get_dummies(use["region"], prefix="region", drop_first=True, dtype=float))
    for covariate in baseline_covariates:
        x_cov = pd.to_numeric(use[covariate], errors="coerce").to_numpy(float)
        z_cov = zscore(x_cov)
        if np.nanstd(z_cov) > 1e-10:
            pieces.append(pd.Series(z_cov, index=use.index, name=covariate))
    X0 = pd.concat(pieces, axis=1).to_numpy(float)

    x_ser = pd.to_numeric(use[exposure], errors="coerce")
    x_med = x_ser.median(skipna=True)
    x = x_ser.fillna(x_med if np.isfinite(x_med) else 0).to_numpy(float)
    xx = zscore(x)
    if np.nanstd(xx) <= 1e-10:
        return np.nan, np.nan, np.nan, int(len(use)), np.nan, np.nan
    X1 = np.column_stack([X0, xx])

    ww = w / np.nanmean(w)
    sw = np.sqrt(ww)
    y_w = y * sw
    X0_w = X0 * sw[:, None]
    X1_w = X1 * sw[:, None]
    beta0 = np.linalg.pinv(X0_w.T @ X0_w) @ (X0_w.T @ y_w)
    beta1 = np.linalg.pinv(X1_w.T @ X1_w) @ (X1_w.T @ y_w)
    resid0 = y_w - X0_w @ beta0
    resid1 = y_w - X1_w @ beta1
    sse0 = float(resid0 @ resid0)
    sse1 = float(resid1 @ resid1)
    sst = float(((y_w - np.average(y_w)) ** 2).sum())
    base_r2 = max(0.0, 1 - sse0 / sst) if sst > 0 else np.nan
    full_r2 = max(0.0, 1 - sse1 / sst) if sst > 0 else np.nan
    incr_r2 = max(0.0, full_r2 - base_r2) if np.isfinite(base_r2) and np.isfinite(full_r2) else np.nan
    beta = float(beta1[-1])

    df = len(y) - X1.shape[1]
    if df > 0:
        mse = sse1 / df
        cov = mse * np.linalg.pinv(X1_w.T @ X1_w)
        se = math.sqrt(max(cov[-1, -1], 0))
        t = beta / se if se > 0 else np.nan
        try:
            from scipy import stats

            p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else np.nan
        except Exception:
            p = np.nan
    else:
        p = np.nan
    return beta, p, incr_r2, int(len(use)), base_r2, full_r2


def load_clean_curated_macro_exwas_variables() -> pd.DataFrame:
    """Load the full four-class Supplementary Data 2 ExWAS variable set."""
    class_order = [
        "Nature and climate",
        "Forage and pasture condition",
        "Chemical and pollution exposome",
        "Market and production-system",
    ]
    domain_map = {
        "Agricultural pesticides": "Pesticides",
        "Forage condition": "Forage",
    }
    vars_df = pd.read_excel(SUPP2_CLEAN_EXWAS_PATH, sheet_name="exwas_variables")
    vars_df = vars_df[
        vars_df["class"].isin(class_order)
        & vars_df["used_in_exwas"].astype(bool)
    ].copy()
    vars_df["source_class"] = vars_df["class"]
    vars_df["domain_label"] = vars_df["domain"].replace(domain_map)
    vars_df["subdomain_label"] = vars_df["Subdomain"]
    vars_df["exposure"] = vars_df["variables_en"]
    vars_df["exposure_zh"] = vars_df["variables_ch"]
    vars_df = vars_df[~vars_df["subdomain_label"].eq("Hay condition")].copy()
    vars_df["domain_label"] = pd.Categorical(vars_df["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    vars_df["_class_order"] = pd.Categorical(vars_df["source_class"], categories=class_order, ordered=True)
    vars_df = vars_df.sort_values(
        ["_class_order", "domain_label", "subdomain_label", "exposure"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return vars_df.drop(columns=["_class_order"])


def load_skyline_panel(vars_df: pd.DataFrame) -> pd.DataFrame:
    annual_y = base.load_annual_percow()
    annual_x = base.load_annual_exposure_matrix(vars_df)
    panel = annual_y.merge(annual_x, on=["state_alpha", "year"], how="inner")
    panel["region"] = panel["state_alpha"].map(L.CENSUS_REGION)
    panel["log_milk_cows_head_baseline"] = np.log(pd.to_numeric(panel["milk_cows_head"], errors="coerce"))
    if BREED_PATH.exists():
        breed = pd.read_csv(BREED_PATH, low_memory=False)
        keep = ["state_alpha", "year"] + [c for c in BREED_BASELINE_COLS if c in breed.columns]
        panel = panel.merge(
            breed[keep].drop_duplicates(["state_alpha", "year"]),
            on=["state_alpha", "year"],
            how="left",
        )
    return panel


def resolve_year_covariates(d: pd.DataFrame, covariates: list[str]) -> list[str]:
    """Use breed context when available; otherwise fall back to herd-scale only."""
    resolved = []
    for covariate in covariates:
        if covariate not in d.columns:
            continue
        values = pd.to_numeric(d[covariate], errors="coerce")
        if values.notna().sum() >= 8 and values.nunique(dropna=True) > 1:
            resolved.append(covariate)
    return resolved


def compute_yearly_baseline_adjusted_r2(
    vars_df: pd.DataFrame,
    baseline: dict,
    output_suffix: str | None = None,
) -> pd.DataFrame:
    panel = load_skyline_panel(vars_df)
    rows = []
    for year in YEARS:
        d = panel[panel["year"].eq(year)].copy()
        year_covariates = resolve_year_covariates(d, list(baseline["covariates"]))
        for row in vars_df.itertuples(index=False):
            exposure = row.exposure
            if exposure not in d.columns:
                beta, p, incr_r2, n, base_r2, full_r2 = np.nan, np.nan, np.nan, 0, np.nan, np.nan
            else:
                beta, p, incr_r2, n, base_r2, full_r2 = weighted_single_beta_p_incremental_r2(
                    d,
                    exposure,
                    include_region=bool(baseline["include_region"]),
                    baseline_covariates=year_covariates,
                )
            rows.append(
                {
                    "year": year,
                    "exposure": exposure,
                    "exposure_zh": getattr(row, "exposure_zh", ""),
                    "domain_label": row.domain_label,
                    "subdomain_label": row.subdomain_label,
                    "beta": beta,
                    "p": p,
                    "single_variable_delta_r2": incr_r2,
                    "single_variable_delta_r2_pct": 100 * incr_r2 if np.isfinite(incr_r2) else np.nan,
                    "baseline_r2": base_r2,
                    "full_r2": full_r2,
                    "n_states": n,
                    "baseline_label": baseline["label"],
                    "baseline_suffix": baseline["suffix"],
                    "include_region_baseline": bool(baseline["include_region"]),
                    "baseline_covariates": ";".join(year_covariates),
                }
            )
    out = pd.DataFrame(rows)
    suffix = output_suffix or baseline["suffix"]
    out.to_csv(
        TAB4 / f"point4_nonredundant_variable_yearly_single_r2_skyline_{suffix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def short_label(x: str) -> str:
    x = x.replace("_", " ")
    x = x.replace("dairy weighted", "dairy-wt")
    x = x.replace("temperature", "temp")
    x = x.replace("chem pesticide", "pesticide")
    return "\n".join(textwrap.wrap(x, width=16)[:2])


def plot_skyline(
    vars_df: pd.DataFrame,
    r2: pd.DataFrame,
    output_stem: str = "main_point4_nonredundant_variable_yearly_r2_skyline",
    baseline_label: str = "Intercept baseline",
    text_free: bool = False,
) -> None:
    target_main_skyline = output_stem.startswith(
        "main_point4_nonredundant_variable_yearly_r2_skyline_herd_breed_adjusted"
    )
    target_wo_legend = (
        output_stem.startswith(
            "main_point4_nonredundant_variable_yearly_r2_skyline_herd_breed_adjusted_wo_legend"
        )
    )
    vars_df = vars_df.copy().reset_index(drop=True)
    sub_cols = make_subdomain_colors(vars_df)
    if target_main_skyline:
        sub_cols.update(
            {
                "Composite forage condition": "#1D7B8D",
                "Pasture condition": "#1D7B8D",
            }
        )
    camera_elev = 46.5 if target_wo_legend else 47
    camera_azim = -125 if target_main_skyline else (-125 if text_free else -58)
    visual_front_theta = np.deg2rad(camera_azim)
    front_axis_theta = visual_front_theta + (
        np.deg2rad(float(globals().get("YEAR_LABEL_THETA_OFFSET_DEG", 0.0))) if text_free else 0.0
    )
    layout_front_theta = visual_front_theta
    visual_back_theta = np.deg2rad(camera_azim + 180)
    layout_back_theta = np.deg2rad(122)

    # Add visible gaps between subdomains on the circular base. For the
    # axis-only version keep the original skyline grouping, but rotate the
    # unavoidable circular seam/gap to the front year-axis spoke.
    gap = 1.75
    domain_gap = 3.4 if text_free and bool(globals().get("ALIGN_AXES_TO_SUBDOMAIN_GAPS", False)) else gap
    axis_gap = 9.0 if text_free and bool(globals().get("ALIGN_AXES_TO_SUBDOMAIN_GAPS", False)) else domain_gap

    def angular_distance(a, b):
        return abs(np.angle(np.exp(1j * (a - b))))

    def signed_angle_to(a, b):
        return float(np.angle(np.exp(1j * (b - a))))

    def boundary_sequence(use_axis_gaps: bool = False, axis_ids: set[int] | None = None):
        positions_out = []
        boundaries_out = []
        cursor_out = 0.0
        last_sub_out = None
        last_domain_out = None
        boundary_id = 0
        for row_out in vars_df.itertuples(index=False):
            if last_sub_out is not None and row_out.subdomain_label != last_sub_out:
                is_domain_boundary = row_out.domain_label != last_domain_out
                gap_width = domain_gap if is_domain_boundary else gap
                if use_axis_gaps and axis_ids and boundary_id in axis_ids:
                    gap_width = axis_gap
                boundaries_out.append(
                    {
                        "id": boundary_id,
                        "center": cursor_out + gap_width / 2,
                        "width": gap_width,
                        "is_domain_boundary": is_domain_boundary,
                        "left_domain": last_domain_out,
                        "right_domain": row_out.domain_label,
                        "left_subdomain": last_sub_out,
                        "right_subdomain": row_out.subdomain_label,
                    }
                )
                cursor_out += gap_width
                boundary_id += 1
            positions_out.append(cursor_out)
            cursor_out += 1.0
            last_sub_out = row_out.subdomain_label
            last_domain_out = row_out.domain_label
        return positions_out, boundaries_out, cursor_out

    positions, gap_records, total = boundary_sequence()
    axis_gap_ids: set[int] = set()
    if text_free and bool(globals().get("ALIGN_AXES_TO_SUBDOMAIN_GAPS", False)) and gap_records:
        candidate_records = [b for b in gap_records if b["is_domain_boundary"]] or gap_records
        candidate_theta = {
            b["id"]: 2 * np.pi * float(b["center"]) / total for b in candidate_records
        }
        best_pair = None
        best_mismatch = float("inf")
        for left in candidate_records:
            for right in candidate_records:
                if left["id"] == right["id"]:
                    continue
                mismatch = angular_distance(candidate_theta[left["id"]] + np.pi, candidate_theta[right["id"]])
                if mismatch < best_mismatch:
                    best_pair = (left["id"], right["id"])
                    best_mismatch = mismatch
        if best_pair is not None:
            axis_gap_ids = set(best_pair)
            positions, gap_records, total = boundary_sequence(use_axis_gaps=True, axis_ids=axis_gap_ids)

    vars_df["base_pos"] = positions

    raw_theta = 2 * np.pi * vars_df["base_pos"] / total
    heat_mask = vars_df["domain_label"].astype(str).eq("Heat")
    if text_free and bool(globals().get("ALIGN_AXES_TO_SUBDOMAIN_GAPS", False)) and gap_records:
        axis_records = [b for b in gap_records if b["id"] in axis_gap_ids] or gap_records
        gap_theta = {b["id"]: 2 * np.pi * float(b["center"]) / total for b in axis_records}
        if len(axis_records) >= 2:
            front_rec, back_rec = min(
                ((a, b) for a in axis_records for b in axis_records if a["id"] != b["id"]),
                key=lambda pair: angular_distance(gap_theta[pair[0]["id"]] + np.pi, gap_theta[pair[1]["id"]]),
            )
            front_gap = gap_theta[front_rec["id"]]
            back_gap = gap_theta[back_rec["id"]]
            mismatch = signed_angle_to(front_gap + np.pi, back_gap)
            rotation_offset = visual_front_theta - (front_gap + mismatch / 2.0)
        else:
            front_gap = gap_theta[axis_records[0]["id"]]
            rotation_offset = visual_front_theta - front_gap
    elif heat_mask.any():
        climate_center = math.atan2(
            float(np.sin(raw_theta[heat_mask]).mean()),
            float(np.cos(raw_theta[heat_mask]).mean()),
        )
        rotation_offset = layout_back_theta - climate_center
    else:
        rotation_offset = 0.0
    if target_main_skyline:
        rotation_offset += np.pi
        front_boundaries = [
            b for b in gap_records
            if b.get("is_domain_boundary")
            and b.get("left_domain") == "Severe weather"
            and b.get("right_domain") == "Forage"
        ]
        if front_boundaries:
            boundary_angle = 2 * np.pi * float(front_boundaries[0]["center"]) / total + rotation_offset
            rotation_offset += float(np.angle(np.exp(1j * (front_axis_theta - boundary_angle))))

    def theta_for_pos(pos: float) -> float:
        return 2 * np.pi * pos / total + rotation_offset

    def two_half_theta_layout(df: pd.DataFrame, front_theta: float) -> list[float]:
        """Place climate on the left half and the other classes on the right half."""
        back_theta = front_theta + np.pi
        back_axis_gap = np.deg2rad(8.0)
        front_left_gap = np.deg2rad(18.0)
        front_right_gap = np.deg2rad(10.0)
        right_back_gap = np.deg2rad(5.0)
        class_order = {
            "Nature and climate": 0,
            "Forage and pasture condition": 1,
            "Chemical and pollution exposome": 2,
            "Market and production-system": 3,
        }
        domain_order = {d: i for i, d in enumerate(DOMAIN_ORDER)}

        def side_positions(
            side_df: pd.DataFrame,
            subdomain_gap: float = 1.15,
            domain_gap: float = 3.2,
        ) -> dict[int, tuple[float, float]]:
            ordered_idx = side_df.index.tolist()
            if not ordered_idx:
                return {}
            pos = {}
            cursor = 0.0
            last_sub = None
            last_domain = None
            for idx in ordered_idx:
                row = df.loc[idx]
                if last_sub is not None and row.subdomain_label != last_sub:
                    cursor += domain_gap if row.domain_label != last_domain else subdomain_gap
                pos[idx] = (cursor, 0.0)
                cursor += 1.0
                last_sub = row.subdomain_label
                last_domain = row.domain_label
            total_side = max(cursor - 1.0, 1.0)
            return {idx: (p, total_side) for idx, (p, _) in pos.items()}

        ordered = df.assign(
            _class_order=df["source_class"].map(class_order).fillna(99),
            _domain_order=df["domain_label"].astype(str).map(domain_order).fillna(99),
        ).sort_values(["_class_order", "_domain_order", "subdomain_label", "base_pos"])
        climate = ordered[ordered["source_class"].eq("Nature and climate")]
        right = ordered[~ordered["source_class"].eq("Nature and climate")]
        climate_pos = side_positions(climate, subdomain_gap=0.35, domain_gap=0.85)
        right_pos = side_positions(right, subdomain_gap=1.15, domain_gap=3.2)
        theta_map = {}
        for idx, (p, side_total) in climate_pos.items():
            theta_map[idx] = back_theta + back_axis_gap + (p / side_total) * (np.pi - back_axis_gap - front_left_gap)
        for idx, (p, side_total) in right_pos.items():
            theta_map[idx] = front_theta + front_right_gap + (p / side_total) * (np.pi - front_right_gap - right_back_gap)
        return [theta_map[i] for i in df.index]

    if target_main_skyline:
        vars_df["theta"] = two_half_theta_layout(vars_df, layout_front_theta)
    else:
        vars_df["theta"] = [theta_for_pos(float(x)) for x in vars_df["base_pos"]]

    plot = r2.merge(vars_df[["exposure", "theta", "base_pos", "source_class"]], on="exposure", how="left")
    r_min, r_max = 0.65, 5.25
    plot["radius"] = r_max - (plot["year"] - min(YEARS)) / (max(YEARS) - min(YEARS)) * (r_max - r_min)
    plot["x"] = plot["radius"] * np.cos(plot["theta"])
    plot["y"] = plot["radius"] * np.sin(plot["theta"])
    plot["z_raw"] = plot["single_variable_delta_r2_pct"].clip(lower=0, upper=65)
    # Keep the terrain continuous through years where a variable has no
    # estimable value. Dropping those rows makes the surface stop before the
    # outer 2000 ring, which visually reads as "no 2000 data" for the whole
    # domain. For plotting geometry, missing/negative incremental R2 belongs on
    # the floor; the vertical spike layer still only draws positive values.
    plot["z"] = plot["z_raw"].fillna(0.0)
    plot["color"] = plot["subdomain_label"].map(sub_cols).fillna("#999999")
    y_min, y_max = float(plot["y"].min()), float(plot["y"].max())
    plot["color_depth"] = [
        depth_shade(c, y, y_min, y_max) for c, y in zip(plot["color"].tolist(), plot["y"].to_numpy(float))
    ]
    terrain_edge_alpha = 0.56 if target_wo_legend else (0.34 if text_free else 0.22)
    terrain_surface_alpha = 0.48 if target_wo_legend else 0.56
    terrain_overlap_surface_alpha = 0.24 if target_wo_legend else terrain_surface_alpha
    outer_wall_alpha = 0.84 if target_wo_legend else 0.72
    outer_wall_edge_alpha = 0.68 if target_wo_legend else 0.42
    vertical_skyline_alpha = 0.76 if target_wo_legend else (0.44 if text_free else 0.55)

    fig = plt.figure(figsize=(8.6, 8.4))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    # The 3D SVG backend clips artists to the axes patch. The outer 2000 ring
    # sits exactly at the projected boundary in the text-free skyline, which
    # can make the first-year surface look truncated rather than merely cropped.
    # Keep a small in-figure gutter so the outer ring and filled panels stay
    # inside the clip rectangle.
    ax.set_position([0.045, 0.045, 0.91, 0.91] if text_free else [0.02, 0.02, 0.96, 0.96])
    ax.view_init(elev=camera_elev, azim=camera_azim)
    ax.set_proj_type("persp", focal_length=0.20)
    ax.set_box_aspect((1, 1, 0.72))
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)

    base_sector_collections = []

    def add_outer_band(group_col: str, color_map: dict[str, str], inner: float, outer: float, z: float, alpha: float) -> None:
        if not target_wo_legend or group_col not in vars_df.columns:
            return
        front_clearance = np.deg2rad(7.5)
        ordered = vars_df.sort_values("base_pos").reset_index(drop=True)
        runs = []
        run_start = 0
        for idx in range(1, len(ordered) + 1):
            if idx == len(ordered) or ordered.loc[idx, group_col] != ordered.loc[run_start, group_col]:
                runs.append(ordered.iloc[run_start:idx])
                run_start = idx
        for g_band in runs:
            label = str(g_band[group_col].iloc[0])
            color = color_map.get(label, "#dddddd")
            if target_wo_legend:
                group_theta = np.sort(np.unwrap(g_band["theta"].to_numpy(float)))
                pad = np.deg2rad(0.8)
                theta0 = float(group_theta[0] - pad)
                theta1 = float(group_theta[-1] + pad)
            else:
                theta0 = theta_for_pos(max(float(g_band["base_pos"].min()) - 0.46, 0))
                theta1 = theta_for_pos(min(float(g_band["base_pos"].max()) + 0.46, total))
            theta_vals = np.linspace(theta0, theta1, 64)
            verts = []
            for a0, a1 in zip(theta_vals[:-1], theta_vals[1:]):
                amid = float(np.angle(np.exp(1j * ((a0 + a1) / 2 - front_axis_theta))))
                if abs(amid) < front_clearance:
                    continue
                verts.append(
                    [
                        (inner * np.cos(a0), inner * np.sin(a0), z),
                        (outer * np.cos(a0), outer * np.sin(a0), z),
                        (outer * np.cos(a1), outer * np.sin(a1), z),
                        (inner * np.cos(a1), inner * np.sin(a1), z),
                    ]
                )
            band = Poly3DCollection(
                verts,
                facecolors=color,
                edgecolors="white",
                linewidths=0.18,
                alpha=alpha,
                zorder=8,
            )
            ax.add_collection3d(band)

    add_outer_band("source_class", CLASS_BASE, r_max + 0.08, r_max + 0.44, 0.028, 1.0)

    ring_projection_points: np.ndarray | None = None
    if target_wo_legend:
        projection = ax.get_proj()
        front_clearance = np.deg2rad(7.5)
        ring_pts = []
        for radius in np.linspace(r_max + 0.08, r_max + 0.44, 4):
            for theta in np.linspace(0, 2 * np.pi, 360, endpoint=False):
                amid = float(np.angle(np.exp(1j * (theta - front_axis_theta))))
                if abs(amid) < front_clearance:
                    continue
                back_distance = abs(np.angle(np.exp(1j * (theta - visual_back_theta))))
                if back_distance > np.deg2rad(115.0):
                    continue
                x2, y2, _ = proj3d.proj_transform(radius * np.cos(theta), radius * np.sin(theta), 0.026, projection)
                ring_pts.append((x2, y2))
        ring_projection_points = np.asarray(ring_pts, dtype=float)

    def visually_overlaps_outer_band(xyz_tri: np.ndarray) -> bool:
        if ring_projection_points is None or len(ring_projection_points) == 0:
            return False
        centroid = np.nanmean(xyz_tri, axis=0)
        x2, y2, _ = proj3d.proj_transform(centroid[0], centroid[1], centroid[2], ax.get_proj())
        d2 = np.sum((ring_projection_points - np.array([x2, y2])) ** 2, axis=1)
        return bool(float(np.nanmin(d2)) < 0.00010)

    # Domain-colored base sectors.
    for domain, g in vars_df.groupby("domain_label", sort=False, observed=True):
        if g.empty:
            continue
        if text_free:
            group_theta = np.sort(np.unwrap(g["theta"].to_numpy(float)))
            pad = np.deg2rad(0.8)
            theta0 = float(group_theta[0] - pad)
            theta1 = float(group_theta[-1] + pad)
        else:
            theta0 = theta_for_pos(max(float(g["base_pos"].min()) - 0.48, 0))
            theta1 = theta_for_pos(min(float(g["base_pos"].max()) + 0.48, total))
        theta_vals = np.linspace(theta0, theta1, 80)
        verts = []
        for a0, a1 in zip(theta_vals[:-1], theta_vals[1:]):
            verts.append(
                [
                    (r_min * np.cos(a0), r_min * np.sin(a0), -0.18),
                    (r_max * np.cos(a0), r_max * np.sin(a0), -0.18),
                    (r_max * np.cos(a1), r_max * np.sin(a1), -0.18),
                    (r_min * np.cos(a1), r_min * np.sin(a1), -0.18),
                ]
            )
        poly = Poly3DCollection(
            verts,
            facecolors=DOMAIN_BASE.get(str(domain), "#dddddd"),
            edgecolors="none",
            alpha=0.11,
            zorder=-5,
        )
        ax.add_collection3d(poly)
        base_sector_collections.append(poly)

    # Soft annual base rings.
    theta_grid = np.linspace(0, 2 * np.pi, 500)
    anchor_years = {2000, 2005, 2010, 2015, 2020, 2025}
    for year in YEARS:
        radius = r_max - (year - min(YEARS)) / (max(YEARS) - min(YEARS)) * (r_max - r_min)
        is_anchor = year in anchor_years
        alpha = 0.45 if is_anchor else 0.06
        lw = 0.50 if is_anchor else 0.14
        ax.plot(
            radius * np.cos(theta_grid),
            radius * np.sin(theta_grid),
            np.zeros_like(theta_grid),
            color="black" if is_anchor else "#bdbdbd",
            alpha=alpha,
            lw=lw,
            zorder=5,
        )

    # Subdomain-colored radial base rays.
    for row in vars_df.itertuples(index=False):
        color = sub_cols.get(row.subdomain_label, "#999999")
        ax.plot(
            [r_min * np.cos(row.theta), r_max * np.cos(row.theta)],
            [r_min * np.sin(row.theta), r_max * np.sin(row.theta)],
            [0, 0],
            color=color,
            alpha=0.045 if text_free else 0.18,
            lw=0.20 if text_free else 0.42,
            zorder=6,
        )

    # Continuous subdomain terrains. These make the plot read as a landscape
    # instead of thousands of independent bristles while still using every
    # year-variable point.
    terrain_collections = []
    if target_wo_legend:
        climate_fill = plot[plot["source_class"].eq("Nature and climate")].dropna(
            subset=["x", "y", "z", "base_pos"]
        ).copy()
        if (
            climate_fill["exposure"].nunique() >= 3
            and climate_fill["year"].nunique() >= 3
            and len(climate_fill) >= 12
        ):
            try:
                tri = Triangulation(
                    climate_fill["base_pos"].to_numpy(float),
                    climate_fill["year"].to_numpy(float),
                )
                xyz = climate_fill[["x", "y", "z"]].to_numpy(float)
                verts = [xyz[t].tolist() for t in tri.triangles]
                fill = Poly3DCollection(
                    verts,
                    facecolors=mpl.colors.to_rgba("#8fd9cf", 0.20),
                    edgecolors="none",
                    zorder=24,
                )
                ax.add_collection3d(fill)
                terrain_collections.append(fill)
            except Exception:
                pass

    subdomain_groups = [(subdomain, g) for subdomain, g in plot.groupby("subdomain_label", sort=False)]
    if target_wo_legend:
        front_vec = np.array([np.cos(visual_front_theta), np.sin(visual_front_theta)])

        def subdomain_frontness(item):
            _, g_depth = item
            xy = g_depth[["x", "y"]].dropna().to_numpy(float)
            if len(xy) == 0:
                return 0.0
            return float(np.nanmean(xy @ front_vec))

        subdomain_groups = sorted(subdomain_groups, key=subdomain_frontness)

    for draw_i, (subdomain, g) in enumerate(subdomain_groups):
        g = g.dropna(subset=["x", "y", "z", "base_pos"]).copy()
        if g["exposure"].nunique() < 2 or g["year"].nunique() < 3 or len(g) < 8:
            continue
        draw_zorder = 30 + draw_i * 0.05 if target_wo_legend else 30
        local_x = g["base_pos"].to_numpy(float)
        local_y = g["year"].to_numpy(float)
        try:
            xyz = g[["x", "y", "z"]].to_numpy(float)
            tri = Triangulation(local_x, local_y)
            verts = [xyz[t].tolist() for t in tri.triangles]
            facecols = []
            for t in tri.triangles:
                base_color = depth_shade(sub_cols.get(subdomain, "#999999"), float(np.nanmean(xyz[t, 1])), y_min, y_max)
                alpha_face = terrain_overlap_surface_alpha if target_wo_legend and visually_overlaps_outer_band(xyz[t]) else terrain_surface_alpha
                facecols.append(mpl.colors.to_rgba(base_color, alpha_face))
            edgecols = [
                mpl.colors.to_rgba(
                    depth_shade(sub_cols.get(subdomain, "#999999"), float(np.nanmean(xyz[t, 1])), y_min, y_max),
                    terrain_overlap_surface_alpha if target_wo_legend and visually_overlaps_outer_band(xyz[t]) else terrain_edge_alpha,
                )
                for t in tri.triangles
            ]
            surf = Poly3DCollection(
                verts,
                facecolors=facecols,
                edgecolors=edgecols,
                linewidths=0.16 if text_free else 0.12,
                zorder=draw_zorder,
            )
            ax.add_collection3d(surf)
            terrain_collections.append(surf)

            outer = g[g["year"].eq(min(YEARS))].sort_values("base_pos")
            if len(outer) >= 2:
                outer_xyz = outer[["x", "y", "z"]].to_numpy(float)
                floor_z = -0.12 if text_free else 0.0
                wall_verts = []
                wall_facecols = []
                wall_edgecols = []
                for p0, p1 in zip(outer_xyz[:-1], outer_xyz[1:]):
                    wall_verts.append(
                        [
                            (p0[0], p0[1], floor_z),
                            (p1[0], p1[1], floor_z),
                            (p1[0], p1[1], p1[2]),
                            (p0[0], p0[1], p0[2]),
                        ]
                    )
                    wall_color = depth_shade(
                        sub_cols.get(subdomain, "#999999"),
                        float(np.nanmean([p0[1], p1[1]])),
                        y_min,
                        y_max,
                    )
                    wall_facecols.append(mpl.colors.to_rgba(wall_color, outer_wall_alpha))
                    wall_edgecols.append(mpl.colors.to_rgba(wall_color, outer_wall_edge_alpha))
                wall = Poly3DCollection(
                    wall_verts,
                    facecolors=wall_facecols,
                    edgecolors=wall_edgecols,
                    linewidths=0.18 if text_free else 0.12,
                    zorder=draw_zorder + 0.01,
                )
                ax.add_collection3d(wall)
                terrain_collections.append(wall)

            if target_wo_legend:
                g_lines = g.sort_values(["y", "year", "base_pos"], ascending=[False, True, True])
                for rec in g_lines.itertuples(index=False):
                    if not np.isfinite(rec.z) or rec.z <= 0:
                        continue
                    ax.plot(
                        [rec.x, rec.x],
                        [rec.y, rec.y],
                        [-0.16 if text_free else 0, rec.z],
                        color=rec.color_depth,
                        alpha=vertical_skyline_alpha,
                        lw=0.34 if text_free else 0.45,
                        solid_capstyle="round",
                        zorder=draw_zorder + 0.02,
                    )
        except Exception:
            continue

    for poly in base_sector_collections:
        poly.set_zorder(-20)
    if not target_wo_legend:
        for surf in terrain_collections:
            surf.set_zorder(30)

    # Faint vertical yearly R2 skyline. All variables are still shown, but the
    # surface above carries the visual structure.
    if not target_wo_legend:
        plot_sorted = plot.sort_values(["y", "year", "base_pos"], ascending=[False, True, True])
        for rec in plot_sorted.itertuples(index=False):
            if not np.isfinite(rec.z) or rec.z <= 0:
                continue
            ax.plot(
                [rec.x, rec.x],
                [rec.y, rec.y],
                [-0.16 if text_free else 0, rec.z],
                color=rec.color_depth,
                alpha=vertical_skyline_alpha,
                lw=0.34 if text_free else 0.45,
                solid_capstyle="round",
                zorder=5,
            )

    if not text_free:
        # Subdomain labels around the outside.
        sub_lab = (
            vars_df.groupby(["domain_label", "subdomain_label"], observed=True)
            .agg(theta=("theta", "mean"), n=("exposure", "size"))
            .reset_index()
        )
        for row in sub_lab.itertuples(index=False):
            angle_deg = np.degrees(row.theta)
            r_lab = r_max + 0.42
            x, y = r_lab * np.cos(row.theta), r_lab * np.sin(row.theta)
            color = sub_cols.get(row.subdomain_label, "#555555")
            ax.text(
                x,
                y,
                0,
                short_label(row.subdomain_label),
                color=color,
                fontsize=10.0,
                fontfamily="Arial",
                ha="center",
                va="center",
                rotation=angle_deg,
                rotation_mode="anchor",
                zorder=10,
            )

    # Year labels on the front-left spoke. Keep these in the axis-only version.
    label_theta = front_axis_theta
    label_radius_offset = float(globals().get("YEAR_LABEL_RADIUS_OFFSET", 0.0))
    if target_wo_legend:
        rr = np.linspace(r_min + label_radius_offset, r_max + label_radius_offset, 90)
        xx = rr * np.cos(label_theta)
        yy = rr * np.sin(label_theta)
        zz = np.repeat(0.055, len(rr))
        ax.plot(xx, yy, zz, color="black", alpha=0.72, lw=0.55, solid_capstyle="round", zorder=76)
        tangent = np.array([-np.sin(label_theta), np.cos(label_theta), 0.0])
        tick_len = 0.16
        for year in [2000, 2005, 2010, 2015, 2020, 2025]:
            tick_radius = r_max - (year - min(YEARS)) / (max(YEARS) - min(YEARS)) * (r_max - r_min)
            tick_center = np.array(
                [
                    (tick_radius + label_radius_offset) * np.cos(label_theta),
                    (tick_radius + label_radius_offset) * np.sin(label_theta),
                    0.055,
                ]
            )
            p0 = tick_center - tangent * tick_len / 2
            p1 = tick_center + tangent * tick_len / 2
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color="black", alpha=0.78, lw=0.45, zorder=77)
    if text_free and bool(globals().get("YEAR_LABEL_CORRIDOR", False)):
        rr = np.linspace(r_min + label_radius_offset * 0.65, r_max + label_radius_offset + 0.14, 60)
        xx = rr * np.cos(label_theta)
        yy = rr * np.sin(label_theta)
        zz = np.repeat(0.06, len(rr))
        ax.plot(xx, yy, zz, color="white", alpha=0.88, lw=YEAR_LABEL_CORRIDOR_LW, solid_capstyle="round", zorder=70)
        ax.plot(xx, yy, zz + 0.002, color="black", alpha=0.45, lw=0.35, solid_capstyle="round", zorder=71)
    for year in [2000, 2005, 2010, 2015, 2020, 2025]:
        radius = r_max - (year - min(YEARS)) / (max(YEARS) - min(YEARS)) * (r_max - r_min)
        label_radius = radius + label_radius_offset
        ax.text(
            label_radius * np.cos(label_theta),
            label_radius * np.sin(label_theta),
            0,
            str(year),
            fontsize=10 if text_free else 8,
            fontfamily="Arial",
            color="black",
            ha="center",
            va="center",
            rotation=90 if text_free else 0,
            rotation_mode="anchor",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.12) if text_free and bool(globals().get("YEAR_LABEL_CORRIDOR", False)) else None,
            zorder=80,
        )
    ztop = 50
    zticks = list(range(0, ztop + 1, 10))
    zmax = ztop * 1.04
    ax.set_zlim(-0.18 if text_free else 0, zmax)
    axis_pad = 6.85 if text_free else 6.25
    ax.set_xlim(-axis_pad, axis_pad)
    ax.set_ylim(-axis_pad, axis_pad)
    ax.set_axis_off()

    if text_free:
        # Axis-only version: draw a real 3D z-axis on the circular floor.
        # This keeps the scale physically merged with the base instead of
        # looking like an external overlay.
        axis_theta = visual_back_theta
        axis_radius = r_max
        axis_x = axis_radius * np.cos(axis_theta)
        axis_y = axis_radius * np.sin(axis_theta)
        tangent = np.array([-np.sin(axis_theta), np.cos(axis_theta), 0.0])
        tick_len = 0.20
        label_offset = tangent * -0.48
        title_offset = tangent * -1.18
        ax.plot([axis_x, axis_x], [axis_y, axis_y], [0, ztop], color="black", lw=0.8, alpha=0.95, zorder=260)
        for tick in zticks:
            p0 = np.array([axis_x, axis_y, tick]) - tangent * tick_len / 2
            p1 = np.array([axis_x, axis_y, tick]) + tangent * tick_len / 2
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [tick, tick], color="black", lw=0.5, zorder=261)
            ax.text(
                axis_x + label_offset[0],
                axis_y + label_offset[1],
                tick,
                str(tick),
                fontsize=10,
                fontfamily="Arial",
                color="black",
                ha="center",
                va="center",
                rotation=0,
                zorder=262,
            )
        ax.text(
            axis_x + title_offset[0],
            axis_y + title_offset[1],
            ztop * 0.52,
            "" if target_wo_legend else "Incremental R² (%)",
            fontsize=10,
            fontfamily="Arial",
            color="black",
            ha="center",
            va="center",
            rotation=0,
            zorder=262,
        )
    else:
        # Hand-drawn z axis. This avoids the large 3D frame while retaining a
        # visible height scale.
        axis_x, axis_y = -5.58, -5.28
        ax.plot([axis_x, axis_x], [axis_y, axis_y], [0, ztop], color="black", lw=0.7, alpha=0.9)
        for tick in zticks:
            ax.plot([axis_x - 0.12, axis_x + 0.12], [axis_y, axis_y], [tick, tick], color="black", lw=0.45)
            ax.text(axis_x - 0.35, axis_y, tick, str(tick), fontsize=7.5, color="black", ha="right", va="center")
        ax.text(
            axis_x - 0.72,
            axis_y,
            ztop * 0.52,
            "Single-variable\nincremental R² (%)",
            fontsize=8.0,
            fontfamily="Arial",
            color="black",
            ha="center",
            va="center",
            rotation=90,
        )

    if not text_free:
        fig.text(
            0.5,
            0.020,
            f"Base: union of year-specific nonredundant variables; rings: outer 2000 to inner 2025; height: yearly single-variable ΔR² beyond {baseline_label}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontfamily="Arial",
            color="black",
        )

    out = FIG4 / f"{output_stem}.svg"
    for artist in ax.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.04, transparent=True)
    if target_wo_legend:
        # Matplotlib writes SVG text sizes as CSS px. A 9 pt label becomes
        # visibly smaller in SVG editors unless converted to the 96 dpi px
        # equivalent, so keep the PNG at 9 pt and postprocess the target SVG.
        svg_text = out.read_text(encoding="utf-8")
        svg_text = re.sub(r"font: 9px 'Arial'", "font: 12px 'Arial'", svg_text)
        out.write_text(svg_text, encoding="utf-8")
    fig.savefig(FIG4 / f"{output_stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)


def main() -> int:
    vars_df = load_clean_curated_macro_exwas_variables()
    baseline = BASELINE_SKYLINES[0]
    r2 = compute_yearly_baseline_adjusted_r2(
        vars_df,
        baseline,
        output_suffix=f"{baseline['suffix']}_clean_macro_exwas_full",
    )
    plot_skyline(
        vars_df,
        r2,
        output_stem=f"main_point4_nonredundant_variable_yearly_r2_skyline_{baseline['suffix']}",
        baseline_label=baseline["label"],
    )
    plot_skyline(
        vars_df,
        r2,
        output_stem=f"main_point4_nonredundant_variable_yearly_r2_skyline_{baseline['suffix']}_wo_legend",
        baseline_label=baseline["label"],
        text_free=True,
    )
    print(
        "Wrote herd/breed adjusted skyline figures:",
        FIG4 / f"main_point4_nonredundant_variable_yearly_r2_skyline_{baseline['suffix']}.svg",
        FIG4 / f"main_point4_nonredundant_variable_yearly_r2_skyline_{baseline['suffix']}_wo_legend.svg",
    )
    print(f"Variables from Supplementary Data 2 four-class macro ExWAS set: {vars_df['exposure'].nunique()}")
    top = (
        r2.sort_values("single_variable_delta_r2_pct", ascending=False)
        .head(10)[["year", "domain_label", "subdomain_label", "exposure", "single_variable_delta_r2_pct"]]
    )
    print(top.round({"single_variable_delta_r2_pct": 2}).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
