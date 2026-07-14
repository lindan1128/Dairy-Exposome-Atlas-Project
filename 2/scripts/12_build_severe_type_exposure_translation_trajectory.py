#!/usr/bin/env python3
"""Severe-weather natural-disaster type exposure-to-loss translation trajectories."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

TAB = POINT / "tables"
US_MILK = ROOT / "data" / "us_milk" / "processed"
US_EXPOSE_NEW = ROOT / "data" / "us_expose_new" / "processed"
CLEAN_DICT = (
    ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
KEY = ["state_alpha", "year", "month"]

OUT_AUDIT = TAB / "point2_severe_type_variable_audit.csv"
OUT_TRAJ = TAB / "point2_severe_type_exposure_translation_trajectory.csv"
OUT_TREND = TAB / "point2_severe_type_exposure_translation_trend_summary.csv"


TYPE_MAP = {
    "storm_heat_events": ("Heat events", "heat event count"),
    "storm_cold_events": ("Cold/winter/ice", "cold event count"),
    "storm_winter_events": ("Cold/winter/ice", "winter weather event count"),
    "storm_ice_events": ("Cold/winter/ice", "ice event count"),
    "storm_flood_events": ("Flood events", "flood event count"),
    "storm_fire_events": ("Fire events", "fire event count"),
    "storm_wind_events": ("Wind events", "wind event count"),
    "storm_hail_events": ("Hail events", "hail event count"),
}

STRICT_REPRESENTATIVE_TYPES = {
    "storm_heat_events",
    "storm_cold_events",
    "storm_flood_events",
    "storm_fire_events",
    "storm_wind_events",
    "storm_hail_events",
}


def load_panel() -> pd.DataFrame:
    milk = pd.read_csv(US_MILK / "state_month_panel.csv", low_memory=False)
    exp = pd.read_csv(US_EXPOSE_NEW / "exposure_state_month_expanded.csv", low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    return milk.merge(exp, on=KEY, how="left")


def build_variable_audit(panel: pd.DataFrame) -> pd.DataFrame:
    dictionary = pd.read_excel(CLEAN_DICT, sheet_name="exwas_variables")
    dictionary = dictionary[dictionary["used_in_exwas"].fillna(False).astype(bool)].copy()
    severe = dictionary[dictionary["domain"].eq("Severe weather")].copy()
    severe = severe[
        [
            "domain",
            "variables_en",
            "variables_ch",
            "construct",
            "mechanistic_domain_en",
            "form",
            "n_nonmissing_exposure_months",
            "n_states_exposure",
            "year_min_exposure",
            "year_max_exposure",
        ]
    ].drop_duplicates()
    severe["natural_disaster_type"] = severe["variables_en"].map(
        lambda x: TYPE_MAP[x][0] if x in TYPE_MAP else np.nan
    )
    severe["type_note"] = severe["variables_en"].map(
        lambda x: TYPE_MAP[x][1] if x in TYPE_MAP else np.nan
    )
    severe["in_clean_natural_type_events"] = severe["variables_en"].isin(TYPE_MAP)
    severe["available_in_panel"] = severe["variables_en"].isin(panel.columns)

    chord = pd.read_csv(TAB / "point2_heat_severe_subform_variable_summary.csv", low_memory=False)
    chord_vars = set(
        chord.loc[
            chord["domain"].eq("Severe weather")
            & chord["exposure"].isin(TYPE_MAP),
            "exposure",
        ]
    )
    severe["in_chord_signal_natural_type_events"] = severe["variables_en"].isin(chord_vars)

    # Third scope is a stricter single-representative sensitivity: one direct
    # event-count variable per natural-disaster type. This avoids compound
    # categories being driven by multiple correlated variants (e.g. cold,
    # winter, and ice events).
    severe["in_strict_representative_type_events"] = severe["variables_en"].isin(
        STRICT_REPRESENTATIVE_TYPES
    )
    return severe.sort_values(["natural_disaster_type", "variables_en"])


def summarize_pool(
    assoc: pd.DataFrame,
    panel: pd.DataFrame,
    variable_map: pd.DataFrame,
    pool_name: str,
    flag_col: str,
) -> pd.DataFrame:
    variables = variable_map[
        variable_map[flag_col]
        & variable_map["available_in_panel"]
        & variable_map["natural_disaster_type"].notna()
    ][["variables_en", "natural_disaster_type"]].drop_duplicates()
    variables = variables.rename(columns={"variables_en": "exposure", "natural_disaster_type": "disaster_type"})

    d = assoc[
        assoc["phenotype_scope"].eq("per_cow_26")
        & assoc["domain"].eq("Severe weather")
        & assoc["status"].eq("ok")
    ].merge(variables, on="exposure", how="inner")
    d["loss_beta"] = np.maximum(-pd.to_numeric(d["beta"], errors="coerce"), 0.0)
    response = (
        d.groupby(["disaster_type", "year"], dropna=False)
        .agg(
            n_response_variables=("exposure", "nunique"),
            median_beta=("beta", "median"),
            mean_beta=("beta", "mean"),
            absolute_response_strength=("beta", lambda x: float(np.median(np.abs(x)))),
            loss_response_strength=("loss_beta", "mean"),
            frac_negative=("beta", lambda x: float((x < 0).mean())),
        )
        .reset_index()
    )

    exposure_rows = []
    for _, row in variables.iterrows():
        v = row["exposure"]
        x = pd.to_numeric(panel[v], errors="coerce")
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - x.mean()) / sd
        exposure_rows.append(
            pd.DataFrame(
                {
                    "disaster_type": row["disaster_type"],
                    "exposure": v,
                    "year": panel["year"],
                    "exposure_z": z,
                    "abs_exposure_z": z.abs(),
                }
            ).dropna()
        )
    exposure = pd.concat(exposure_rows, ignore_index=True)
    burden = (
        exposure.groupby(["disaster_type", "year"], dropna=False)
        .agg(
            n_exposure_variables=("exposure", "nunique"),
            mean_abs_exposure_z=("abs_exposure_z", "mean"),
            exposure_contrast_sd_z=("exposure_z", "std"),
        )
        .reset_index()
    )
    out = response.merge(burden, on=["disaster_type", "year"], how="inner")
    out["pool"] = pool_name
    pieces = []
    for dtype, g in out.groupby("disaster_type", dropna=False):
        g = g.sort_values("year").copy()
        base = g[g["year"].eq(2000)]
        if base.empty:
            continue
        b_exp = float(base["mean_abs_exposure_z"].iloc[0])
        b_loss = float(base["loss_response_strength"].iloc[0])
        b_abs = float(base["absolute_response_strength"].iloc[0])
        g["exposure_burden_index_2000"] = g["mean_abs_exposure_z"] / b_exp if b_exp else np.nan
        g["absolute_response_index_2000"] = g["absolute_response_strength"] / b_abs if b_abs else np.nan
        g["loss_response_index_2000"] = g["loss_response_strength"] / b_loss if b_loss else np.nan
        g["absolute_translation"] = g["absolute_response_strength"] / g["mean_abs_exposure_z"]
        g["loss_translation"] = g["loss_response_strength"] / g["mean_abs_exposure_z"]
        g["loss_translation_index_2000"] = g["loss_response_index_2000"] / g["exposure_burden_index_2000"]
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def slope(y: pd.Series, x: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 5:
        return np.nan
    xx = d["x"].to_numpy(float)
    yy = d["y"].to_numpy(float)
    xx = xx - xx.mean()
    den = xx @ xx
    return float((xx @ (yy - yy.mean())) / den) if den else np.nan


def summarize_trends(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool, dtype), g in traj.groupby(["pool", "disaster_type"], dropna=False):
        g = g.sort_values("year")
        start = g[g["year"].eq(g["year"].min())].iloc[0]
        end = g[g["year"].eq(g["year"].max())].iloc[0]
        rows.append(
            {
                "pool": pool,
                "disaster_type": dtype,
                "n_years": int(g["year"].nunique()),
                "n_response_variables": int(g["n_response_variables"].median()),
                "n_exposure_variables": int(g["n_exposure_variables"].median()),
                "exposure_pct_change": float(100 * (end["exposure_burden_index_2000"] / start["exposure_burden_index_2000"] - 1)),
                "loss_translation_start": float(start["loss_translation"]),
                "loss_translation_end": float(end["loss_translation"]),
                "loss_translation_pct_change": float(100 * (end["loss_translation"] / start["loss_translation"] - 1))
                if start["loss_translation"] != 0
                else np.nan,
                "frac_negative_start": float(start["frac_negative"]),
                "frac_negative_end": float(end["frac_negative"]),
                "loss_translation_slope": slope(g["loss_translation"], g["year"]),
                "exposure_burden_slope": slope(g["exposure_burden_index_2000"], g["year"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    panel = load_panel()
    audit = build_variable_audit(panel)
    audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")

    assoc = pd.read_csv(TAB / "point2_all_clean_yearly_variable_associations.csv", low_memory=False)
    pools = [
        ("clean natural type events", "in_clean_natural_type_events"),
        ("chord-signal natural events", "in_chord_signal_natural_type_events"),
        ("strict representative type events", "in_strict_representative_type_events"),
    ]
    parts = [summarize_pool(assoc, panel, audit, name, flag) for name, flag in pools]
    traj = pd.concat([x for x in parts if not x.empty], ignore_index=True)
    traj.to_csv(OUT_TRAJ, index=False, encoding="utf-8-sig")
    trends = summarize_trends(traj)
    trends.to_csv(OUT_TREND, index=False, encoding="utf-8-sig")

    print("Severe natural-disaster variables")
    print(audit[audit["in_clean_natural_type_events"]].groupby("natural_disaster_type")["variables_en"].nunique().to_string())
    print("\nPool x type counts")
    for name, flag in pools:
        x = audit[audit[flag] & audit["available_in_panel"] & audit["natural_disaster_type"].notna()]
        print("\n", name)
        print(x.groupby("natural_disaster_type")["variables_en"].nunique().to_string())
    print("\nTrend summary")
    print(trends[["pool", "disaster_type", "n_response_variables", "exposure_pct_change", "loss_translation_pct_change", "frac_negative_start", "frac_negative_end"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
