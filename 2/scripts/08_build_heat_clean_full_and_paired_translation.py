#!/usr/bin/env python3
"""Sensitivity heat translation trajectories for clean-full and paired heat pools.

This script reuses the Point 2 all-clean yearly per-cow association fits and
recomputes heat exposure burden and milk-loss translation for two comparison
sets:

1. clean_full_heat: all clean native Heat variables.
2. strict_paired_heat: mechanistically paired humid/wet-bulb and dry/VPD forms.
"""

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
US_EXPOSE_NEW = ROOT / "data" / "us_expose_new" / "processed"
CLEAN_DICT = (
    ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
KEY = ["state_alpha", "year", "month"]

OUT_VAR = TAB / "point2_heat_clean_full_paired_variable_audit.csv"
OUT_TRAJ = TAB / "point2_heat_clean_full_paired_translation_trajectory.csv"
OUT_TREND = TAB / "point2_heat_clean_full_paired_translation_trend_summary.csv"


STRICT_PAIRS = {
    "daymet_dairy_weighted_wetbulb_days_ge_22c": ("Humid paired heat", "threshold days: mild", "wet-bulb >=22C days"),
    "daymet_dairy_weighted_vpd_days_ge_2kpa": ("Dry paired heat", "threshold days: mild", "VPD >=2 kPa days"),
    "daymet_dairy_weighted_wetbulb_days_ge_24c": ("Humid paired heat", "threshold days: moderate", "wet-bulb >=24C days"),
    "daymet_dairy_weighted_vpd_days_ge_3kpa": ("Dry paired heat", "threshold days: moderate", "VPD >=3 kPa days"),
    "daymet_dairy_weighted_wetbulb_days_ge_26c": ("Humid paired heat", "threshold days: severe", "wet-bulb >=26C days"),
    "daymet_dairy_weighted_vpd_days_ge_4kpa": ("Dry paired heat", "threshold days: severe", "VPD >=4 kPa days"),
    "daymet_dairy_weighted_wetbulb_heatload_ge22": ("Humid paired heat", "threshold heatload", "wet-bulb heatload >=22C"),
    "daymet_dairy_weighted_vpd_heatload_ge2": ("Dry paired heat", "threshold heatload", "VPD heatload >=2 kPa"),
    "daymet_dairy_weighted_humid_hot_days_t72wb24": ("Humid paired heat", "joint hot-days", "T/THI hot and wet-bulb >=24C days"),
    "daymet_dairy_weighted_dry_hot_days_t72wb_lt22": ("Dry paired heat", "joint hot-days", "T/THI hot and wet-bulb <22C days"),
    "daymet_dairy_weighted_wetbulb_mean_c": ("Humid paired heat", "monthly intensity", "mean wet-bulb temperature"),
    "daymet_dairy_weighted_vpd_kpa": ("Dry paired heat", "monthly intensity", "mean VPD"),
    "daymet_dairy_weighted_wetbulb_max_c": ("Humid paired heat", "monthly extreme", "maximum wet-bulb temperature"),
    "daymet_dairy_weighted_vpd_max": ("Dry paired heat", "monthly extreme", "maximum VPD"),
}


def clean_full_subform(mech: str) -> str:
    if mech == "Heat: humid (wet-bulb)":
        return "Humid/wet-bulb heat"
    if mech == "Heat: dry (VPD/aridity)":
        return "Dry/VPD heat"
    if mech == "Heat: combined (THI)":
        return "Combined THI heat"
    if mech == "Heat: nighttime recovery":
        return "Night/no-relief heat"
    return "Other heat"


def load_panel() -> pd.DataFrame:
    milk = pd.read_csv(L.US_MILK / "state_month_panel.csv", low_memory=False)
    exp = pd.read_csv(US_EXPOSE_NEW / "exposure_state_month_expanded.csv", low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    return milk.merge(exp, on=KEY, how="left")


def build_variable_audit() -> pd.DataFrame:
    dictionary = pd.read_excel(CLEAN_DICT, sheet_name="exwas_variables")
    dictionary = dictionary[dictionary["used_in_exwas"].fillna(False).astype(bool)].copy()
    heat = dictionary[dictionary["domain"].eq("Heat")].copy()
    heat = heat[
        [
            "domain",
            "variables_en",
            "variables_ch",
            "construct",
            "mechanistic_domain_en",
            "form",
            "is_dairy_weighted_exposure",
            "n_nonmissing_exposure_months",
            "n_states_exposure",
            "year_min_exposure",
            "year_max_exposure",
        ]
    ].drop_duplicates()
    heat["clean_full_subform"] = heat["mechanistic_domain_en"].map(clean_full_subform)
    heat["in_clean_full_heat"] = True
    heat["in_strict_paired_heat"] = heat["variables_en"].isin(STRICT_PAIRS)
    heat["strict_paired_subform"] = heat["variables_en"].map(
        lambda x: STRICT_PAIRS[x][0] if x in STRICT_PAIRS else np.nan
    )
    heat["strict_pair_group"] = heat["variables_en"].map(
        lambda x: STRICT_PAIRS[x][1] if x in STRICT_PAIRS else np.nan
    )
    heat["strict_pair_label"] = heat["variables_en"].map(
        lambda x: STRICT_PAIRS[x][2] if x in STRICT_PAIRS else np.nan
    )
    complete_groups = (
        heat[heat["in_strict_paired_heat"]]
        .groupby("strict_pair_group")["strict_paired_subform"]
        .nunique()
    )
    complete_groups = set(complete_groups[complete_groups.eq(2)].index)
    heat["in_strict_paired_heat"] = (
        heat["in_strict_paired_heat"] & heat["strict_pair_group"].isin(complete_groups)
    )

    assoc = pd.read_csv(TAB / "point2_all_clean_yearly_variable_associations.csv", low_memory=False)
    observed = (
        assoc.loc[assoc["domain"].eq("Heat")]
        .groupby("exposure", dropna=False)
        .agg(n_yearly_fits=("year", "nunique"), n_ok_yearly_fits=("status", lambda x: int((x == "ok").sum())))
        .reset_index()
        .rename(columns={"exposure": "variables_en"})
    )
    heat = heat.merge(observed, on="variables_en", how="left")
    heat["n_yearly_fits"] = heat["n_yearly_fits"].fillna(0).astype(int)
    heat["n_ok_yearly_fits"] = heat["n_ok_yearly_fits"].fillna(0).astype(int)
    return heat.sort_values(["clean_full_subform", "variables_en"])


def summarize_trajectory(
    assoc: pd.DataFrame,
    panel: pd.DataFrame,
    variable_map: pd.DataFrame,
    pool_name: str,
    subform_col: str,
) -> pd.DataFrame:
    variables = variable_map[["variables_en", subform_col]].dropna().drop_duplicates()
    variables = variables.rename(columns={"variables_en": "exposure", subform_col: "subform"})
    d = assoc[
        assoc["phenotype_scope"].eq("per_cow_26")
        & assoc["domain"].eq("Heat")
        & assoc["status"].eq("ok")
    ].merge(variables, on="exposure", how="inner")
    d["loss_beta"] = np.maximum(-pd.to_numeric(d["beta"], errors="coerce"), 0.0)
    response = (
        d.groupby(["subform", "year"], dropna=False)
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
        if v not in panel.columns:
            continue
        x = pd.to_numeric(panel[v], errors="coerce")
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - x.mean()) / sd
        exposure_rows.append(
            pd.DataFrame(
                {
                    "subform": row["subform"],
                    "exposure": v,
                    "year": panel["year"],
                    "exposure_z": z,
                    "abs_exposure_z": z.abs(),
                }
            ).dropna()
        )
    exposure = pd.concat(exposure_rows, ignore_index=True)
    burden = (
        exposure.groupby(["subform", "year"], dropna=False)
        .agg(
            n_exposure_variables=("exposure", "nunique"),
            mean_abs_exposure_z=("abs_exposure_z", "mean"),
            exposure_contrast_sd_z=("exposure_z", "std"),
        )
        .reset_index()
    )
    out = response.merge(burden, on=["subform", "year"], how="inner")
    out["pool"] = pool_name
    out = out.sort_values(["pool", "subform", "year"])
    pieces = []
    for subform, g in out.groupby("subform", dropna=False):
        g = g.copy()
        base = g[g["year"].eq(2000)]
        if base.empty:
            continue
        b_exp = float(base["mean_abs_exposure_z"].iloc[0])
        b_loss = float(base["loss_response_strength"].iloc[0])
        b_abs = float(base["absolute_response_strength"].iloc[0])
        b_contrast = float(base["exposure_contrast_sd_z"].iloc[0])
        g["exposure_burden_index_2000"] = g["mean_abs_exposure_z"] / b_exp if b_exp else np.nan
        g["exposure_contrast_index_2000"] = g["exposure_contrast_sd_z"] / b_contrast if b_contrast else np.nan
        g["loss_response_index_2000"] = g["loss_response_strength"] / b_loss if b_loss else np.nan
        g["absolute_response_index_2000"] = g["absolute_response_strength"] / b_abs if b_abs else np.nan
        g["loss_translation"] = g["loss_response_strength"] / g["mean_abs_exposure_z"]
        g["absolute_translation"] = g["absolute_response_strength"] / g["mean_abs_exposure_z"]
        g["loss_translation_index_2000"] = g["loss_response_index_2000"] / g["exposure_burden_index_2000"]
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def slope(y: pd.Series, x: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 5:
        return np.nan
    xx = d["x"].to_numpy(float)
    yy = d["y"].to_numpy(float)
    xx = xx - xx.mean()
    den = xx @ xx
    return float((xx @ (yy - yy.mean())) / den) if den else np.nan


def summarize_trend(traj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool, subform), g in traj.groupby(["pool", "subform"], dropna=False):
        g = g.sort_values("year")
        start = g[g["year"].eq(g["year"].min())].iloc[0]
        end = g[g["year"].eq(g["year"].max())].iloc[0]
        rows.append(
            {
                "pool": pool,
                "subform": subform,
                "n_years": int(g["year"].nunique()),
                "n_response_variables": int(g["n_response_variables"].median()),
                "n_exposure_variables": int(g["n_exposure_variables"].median()),
                "exposure_start": float(start["exposure_burden_index_2000"]),
                "exposure_end": float(end["exposure_burden_index_2000"]),
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
    TAB.mkdir(parents=True, exist_ok=True)
    audit = build_variable_audit()
    audit.to_csv(OUT_VAR, index=False, encoding="utf-8-sig")

    assoc = pd.read_csv(TAB / "point2_all_clean_yearly_variable_associations.csv", low_memory=False)
    panel = load_panel()

    clean_full = summarize_trajectory(
        assoc,
        panel,
        audit[audit["in_clean_full_heat"]],
        "clean_full_heat",
        "clean_full_subform",
    )
    strict = summarize_trajectory(
        assoc,
        panel,
        audit[audit["in_strict_paired_heat"]],
        "strict_paired_heat",
        "strict_paired_subform",
    )
    traj = pd.concat([clean_full, strict], ignore_index=True)
    traj.to_csv(OUT_TRAJ, index=False, encoding="utf-8-sig")

    trends = summarize_trend(traj)
    trends.to_csv(OUT_TREND, index=False, encoding="utf-8-sig")

    print("Clean-full Heat variables")
    print(audit.groupby("clean_full_subform")["variables_en"].nunique().to_string())
    print("\nStrict paired Heat variables")
    print(audit[audit["in_strict_paired_heat"]].groupby(["strict_pair_group", "strict_paired_subform"])["variables_en"].nunique().to_string())
    print("\nTrend summary")
    print(trends[["pool", "subform", "n_response_variables", "exposure_pct_change", "loss_translation_pct_change", "frac_negative_start", "frac_negative_end"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
