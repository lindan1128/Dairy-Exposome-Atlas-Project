#!/usr/bin/env python3
"""Decompose Point 2 heat and severe-weather signals into mechanistic subforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
ROOT = Path(__file__).resolve().parents[5]
TAB = POINT / "tables"
CLEAN_DICT = (
    ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
US_MILK = ROOT / "data" / "us_milk" / "processed"
US_EXPOSE_NEW = ROOT / "data" / "us_expose_new" / "processed"
OUT_VAR = TAB / "point2_heat_severe_subform_variable_summary.csv"
OUT_YEAR = TAB / "point2_heat_severe_subform_yearly_summary.csv"
OUT_SUB = TAB / "point2_heat_severe_subform_summary.csv"
OUT_HEAT_TRAJ = TAB / "point2_heat_subform_exposure_translation_trajectory.csv"


def heat_subform(x: str) -> tuple[str, str]:
    s = x.lower()
    if "wetbulb" in s or "humid_hot" in s:
        return "Humid threshold heat", "Wet-bulb / humid heat threshold days or heatload"
    if "thi_days" in s or "thi_heatload" in s or "consec_thi" in s:
        return "THI threshold heat", "THI threshold days, duration, or cumulative heatload"
    if "warm_nights" in s or "no_relief" in s or "diurnal_range" in s:
        return "Night heat / no relief", "Warm nights or limited nocturnal recovery"
    if "vpd" in s or "dry_hot" in s:
        return "Dry heat", "Dry-bulb temperature extremes, high vapor-pressure deficit, or dry-hot days"
    if "tmax_days" in s:
        return "Dry heat", "Dry-bulb temperature extremes, high vapor-pressure deficit, or dry-hot days"
    if any(k in s for k in ["tmax", "tmean", "thi_mean", "thi_max", "wetbulb_max"]):
        return "Mean / max heat", "Monthly mean or maximum heat intensity"
    return "Other heat", "Other heat form"


def severe_subform(x: str) -> tuple[str, str]:
    s = x.lower()
    if "flood" in s:
        return "Flood events", "Storm-event flood count"
    if "fire" in s:
        return "Fire events", "Storm-event fire count"
    if "heat_events" in s:
        return "Severe heat events", "Storm-event heat count"
    if "damage" in s:
        return "Crop/storm damage", "Storm-related crop damage in USD"
    return "Other severe weather", "Other severe-weather form"


def slope(y: pd.Series, x: pd.Series) -> float:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 5:
        return np.nan
    xx = d["x"].to_numpy(float)
    yy = d["y"].to_numpy(float)
    xx = xx - xx.mean()
    den = float(xx @ xx)
    if den == 0:
        return np.nan
    return float((xx @ (yy - yy.mean())) / den)


def main() -> int:
    assoc = pd.read_csv(TAB / "point2_chord_signal_yearly_variable_associations.csv", low_memory=False)
    dictionary = pd.read_excel(CLEAN_DICT, sheet_name="exwas_variables")
    dictionary = dictionary[dictionary["used_in_exwas"].fillna(False).astype(bool)].copy()
    dictionary = dictionary[["domain", "variables_en", "variables_ch", "construct", "mechanistic_domain_en", "form"]].drop_duplicates()

    d = assoc[
        assoc["phenotype_scope"].eq("per_cow_26")
        & assoc["domain"].isin(["Heat", "Severe weather"])
        & assoc["status"].eq("ok")
    ].copy()
    d["abs_beta"] = d["beta"].abs()
    d["neglogp"] = -np.log10(d["p"].clip(lower=1e-300))
    sub = d["exposure"].apply(lambda x: heat_subform(x) if "Heat" in d.loc[d["exposure"].eq(x), "domain"].iloc[0] else severe_subform(x))
    d["subform"] = [x[0] for x in sub]
    d["subform_note"] = [x[1] for x in sub]

    d = d.merge(
        dictionary,
        left_on=["domain", "exposure"],
        right_on=["domain", "variables_en"],
        how="left",
    )
    d["variable_label"] = d["variables_ch"].fillna(d["exposure"])

    var_summary = (
        d.groupby(
            [
                "domain",
                "subform",
                "subform_note",
                "exposure",
                "variable_label",
                "construct",
                "mechanistic_domain_en",
                "form",
                "source_effect_directions",
                "source_signal_tiers",
            ],
            dropna=False,
        )
        .agg(
            n_years=("year", "nunique"),
            median_beta=("beta", "median"),
            mean_beta=("beta", "mean"),
            median_abs_beta=("abs_beta", "median"),
            mean_abs_beta=("abs_beta", "mean"),
            max_abs_beta=("abs_beta", "max"),
            frac_negative=("beta", lambda x: float((x < 0).mean())),
            median_neglogp=("neglogp", "median"),
            max_neglogp=("neglogp", "max"),
            beta_year_slope=("beta", lambda x: slope(x, d.loc[x.index, "year"])),
            abs_beta_year_slope=("abs_beta", lambda x: slope(x, d.loc[x.index, "year"])),
            best_source_plot_p=("best_source_plot_p", "min"),
            max_source_plot_incr_r2=("max_source_plot_incr_r2", "max"),
        )
        .reset_index()
        .sort_values(["domain", "median_abs_beta"], ascending=[True, False])
    )
    var_summary.to_csv(OUT_VAR, index=False, encoding="utf-8-sig")

    yearly = (
        d.groupby(["domain", "subform", "subform_note", "year"], dropna=False)
        .agg(
            n_variables=("exposure", "nunique"),
            median_beta=("beta", "median"),
            median_abs_beta=("abs_beta", "median"),
            mean_abs_beta=("abs_beta", "mean"),
            median_loss_beta=("beta", lambda x: float(np.median(np.maximum(-x.to_numpy(float), 0.0)))),
            mean_loss_beta=("beta", lambda x: float(np.mean(np.maximum(-x.to_numpy(float), 0.0)))),
            frac_negative=("beta", lambda x: float((x < 0).mean())),
            median_neglogp=("neglogp", "median"),
            max_neglogp=("neglogp", "max"),
        )
        .reset_index()
    )
    yearly.to_csv(OUT_YEAR, index=False, encoding="utf-8-sig")

    sub_summary = (
        d.groupby(["domain", "subform", "subform_note"], dropna=False)
        .agg(
            n_variables=("exposure", "nunique"),
            n_years=("year", "nunique"),
            median_abs_beta=("abs_beta", "median"),
            mean_abs_beta=("abs_beta", "mean"),
            median_beta=("beta", "median"),
            frac_negative=("beta", lambda x: float((x < 0).mean())),
            median_neglogp=("neglogp", "median"),
            max_neglogp=("neglogp", "max"),
            median_source_incr_r2=("max_source_plot_incr_r2", "median"),
            best_source_plot_p=("best_source_plot_p", "min"),
        )
        .reset_index()
        .sort_values(["domain", "median_abs_beta"], ascending=[True, False])
    )
    sub_summary.to_csv(OUT_SUB, index=False, encoding="utf-8-sig")

    # Heat-subform exposure burden and translation trajectory.
    # Burden is the yearly mean absolute standardized exposure across variables
    # in a subform. Translation is yearly response strength divided by burden,
    # both indexed to 2000 = 1 within each subform.
    import sys
    sys.path.insert(0, str(STAT))
    import lib_statistics_panel as L  # noqa: E402

    key = ["state_alpha", "year", "month"]
    milk = pd.read_csv(US_MILK / "state_month_panel.csv", low_memory=False)
    exp = pd.read_csv(US_EXPOSE_NEW / "exposure_state_month_expanded.csv", low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in key]
    if overlap:
        milk = milk.drop(columns=overlap)
    full_panel = milk.merge(exp, on=key, how="left")

    heat_vars = (
        d.loc[d["domain"].eq("Heat"), ["exposure", "subform"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    exposure_rows = []
    for _, hv in heat_vars.iterrows():
        v = hv["exposure"]
        if v not in full_panel.columns:
            continue
        x = pd.to_numeric(full_panel[v], errors="coerce")
        sd = x.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        z = (x - x.mean()) / sd
        tmp = pd.DataFrame(
            {
                "year": full_panel["year"],
                "subform": hv["subform"],
                "exposure": v,
                "abs_exposure_z": z.abs(),
                "exposure_z": z,
            }
        ).dropna()
        exposure_rows.append(tmp)
    exposure_year = pd.concat(exposure_rows, ignore_index=True)
    exposure_summary = (
        exposure_year.groupby(["subform", "year"], dropna=False)
        .agg(
            n_exposure_variables=("exposure", "nunique"),
            mean_abs_exposure_z=("abs_exposure_z", "mean"),
            exposure_contrast_sd_z=("exposure_z", "std"),
        )
        .reset_index()
    )
    response_summary = (
        yearly.loc[
            yearly["domain"].eq("Heat"),
            [
                "subform",
                "year",
                "n_variables",
                "median_abs_beta",
                "mean_loss_beta",
                "median_beta",
                "frac_negative",
            ],
        ]
        .rename(
            columns={
                "n_variables": "n_response_variables",
                "median_abs_beta": "absolute_response_strength",
                "mean_loss_beta": "loss_response_strength",
            }
        )
    )
    traj = response_summary.merge(exposure_summary, on=["subform", "year"], how="inner")
    traj = traj.sort_values(["subform", "year"])
    out_parts = []
    for sf, g in traj.groupby("subform", dropna=False):
        g = g.copy()
        base = g.loc[g["year"].eq(2000)]
        if base.empty:
            continue
        b_exp = float(base["mean_abs_exposure_z"].iloc[0])
        b_abs_resp = float(base["absolute_response_strength"].iloc[0])
        b_loss_resp = float(base["loss_response_strength"].iloc[0])
        b_contrast = float(base["exposure_contrast_sd_z"].iloc[0])
        g["exposure_burden_index_2000"] = g["mean_abs_exposure_z"] / b_exp if b_exp else np.nan
        g["exposure_contrast_index_2000"] = g["exposure_contrast_sd_z"] / b_contrast if b_contrast else np.nan
        g["absolute_response_index_2000"] = g["absolute_response_strength"] / b_abs_resp if b_abs_resp else np.nan
        g["loss_response_index_2000"] = g["loss_response_strength"] / b_loss_resp if b_loss_resp else np.nan
        g["absolute_translation"] = g["absolute_response_strength"] / g["mean_abs_exposure_z"]
        g["loss_translation"] = g["loss_response_strength"] / g["mean_abs_exposure_z"]
        g["absolute_translation_index_2000"] = g["absolute_response_index_2000"] / g["exposure_burden_index_2000"]
        g["loss_translation_index_2000"] = g["loss_response_index_2000"] / g["exposure_burden_index_2000"]
        out_parts.append(g)
    heat_traj = pd.concat(out_parts, ignore_index=True)
    heat_traj.to_csv(OUT_HEAT_TRAJ, index=False, encoding="utf-8-sig")

    print("Top Heat variables")
    print(var_summary[var_summary["domain"].eq("Heat")][["subform", "exposure", "median_abs_beta", "frac_negative", "best_source_plot_p"]].head(12).to_string(index=False))
    print("\nSevere weather variables")
    print(var_summary[var_summary["domain"].eq("Severe weather")][["subform", "exposure", "median_abs_beta", "frac_negative", "best_source_plot_p"]].to_string(index=False))
    print(f"Wrote {OUT_VAR}")
    print(f"Wrote {OUT_YEAR}")
    print(f"Wrote {OUT_SUB}")
    print(f"Wrote {OUT_HEAT_TRAJ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
