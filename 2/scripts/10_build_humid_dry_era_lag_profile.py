#!/usr/bin/env python3
"""Humid-vs-dry heat lag-recovery profiles across three heat variable pools."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

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
OUT_PROFILE = TAB / "point2_humid_dry_era_lag_profile.csv"
OUT_AUDIT = TAB / "point2_humid_dry_era_lag_variable_audit.csv"
KEY = ["state_alpha", "year", "month"]

PRIMARY = "milk_per_cow_lb"
WEIGHT = "milk_cows_head"
ERAS = {
    "2000-2008": (2000, 2008),
    "2009-2016": (2009, 2016),
    "2017-2025": (2017, 2025),
}
LAGS = range(1, 6)
BOOTSTRAPS = 0

STRICT_PAIRS = {
    "daymet_dairy_weighted_wetbulb_days_ge_22c": ("Humid paired heat", "threshold days: mild"),
    "daymet_dairy_weighted_vpd_days_ge_2kpa": ("Dry paired heat", "threshold days: mild"),
    "daymet_dairy_weighted_wetbulb_days_ge_24c": ("Humid paired heat", "threshold days: moderate"),
    "daymet_dairy_weighted_vpd_days_ge_3kpa": ("Dry paired heat", "threshold days: moderate"),
    "daymet_dairy_weighted_wetbulb_days_ge_26c": ("Humid paired heat", "threshold days: severe"),
    "daymet_dairy_weighted_vpd_days_ge_4kpa": ("Dry paired heat", "threshold days: severe"),
    "daymet_dairy_weighted_wetbulb_heatload_ge22": ("Humid paired heat", "threshold heatload"),
    "daymet_dairy_weighted_vpd_heatload_ge2": ("Dry paired heat", "threshold heatload"),
    "daymet_dairy_weighted_humid_hot_days_t72wb24": ("Humid paired heat", "joint hot-days"),
    "daymet_dairy_weighted_dry_hot_days_t72wb_lt22": ("Dry paired heat", "joint hot-days"),
    "daymet_dairy_weighted_wetbulb_mean_c": ("Humid paired heat", "monthly intensity"),
    "daymet_dairy_weighted_vpd_kpa": ("Dry paired heat", "monthly intensity"),
    "daymet_dairy_weighted_wetbulb_max_c": ("Humid paired heat", "monthly extreme"),
    "daymet_dairy_weighted_vpd_max": ("Dry paired heat", "monthly extreme"),
}


def z(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / sd if np.isfinite(sd) and sd > 0 else x * 0.0


def load_panel() -> pd.DataFrame:
    milk = pd.read_csv(US_MILK / "state_month_panel.csv", low_memory=False)
    exp = pd.read_csv(US_EXPOSE_NEW / "exposure_state_month_expanded.csv", low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    panel = milk.merge(exp, on=KEY, how="left")
    panel["t"] = panel["year"].astype(int) * 12 + panel["month"].astype(int)
    return panel


def build_variable_sets(panel: pd.DataFrame) -> pd.DataFrame:
    dictionary = pd.read_excel(CLEAN_DICT, sheet_name="exwas_variables")
    dictionary = dictionary[dictionary["used_in_exwas"].fillna(False).astype(bool)].copy()
    heat = dictionary[dictionary["domain"].eq("Heat")].copy()
    rows: list[dict] = []

    for _, r in heat.iterrows():
        v = r["variables_en"]
        if v not in panel.columns:
            continue
        mech = r["mechanistic_domain_en"]
        if mech == "Heat: humid (wet-bulb)":
            rows.append(
                {
                    "pool": "clean-full heat",
                    "heat_form": "Humid heat",
                    "exposure": v,
                    "pair_group": np.nan,
                    "variables_ch": r.get("variables_ch", v),
                    "mechanistic_domain_en": mech,
                    "form": r.get("form", np.nan),
                }
            )
        elif mech == "Heat: dry (VPD/aridity)":
            rows.append(
                {
                    "pool": "clean-full heat",
                    "heat_form": "Dry heat",
                    "exposure": v,
                    "pair_group": np.nan,
                    "variables_ch": r.get("variables_ch", v),
                    "mechanistic_domain_en": mech,
                    "form": r.get("form", np.nan),
                }
            )

    for v, (hf, pg) in STRICT_PAIRS.items():
        if v not in panel.columns:
            continue
        match = heat[heat["variables_en"].eq(v)]
        if match.empty:
            continue
        rows.append(
            {
                "pool": "strict paired heat",
                "heat_form": "Humid heat" if hf.startswith("Humid") else "Dry heat",
                "exposure": v,
                "pair_group": pg,
                "variables_ch": match["variables_ch"].iloc[0] if not match.empty else v,
                "mechanistic_domain_en": match["mechanistic_domain_en"].iloc[0] if not match.empty else np.nan,
                "form": match["form"].iloc[0] if not match.empty else np.nan,
            }
        )

    chord = pd.read_csv(TAB / "point2_heat_severe_subform_variable_summary.csv", low_memory=False)
    chord = chord[
        chord["domain"].eq("Heat")
        & chord["subform"].isin(["Humid threshold heat", "Dry heat"])
    ].copy()
    for _, r in chord.iterrows():
        v = r["exposure"]
        if v not in panel.columns:
            continue
        rows.append(
            {
                "pool": "chord-signal heat",
                "heat_form": "Humid heat" if r["subform"] == "Humid threshold heat" else "Dry heat",
                "exposure": v,
                "pair_group": np.nan,
                "variables_ch": r.get("variable_label", v),
                "mechanistic_domain_en": r.get("mechanistic_domain_en", np.nan),
                "form": r.get("form", np.nan),
            }
        )

    out = pd.DataFrame(rows).drop_duplicates(["pool", "heat_form", "exposure"])
    strict = out[out["pool"].eq("strict paired heat")].copy()
    complete_groups = strict.groupby("pair_group")["heat_form"].nunique()
    complete_groups = set(complete_groups[complete_groups.eq(2)].index)
    out = out[
        ~out["pool"].eq("strict paired heat")
        | out["pair_group"].isin(complete_groups)
    ].copy()
    out = out.sort_values(["pool", "heat_form", "exposure"])
    return out


def add_lagged_composites(panel: pd.DataFrame, variable_sets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    for (pool, heat_form), g in variable_sets.groupby(["pool", "heat_form"], sort=False):
        exposures = [v for v in g["exposure"].tolist() if v in panel.columns]
        for lag in LAGS:
            zcols = []
            used = []
            for v in exposures:
                base = panel[["state_alpha", "t", v]].rename(columns={v: "_v"})
                shifted = base.copy()
                shifted["t"] = shifted["t"] + lag
                x = panel[["state_alpha", "t"]].merge(
                    shifted.rename(columns={"_v": "_x"}),
                    on=["state_alpha", "t"],
                    how="left",
                )["_x"].to_numpy(float)
                if np.isfinite(x).sum() < 100 or np.nanstd(x) == 0:
                    continue
                zcols.append(z(x))
                used.append(v)
            comp_col = f"__{pool.replace(' ', '_').replace('-', '_')}__{heat_form.replace(' ', '_')}__lag{lag}"
            if zcols:
                panel[comp_col] = z(np.nanmean(np.column_stack(zcols), axis=1))
            else:
                panel[comp_col] = np.nan
            audit_rows.append(
                {
                    "pool": pool,
                    "heat_form": heat_form,
                    "lag": lag,
                    "composite_col": comp_col,
                    "n_variables": len(used),
                    "variables": ";".join(used),
                }
            )
    return panel, pd.DataFrame(audit_rows)


def estimate_loss(df: pd.DataFrame, exposure_col: str) -> float:
    df = df[["state_alpha", "year", "month", PRIMARY, WEIGHT, exposure_col]].copy()
    fit = L.fit_exposure(
        df,
        PRIMARY,
        exposure_col,
        spec="twoway",
        standardize=False,
        weight_col=WEIGHT,
    )
    if fit["status"] != "ok":
        return np.nan
    beta = fit["results"][exposure_col]["beta"]
    return -beta * 100.0


def main() -> int:
    panel = load_panel()
    variable_sets = build_variable_sets(panel)
    panel, comp_audit = add_lagged_composites(panel, variable_sets)
    variable_sets.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(20260618)
    rows = []
    for _, r in comp_audit.iterrows():
        for era, (lo, hi) in ERAS.items():
            keep_cols = ["state_alpha", "year", "month", PRIMARY, WEIGHT, r["composite_col"]]
            sub = panel.loc[(panel["year"] >= lo) & (panel["year"] <= hi), keep_cols].copy()
            states = sorted(
                sub.dropna(subset=[PRIMARY, WEIGHT, r["composite_col"]])["state_alpha"].unique()
            )
            loss = estimate_loss(sub, r["composite_col"])
            boot = []
            if BOOTSTRAPS > 0:
                for _ in range(BOOTSTRAPS):
                    sampled = rng.choice(states, len(states), replace=True)
                    pieces = [
                        sub[sub["state_alpha"].eq(s)].assign(state_alpha=f"{s}__{i}")
                        for i, s in enumerate(sampled)
                    ]
                    boot_df = pd.concat(pieces, ignore_index=True)
                    boot.append(estimate_loss(boot_df, r["composite_col"]))
            b = np.array([x for x in boot if np.isfinite(x)])
            rows.append(
                {
                    "pool": r["pool"],
                    "heat_form": r["heat_form"],
                    "lag": int(r["lag"]),
                    "era": era,
                    "loss": loss,
                    "lo": float(np.percentile(b, 2.5)) if len(b) else np.nan,
                    "hi": float(np.percentile(b, 97.5)) if len(b) else np.nan,
                    "n_variables": int(r["n_variables"]),
                    "composite_col": r["composite_col"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PROFILE, index=False, encoding="utf-8-sig")

    print("Variable counts")
    print(variable_sets.groupby(["pool", "heat_form"])["exposure"].nunique().to_string())
    print("\nLag profile preview")
    print(out.round(3).head(30).to_string(index=False))
    print(f"Wrote {OUT_PROFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
