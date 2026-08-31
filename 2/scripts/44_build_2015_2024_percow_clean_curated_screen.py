#!/usr/bin/env python3
"""Screen clean-curated 7-class exposures against 2015-2024 milk per cow."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys
import warnings

import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
ROOT = POINT.parents[3]
TAB = POINT / "tables"
POINT1 = POINT.parent / "1"
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402


CLEAN_DICT = (
    ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
OUT_SCREEN = TAB / "point2_2015_2024_percow_clean_curated_7class_screen.csv"
OUT_YEARLY = TAB / "point2_2015_2024_percow_nominal_alpha02_yearly_sensitivity.csv"
OUT_SUMMARY = TAB / "point2_2015_2024_percow_clean_curated_7class_screen_summary.csv"
SCREEN_ALPHA = 0.2

DOMAIN_MAP = {
    "Heat": "Heat",
    "Cold": "Cold",
    "Severe weather": "Severe weather",
    "Forage condition": "Forage",
    "Feed market": "Feed market",
    "Dairy market": "Dairy market",
    "Market demand": "Market demand",
}
DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]


def by_adjust(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    vals = p[ok].to_numpy(float)
    m = len(vals)
    if m == 0:
        return out
    order = np.argsort(vals)
    harmonic = np.sum(1 / np.arange(1, m + 1))
    q_sorted = vals[order] * m * harmonic / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1)
    out.loc[ok] = q
    return out


def load_clean_variables() -> pd.DataFrame:
    d = pd.read_excel(CLEAN_DICT, sheet_name="exwas_variables")
    d = d[d["used_in_exwas"].fillna(False).astype(bool)].copy()
    if "class" not in d.columns and "domain" in d.columns:
        d = d.rename(columns={"domain": "class"})
    d = d[d["class"].isin(DOMAIN_MAP)].copy()
    d["class_plot"] = d["class"].map(DOMAIN_MAP)
    return d.drop_duplicates("variables_en").reset_index(drop=True)


def screen_2015_2024(meta: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel().copy()
    panel = panel[panel["year"].between(2015, 2024)].copy()
    rows = []
    for _, m in meta.iterrows():
        exposure = str(m["variables_en"])
        if exposure not in panel.columns:
            rows.append(
                {
                    "class": m["class"],
                    "class_plot": m["class_plot"],
                    "exposure": exposure,
                    "status": "missing_in_panel",
                    "beta": np.nan,
                    "se": np.nan,
                    "p": np.nan,
                    "n": 0,
                    "n_clusters": 0,
                    "incr_r2": np.nan,
                }
            )
            continue
        fit = L.fit_exposure(panel, "milk_per_cow_lb", exposure, spec="twoway", weight_col="milk_cows_head")
        res = fit.get("results", {}).get(exposure, {})
        rows.append(
            {
                "class": m["class"],
                "class_plot": m["class_plot"],
                "exposure": exposure,
                "exposure_zh": m.get("variables_ch"),
                "definition_en": m.get("definition_en"),
                "construct": m.get("construct"),
                "window": m.get("window"),
                "form": m.get("form"),
                "status": fit.get("status"),
                "beta": res.get("beta", np.nan),
                "se": res.get("se_cluster", np.nan),
                "p": res.get("p_cluster", np.nan),
                "n": fit.get("n", np.nan),
                "n_clusters": fit.get("n_clusters", np.nan),
                "incr_r2": fit.get("incr_r2", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    ok = out["status"].eq("ok") & out["p"].notna()
    n_tests = int(ok.sum())
    out["n_tests_2015_2024"] = n_tests
    out["screen_alpha_2015_2024"] = SCREEN_ALPHA
    out["bonferroni_threshold_2015_2024"] = SCREEN_ALPHA / n_tests if n_tests else np.nan
    out["q_by_2015_2024"] = np.nan
    out.loc[ok, "q_by_2015_2024"] = by_adjust(out.loc[ok, "p"])
    out["bonferroni_sig_2015_2024"] = ok & (out["p"] < out["bonferroni_threshold_2015_2024"])
    out["by_fdr_sig_2015_2024"] = ok & (out["q_by_2015_2024"] < SCREEN_ALPHA)
    out["nominal_sig_2015_2024"] = ok & (out["p"] < SCREEN_ALPHA)
    out["p05_sig_2015_2024"] = ok & (out["p"] < 0.05)
    return out


def build_yearly_for_nominal(screen: pd.DataFrame) -> pd.DataFrame:
    sig_col = "nominal_sig_2015_2024" if "nominal_sig_2015_2024" in screen.columns else "p05_sig_2015_2024"
    nominal = screen[screen[sig_col]].copy()
    yearly_mod = runpy.run_path(str(POINT / "scripts" / "08_herd_adjusted_yearly_sensitivity.py"))
    panel = yearly_mod["prepare_panel"]()
    fit_yearly = yearly_mod["fit_yearly_herd_adjusted"]
    rows = []
    for _, r in nominal.iterrows():
        exposure = r["exposure"]
        if exposure not in panel.columns:
            continue
        for fit in fit_yearly(panel, "log_per_cow", exposure):
            rows.append(
                {
                    "outcome": "per_cow",
                    "outcome_label": "Milk per cow (kg)",
                    "class_plot": r["class_plot"],
                    "class_label": r["class_plot"],
                    "source_domain": "2015-2024 nominal p<0.05 clean-curated screen",
                    "exposure": exposure,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    meta = load_clean_variables()
    screen = screen_2015_2024(meta)
    screen.to_csv(OUT_SCREEN, index=False)
    yearly = build_yearly_for_nominal(screen)
    yearly.to_csv(OUT_YEARLY, index=False)

    ok = screen[screen["status"].eq("ok") & screen["p"].notna()].copy()
    summary = (
        ok.groupby("class_plot", observed=True)
        .agg(
            n_fitted=("exposure", "nunique"),
            n_bonferroni=("bonferroni_sig_2015_2024", "sum"),
            n_by_fdr=("by_fdr_sig_2015_2024", "sum"),
            n_nominal=("nominal_sig_2015_2024", "sum"),
            n_p05=("p05_sig_2015_2024", "sum"),
        )
        .reindex(DOMAIN_ORDER)
        .reset_index()
    )
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote {OUT_SCREEN}")
    print(f"Wrote {OUT_YEARLY}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Candidate variables: {meta['variables_en'].nunique()}")
    print(f"Fitted variables: {len(ok)}")
    print(
        "Significant counts: "
        f"Bonferroni={int(ok['bonferroni_sig_2015_2024'].sum())}; "
        f"BY-FDR={int(ok['by_fdr_sig_2015_2024'].sum())}; "
        f"P<{SCREEN_ALPHA}={int(ok['nominal_sig_2015_2024'].sum())}; "
        f"P<0.05={int(ok['p05_sig_2015_2024'].sum())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
