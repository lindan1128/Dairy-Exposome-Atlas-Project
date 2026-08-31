#!/usr/bin/env python3
"""Prepare the Point 1 Manhattan-matched panel for package robustness models."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
TAB = POINT / "tables"
KEY = ["state_alpha", "year", "month"]

sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402


ASSOC_IN = TAB / "point1_native_only_endpoint_exwas_associations.csv"
OUT_META = TAB / "point1_manhattan_robustness_input.csv"
OUT_PANEL = TAB / "point1_manhattan_robustness_panel_for_r.csv"

ENDPOINTS = {
    "per_cow_50": "milk_per_cow_lb",
    "total_same_50": "milk_production_lb_same50",
}

PLOT_DOMAINS = [
    "Heat",
    "Cold",
    "Severe weather",
    "Forage condition",
    "Feed market",
    "Dairy market",
    "Market demand",
]


def harmonize_assoc_columns(assoc: pd.DataFrame) -> pd.DataFrame:
    assoc = assoc.copy()
    if "domain" not in assoc.columns and "class" in assoc.columns:
        assoc = assoc.rename(columns={"class": "domain"})
    if "source_class" not in assoc.columns and "source_domain" in assoc.columns:
        assoc = assoc.rename(columns={"source_domain": "source_class"})
    if "mechanistic_domain_en" not in assoc.columns and "mechanistic_subclass_en" in assoc.columns:
        assoc = assoc.rename(columns={"mechanistic_subclass_en": "mechanistic_domain_en"})
    return assoc


def normalize_domain(df: pd.DataFrame) -> pd.Series:
    return np.select(
        [
            df["domain"].eq("Pandemic shock") & df["mechanistic_domain_en"].eq("COVID"),
            df["domain"].eq("Pandemic shock") & df["mechanistic_domain_en"].eq("HPAI"),
        ],
        ["COVID", "HPAI"],
        default=df["domain"],
    )


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)

    assoc = harmonize_assoc_columns(pd.read_csv(ASSOC_IN, low_memory=False))
    assoc["domain"] = normalize_domain(assoc)
    assoc = assoc[
        assoc["domain"].isin(PLOT_DOMAINS)
        & assoc["window"].eq("native")
        & assoc["phenotype_scope"].isin(ENDPOINTS)
        & np.isfinite(assoc["plot_p"])
    ].copy()

    assoc["outcome_col"] = assoc["phenotype_scope"].map(ENDPOINTS)
    meta_cols = [
        "phenotype",
        "phenotype_scope",
        "phenotype_label",
        "outcome_col",
        "exposure",
        "exposure_zh",
        "domain",
        "mechanistic_domain_en",
        "source_class",
        "construct",
        "window",
        "form",
        "is_dairy_weighted_exposure",
        "measurement_support_variable",
        "beta",
        "se_cluster",
        "p_cluster",
        "incr_r2",
        "plot_beta",
        "plot_p",
        "plot_incr_r2",
        "native_signal_tier",
        "native_bonferroni_p",
        "native_by_fdr_p",
    ]
    meta_cols = [c for c in meta_cols if c in assoc.columns]
    meta = assoc[meta_cols].copy()
    meta = meta.drop_duplicates(["phenotype_scope", "exposure"], keep="first")
    meta.to_csv(OUT_META, index=False, encoding="utf-8-sig")

    exposures = sorted(meta["exposure"].dropna().unique().tolist())
    panel = L.load_panel()
    panel["milk_production_lb_same50"] = panel["milk_production_lb"]
    panel["year_month"] = panel["year"].astype(int).astype(str) + "_" + panel["month"].astype(int).astype(str).str.zfill(2)
    panel["time_index"] = (panel["year"].astype(int) - int(panel["year"].min())) * 12 + panel["month"].astype(int)
    panel["log_milk_cows"] = np.log(panel["milk_cows_head"].where(panel["milk_cows_head"] > 0))

    keep = KEY + [
        "year_month",
        "time_index",
        "milk_cows_head",
        "log_milk_cows",
        "milk_per_cow_lb",
        "milk_production_lb_same50",
    ]
    missing = [x for x in exposures if x not in panel.columns]
    if missing:
        print(f"Warning: {len(missing)} Manhattan exposures are missing from the panel.")
        for x in missing[:20]:
            print(f"  missing: {x}")
    keep += [x for x in exposures if x in panel.columns]
    panel[keep].to_csv(OUT_PANEL, index=False, encoding="utf-8-sig")

    print(f"Wrote {OUT_META} rows={len(meta)} exposures={meta['exposure'].nunique()}")
    print(f"Wrote {OUT_PANEL} rows={len(panel)} cols={len(keep)}")
    print(meta.groupby(["phenotype_scope", "domain"]).size().to_string())


if __name__ == "__main__":
    main()
