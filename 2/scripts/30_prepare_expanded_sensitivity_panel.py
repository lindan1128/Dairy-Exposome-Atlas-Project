#!/usr/bin/env python3
"""Prepare a slim panel CSV for R-based regression sensitivity analyses."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
TAB = POINT / "tables"
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402


SELECTION = TAB / "point2_common_sense_expanded_kept_variables.csv"
YEARLY_SENSITIVITY = TAB / "point2_herd_adjusted_yearly_sensitivity.csv"
PERCOW_2015_2024_SCREEN = TAB / "point2_2015_2024_percow_clean_curated_7class_screen.csv"
OUT = TAB / "point2_expanded_sensitivity_panel_for_r.csv"
LB_TO_KG = 0.45359237
DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
YEARS_FOR_SCATTER = list(range(2015, 2025))


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def main() -> int:
    selected = pd.read_csv(SELECTION)
    selected_exposures = selected[
        selected["expanded_selection_status"].str.startswith("kept_expanded", na=False)
    ]["exposure"].drop_duplicates().tolist()
    scatter_exposures: list[str] = []
    if YEARLY_SENSITIVITY.exists():
        yearly = pd.read_csv(YEARLY_SENSITIVITY, usecols=["outcome", "status", "year", "class_label", "exposure"])
        scatter_exposures = (
            yearly[
                yearly["outcome"].eq("per_cow")
                & yearly["status"].eq("ok")
                & yearly["year"].isin(YEARS_FOR_SCATTER)
                & yearly["class_label"].isin(DOMAIN_ORDER)
            ]["exposure"]
            .drop_duplicates()
            .tolist()
        )
    nominal_exposures: list[str] = []
    if PERCOW_2015_2024_SCREEN.exists():
        screen = pd.read_csv(
            PERCOW_2015_2024_SCREEN,
            usecols=[
                "exposure",
                "bonferroni_sig_2015_2024",
                "by_fdr_sig_2015_2024",
                "nominal_sig_2015_2024",
            ],
        )
        union_sig = (
            screen["bonferroni_sig_2015_2024"].fillna(False).astype(bool)
            | screen["by_fdr_sig_2015_2024"].fillna(False).astype(bool)
            | screen["nominal_sig_2015_2024"].fillna(False).astype(bool)
        )
        nominal_exposures = screen[union_sig][
            "exposure"
        ].drop_duplicates().tolist()
    exposures = ordered_unique(selected_exposures + scatter_exposures + nominal_exposures)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel().copy()
    panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * LB_TO_KG
    keep = [
        "state_alpha",
        "year",
        "month",
        "milk_cows_head",
        "milk_production_lb",
        "milk_per_cow_kg",
    ] + [x for x in exposures if x in panel.columns]
    out = panel.loc[panel["year"].between(2000, 2025), keep].replace([np.inf, -np.inf], np.nan).copy()
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(f"Rows: {len(out)}; exposures: {len(keep) - 6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
