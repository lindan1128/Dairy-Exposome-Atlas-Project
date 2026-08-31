#!/usr/bin/env python3
"""Summarize yearly R2 metrics for regression variants."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
TAB = POINT / "tables"

FIT_STATS = TAB / "point2_four_regression_variant_yearly_fit_stats.csv"
OUT_TABLE = TAB / "point2_four_regression_variant_yearly_r2_by_year.csv"

DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
VARIANT_ORDER = [
    "fixest_fe",
    "fixest_year_month_fe",
    "mgcv_gam",
    "geepack_gee",
]
VARIANT_LABELS = {
    "fixest_fe": "fixest FE",
    "fixest_year_month_fe": "fixest FE + year-month FE",
    "mgcv_gam": "mgcv GAM",
    "geepack_gee": "geepack GEE",
}

def build_summary() -> pd.DataFrame:
    fit = pd.read_csv(FIT_STATS)
    d = fit[
        fit["status"].eq("ok")
        & fit["year"].between(2000, 2024)
        & fit["variant"].isin(VARIANT_ORDER)
        & fit["domain_label"].isin(DOMAIN_ORDER)
        & fit["partial_r2_year"].notna()
    ].copy()
    summary = (
        d.groupby(["variant", "variant_label", "year", "domain_label"], as_index=False)
        .agg(
            median_partial_r2_year=("partial_r2_year", "median"),
            median_partial_adj_r2_year=("partial_adj_r2_year", "median"),
            median_incremental_r2_year=("incremental_r2_year", "median"),
            median_adjusted_incremental_r2_year=("adjusted_incremental_r2_year", "median"),
            median_full_model_r2_year=("r2_year", "median"),
            median_full_model_adj_r2_year=("adj_r2_year", "median"),
            n_exposures=("exposure", "nunique"),
        )
        .sort_values(["variant", "year", "domain_label"])
    )
    summary.to_csv(OUT_TABLE, index=False)
    print(f"Wrote {OUT_TABLE}")
    return summary


def main() -> int:
    summary = build_summary()
    print("\nExposure counts entering R2 panels: min/max")
    print(
        summary.groupby(["variant_label", "domain_label"])["n_exposures"]
        .agg(["min", "max"])
        .reset_index()
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
