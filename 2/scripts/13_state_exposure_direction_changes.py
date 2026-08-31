#!/usr/bin/env python3
"""Simple state-by-exposure direction changes relative to 2000 and 2015."""

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


DECOMP = TAB / "point2_total_insensitivity_log_decomposition.csv"
OUT_LONG = TAB / "point2_state_exposure_direction_changes_relative_2000_2015.csv"
OUT_DOMAIN_STATE = TAB / "point2_state_exposure_direction_changes_domain_state_summary.csv"
OUT_DOMAIN = TAB / "point2_state_exposure_direction_changes_domain_summary.csv"

DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
COMPARISONS = [
    {"baseline_year": 2000, "comparison_year": 2014, "stage": "2000_to_2014"},
    {"baseline_year": 2015, "comparison_year": 2025, "stage": "2015_to_2025"},
]


def selected_exposures() -> pd.DataFrame:
    d = pd.read_csv(DECOMP)
    d["domain_label"] = d["domain_plot"].replace({"Forage condition": "Forage"})
    return d[
        d["domain_label"].isin(DOMAIN_ORDER)
        & ((d["total_p"] < 0.05) | (d["per_cow_p"] < 0.05) | (d["cows_p"] < 0.05))
    ][["exposure", "domain_label", "domain_plot"]].drop_duplicates()


def direction(change: float, tol: float = 1e-12) -> str:
    if not np.isfinite(change):
        return "missing"
    if change > tol:
        return "up"
    if change < -tol:
        return "down"
    return "no_change"


def main() -> int:
    exposures = selected_exposures()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel().copy()

    exposure_cols = [x for x in exposures["exposure"].unique() if x in panel.columns]
    annual = (
        panel[["state_alpha", "region", "year", *exposure_cols]]
        .replace([np.inf, -np.inf], np.nan)
        .groupby(["state_alpha", "region", "year"], as_index=False)
        .mean(numeric_only=True)
    )

    rows = []
    meta = exposures.set_index("exposure")
    for exposure in exposure_cols:
        for comp in COMPARISONS:
            b = annual.loc[
                annual["year"].eq(comp["baseline_year"]),
                ["state_alpha", "region", exposure],
            ].rename(columns={exposure: "baseline_value"})
            e = annual.loc[
                annual["year"].eq(comp["comparison_year"]),
                ["state_alpha", exposure],
            ].rename(columns={exposure: "comparison_value"})
            merged = b.merge(e, on="state_alpha", how="outer")
            merged["region"] = merged["region"].ffill().bfill()
            merged["change"] = merged["comparison_value"] - merged["baseline_value"]
            merged["pct_change"] = np.where(
                merged["baseline_value"].abs() > 1e-12,
                100 * merged["change"] / merged["baseline_value"],
                np.nan,
            )
            merged["direction"] = merged["change"].map(direction)
            for _, r in merged.iterrows():
                rows.append(
                    {
                        "stage": comp["stage"],
                        "baseline_year": comp["baseline_year"],
                        "comparison_year": comp["comparison_year"],
                        "state_alpha": r["state_alpha"],
                        "region": r["region"],
                        "domain_label": meta.loc[exposure, "domain_label"],
                        "domain_plot": meta.loc[exposure, "domain_plot"],
                        "exposure": exposure,
                        "baseline_value": r["baseline_value"],
                        "comparison_value": r["comparison_value"],
                        "change": r["change"],
                        "pct_change": r["pct_change"],
                        "direction": r["direction"],
                    }
                )

    out = pd.DataFrame(rows)
    out["domain_label"] = pd.Categorical(out["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    out = out.sort_values(["stage", "domain_label", "exposure", "state_alpha"])
    out.to_csv(OUT_LONG, index=False)

    ok = out[out["direction"].ne("missing")].copy()
    domain_state = (
        ok.groupby(["stage", "state_alpha", "region", "domain_label"], observed=True, as_index=False)
        .agg(
            n_exposures=("exposure", "nunique"),
            n_up=("direction", lambda x: int((x == "up").sum())),
            n_down=("direction", lambda x: int((x == "down").sum())),
            n_no_change=("direction", lambda x: int((x == "no_change").sum())),
            median_change=("change", "median"),
            median_pct_change=("pct_change", "median"),
        )
    )
    domain_state["frac_up"] = domain_state["n_up"] / domain_state["n_exposures"]
    domain_state["frac_down"] = domain_state["n_down"] / domain_state["n_exposures"]
    domain_state.to_csv(OUT_DOMAIN_STATE, index=False)

    domain = (
        ok.groupby(["stage", "domain_label"], observed=True, as_index=False)
        .agg(
            n_state_exposure_pairs=("exposure", "size"),
            n_exposures=("exposure", "nunique"),
            n_states=("state_alpha", "nunique"),
            n_up=("direction", lambda x: int((x == "up").sum())),
            n_down=("direction", lambda x: int((x == "down").sum())),
            n_no_change=("direction", lambda x: int((x == "no_change").sum())),
            median_change=("change", "median"),
            median_pct_change=("pct_change", "median"),
        )
    )
    domain["frac_up"] = domain["n_up"] / domain["n_state_exposure_pairs"]
    domain["frac_down"] = domain["n_down"] / domain["n_state_exposure_pairs"]
    domain.to_csv(OUT_DOMAIN, index=False)

    print(f"Wrote {OUT_LONG}")
    print(f"Wrote {OUT_DOMAIN_STATE}")
    print(f"Wrote {OUT_DOMAIN}")
    print(domain.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
