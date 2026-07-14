#!/usr/bin/env python3
"""Yearly network-density summary for the fixed nonredundant variable universe."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[5]
STAT4 = ROOT / "analysis" / "statistics" / "4"
TAB4 = STAT4 / "tables"
DATA = ROOT / "data"

STATE26 = {
    "AZ", "CA", "CO", "FL", "GA", "IA", "ID", "IL", "IN", "KS", "KY", "MI",
    "MN", "MO", "NM", "NY", "OH", "OR", "PA", "SD", "TX", "UT", "VA", "VT",
    "WA", "WI",
}
YEARS = list(range(2000, 2026))
MIN_PAIR_N = 12
EDGE_Q_CUT = 0.05


def main() -> int:
    nodes = pd.read_csv(TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_nodes.csv")
    exposures = sorted(nodes["exposure"].dropna().unique().tolist())

    exposure_file = DATA / "us_expose_new" / "processed" / "exposure_state_month_expanded.csv"
    available_cols = pd.read_csv(exposure_file, nrows=0).columns.tolist()
    exposures = [x for x in exposures if x in available_cols]

    exp = pd.read_csv(
        exposure_file,
        usecols=["state_alpha", "year", "month"] + exposures,
        low_memory=False,
    )
    exp = exp[exp["state_alpha"].isin(STATE26) & exp["year"].isin(YEARS)].copy()
    annual_x = exp.groupby(["state_alpha", "year"], dropna=False)[exposures].mean().reset_index()

    milk = pd.read_csv(DATA / "us_milk" / "processed" / "state_month_panel.csv", low_memory=False)
    annual_y = (
        milk.loc[
            milk["state_alpha"].isin(STATE26)
            & milk["year"].isin(YEARS)
            & (milk["milk_cows_head"] > 0)
            & (milk["milk_per_cow_lb"] > 0),
            ["state_alpha", "year", "month"],
        ]
        .groupby(["state_alpha", "year"], dropna=False)
        .agg(milk_months=("month", "size"))
        .reset_index()
    )
    panel = annual_y.merge(annual_x, on=["state_alpha", "year"], how="inner")

    rows = []
    for year in YEARS:
        yp = panel[panel["year"].eq(year)].copy()
        edge_rows = []
        for a, b in combinations(exposures, 2):
            d = yp[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < MIN_PAIR_N or d[a].nunique() <= 1 or d[b].nunique() <= 1:
                continue
            rho, p = stats.spearmanr(d[a].to_numpy(float), d[b].to_numpy(float))
            if np.isfinite(rho) and np.isfinite(p):
                edge_rows.append({"spearman_r": float(rho), "p": float(p)})
        edges = pd.DataFrame(edge_rows)
        if not edges.empty:
            edges = edges.sort_values("p").reset_index(drop=True)
            m = len(edges)
            ranks = np.arange(1, m + 1, dtype=float)
            q = edges["p"].to_numpy(float) * m / ranks
            edges["q_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
            edges = edges[edges["q_bh"] < EDGE_Q_CUT].copy()
        rows.append(
            {
                "year": year,
                "n_states": int(yp["state_alpha"].nunique()),
                "n_variables": len(exposures),
                "n_edges_bh_fdr05": int(len(edges)),
                "mean_abs_edge_r": float(edges["spearman_r"].abs().mean()) if len(edges) else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TAB4 / "point4_yearly_network_density_summary.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
