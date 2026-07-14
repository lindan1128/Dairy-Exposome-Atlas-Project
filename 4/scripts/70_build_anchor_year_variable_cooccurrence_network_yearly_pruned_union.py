#!/usr/bin/env python3
"""Anchor-year variable co-occurrence network with full-year pruning union.

This sensitivity mirrors script 68 but changes the redundancy-pruning step:

1. within each year from 2000-2025 and subdomain, identify highly redundant variables
   using state-level annual Spearman correlations;
2. retain the variable with strongest same-year per-cow ExWAS association in
   each redundancy cluster;
3. take the union of retained variables across all years;
4. rebuild anchor-year networks using this fixed full-year union variable set.

This lets the variable-reduction step vary by year while keeping the plotted
network node universe constant across years.
"""

from __future__ import annotations

import importlib.util
from itertools import combinations
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "anchor_variable_network",
    SCRIPT_DIR / "68_build_anchor_year_variable_cooccurrence_network.py",
)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

TAB4 = base.TAB4
EXPOSE_NEW = base.ROOT / "data" / "us_expose_new" / "processed"
if (EXPOSE_NEW / "exposure_state_month_expanded.csv").exists():
    base.L.US_EXPOSE = EXPOSE_NEW
NETWORK_YEARS = list(base.ANCHOR_YEARS)
SELECTION_YEARS = list(range(2000, 2026))
MIN_PAIR_N = base.MIN_PAIR_N
EDGE_Q_CUT = base.EDGE_P_CUT
REDUNDANT_R_CUT = base.REDUNDANT_R_CUT
REDUNDANT_P_CUT = base.REDUNDANT_P_CUT


def exwas_rank_for_year(exwas: pd.DataFrame, year: int) -> pd.DataFrame:
    d = exwas[exwas["year"].eq(year)].copy()
    d["year_neglogp"] = -np.log10(d["p"].clip(lower=1e-300))
    d["year_abs_beta"] = d["beta"].abs()
    d["year_rank_score"] = d["year_neglogp"].fillna(-np.inf) + 1e-9 * d["year_abs_beta"].fillna(0)
    return d[["exposure", "year_neglogp", "year_abs_beta", "year_rank_score"]]


def select_representatives_for_year(
    variables: pd.DataFrame,
    annual_x: pd.DataFrame,
    exwas: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selected representatives, cluster membership, and redundancy edges."""
    year_x = annual_x[annual_x["year"].eq(year)].copy()
    rank = exwas_rank_for_year(exwas, year)
    vmeta = variables.merge(rank, on="exposure", how="left")
    for col in [
        "year_rank_score",
        "year_neglogp",
        "year_abs_beta",
        "full_exwas_rank_score",
        "full_exwas_incr_r2",
        "full_exwas_abs_beta",
    ]:
        if col in vmeta.columns:
            vmeta[col] = pd.to_numeric(vmeta[col], errors="coerce")
    vmeta["year_rank_score"] = vmeta["year_rank_score"].fillna(-np.inf)
    vmeta["year_neglogp"] = vmeta["year_neglogp"].fillna(0.0)
    vmeta["year_abs_beta"] = vmeta["year_abs_beta"].fillna(0.0)
    vmeta["full_exwas_rank_score"] = vmeta["full_exwas_rank_score"].fillna(-np.inf)
    vmeta["full_exwas_incr_r2"] = vmeta["full_exwas_incr_r2"].fillna(0.0)
    vmeta["full_exwas_abs_beta"] = vmeta["full_exwas_abs_beta"].fillna(0.0)
    kept_rows = []
    cluster_rows = []
    edge_rows = []
    cluster_id = 0

    for subdomain, vdf in vmeta.groupby("subdomain_label", sort=False):
        vars_in = [v for v in vdf["exposure"].tolist() if v in year_x.columns]
        g = nx.Graph()
        g.add_nodes_from(vars_in)
        for a, b in combinations(vars_in, 2):
            pair = year_x[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < MIN_PAIR_N or pair[a].nunique() <= 1 or pair[b].nunique() <= 1:
                continue
            rho, p = stats.spearmanr(pair[a].to_numpy(float), pair[b].to_numpy(float))
            if np.isfinite(rho) and np.isfinite(p) and abs(rho) > REDUNDANT_R_CUT and p < REDUNDANT_P_CUT:
                g.add_edge(a, b, spearman_r=float(rho), p=float(p), n=int(len(pair)))
                edge_rows.append(
                    {
                        "year": year,
                        "subdomain_label": subdomain,
                        "exposure_a": a,
                        "exposure_b": b,
                        "spearman_r": float(rho),
                        "p": float(p),
                        "n": int(len(pair)),
                    }
                )

        for component in nx.connected_components(g):
            cluster_id += 1
            comp = sorted(component)
            comp_df = vdf[vdf["exposure"].isin(comp)].copy()
            comp_df = comp_df.sort_values(
                [
                    "year_rank_score",
                    "full_exwas_rank_score",
                    "full_exwas_incr_r2",
                    "full_exwas_abs_beta",
                    "exposure",
                ],
                ascending=[False, False, False, False, True],
            )
            representative = comp_df.iloc[0]["exposure"]
            for _, row in comp_df.iterrows():
                cluster_rows.append(
                    {
                        **row.to_dict(),
                        "year": year,
                        "yearly_redundancy_cluster_id": f"{year}_{cluster_id}",
                        "yearly_redundancy_cluster_size": len(comp),
                        "yearly_redundancy_representative": representative,
                        "is_year_representative": row["exposure"] == representative,
                        "yearly_redundancy_rule": f"|Spearman rho|>{REDUNDANT_R_CUT} and p<{REDUNDANT_P_CUT}",
                    }
                )
            kept_rows.append({**comp_df.iloc[0].to_dict(), "year": year})

    return pd.DataFrame(kept_rows), pd.DataFrame(cluster_rows), pd.DataFrame(edge_rows)


def main() -> int:
    old_anchor_years = list(base.ANCHOR_YEARS)
    base.ANCHOR_YEARS = SELECTION_YEARS
    variables_raw = base.load_clean_variables()
    annual_x = base.load_annual_exposure_matrix(variables_raw)
    exwas = base.load_exwas()
    base.ANCHOR_YEARS = old_anchor_years

    selected_rows = []
    cluster_rows = []
    redundancy_edges = []
    for year in SELECTION_YEARS:
        selected, clusters, redges = select_representatives_for_year(variables_raw, annual_x, exwas, year)
        selected_rows.append(selected)
        cluster_rows.append(clusters)
        redundancy_edges.append(redges)

    selected_by_year = pd.concat(selected_rows, ignore_index=True)
    clusters_all = pd.concat(cluster_rows, ignore_index=True)
    redges_all = pd.concat(redundancy_edges, ignore_index=True)
    union_vars = sorted(selected_by_year["exposure"].dropna().unique())
    union = variables_raw[variables_raw["exposure"].isin(union_vars)].copy()
    union["n_years_selected"] = union["exposure"].map(selected_by_year.groupby("exposure")["year"].nunique())
    # Backward-compatible alias used by downstream plotting/sorting scripts.
    union["n_anchor_years_selected"] = union["n_years_selected"]
    union = union.sort_values(["domain", "subdomain_label", "exposure"]).reset_index(drop=True)

    selected_by_year.to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_selected_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    clusters_all.to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    redges_all.to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_redundancy_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )
    union.to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        union.groupby(["domain_label", "subdomain_label"], dropna=False)
        .agg(n_union_variables=("exposure", "nunique"), median_selected_years=("n_years_selected", "median"))
        .reset_index()
        .to_csv(
            TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_subdomain_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )

    old_anchor_years = list(base.ANCHOR_YEARS)
    base.ANCHOR_YEARS = SELECTION_YEARS
    annual_y = base.load_annual_percow()
    base.ANCHOR_YEARS = old_anchor_years
    panel = annual_y.merge(annual_x, on=["state_alpha", "year"], how="inner")

    node_rows = []
    edge_rows = []
    summaries = []
    for year in NETWORK_YEARS:
        yp = panel[panel["year"].eq(year)].copy()
        nodes = base.build_year_nodes(yp, union, exwas, year)
        edges = base.build_year_edges(yp, union, year)
        node_rows.append(nodes)
        edge_rows.append(edges)
        summaries.append(
            {
                "year": year,
                "n_states": int(yp["state_alpha"].nunique()),
                "n_variables": int(nodes["exposure"].nunique()),
                "n_edges_bh_fdr05": int(len(edges)),
                "mean_abs_edge_r": float(edges["spearman_r"].abs().mean()) if len(edges) else np.nan,
                "shown_top_n": 999,
            }
        )

    nodes_all = pd.concat(node_rows, ignore_index=True)
    edges_all = pd.concat(edge_rows, ignore_index=True)
    nodes_layout = base.add_layout(nodes_all, edges_all)
    shown = nodes_layout[nodes_layout["plot_backbone"].eq(True)][["year", "exposure"]].drop_duplicates()
    edges_plot = edges_all.merge(shown.rename(columns={"exposure": "exposure_a"}), on=["year", "exposure_a"], how="inner")
    edges_plot = edges_plot.merge(shown.rename(columns={"exposure": "exposure_b"}), on=["year", "exposure_b"], how="inner")

    nodes_layout.to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_nodes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    edges_all.to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )
    edges_plot.to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_edges_plot_backbone.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summaries).to_csv(
        TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Compare with the pooled-pruning representative set if already available.
    pooled_path = TAB4 / "point4_anchor_year_variable_network_redundancy_clusters.csv"
    if pooled_path.exists():
        pooled = pd.read_csv(pooled_path)
        pooled_keep = set(pooled.loc[pooled["is_representative"].astype(bool), "exposure"])
        yearly_keep = set(union["exposure"])
        comparison = pd.DataFrame(
            {
                "metric": [
                    "pooled_pruned_representatives",
                    "yearly_pruned_union_representatives",
                    "intersection",
                    "pooled_only",
                    "yearly_union_only",
                    "jaccard_similarity",
                ],
                "value": [
                    len(pooled_keep),
                    len(yearly_keep),
                    len(pooled_keep & yearly_keep),
                    len(pooled_keep - yearly_keep),
                    len(yearly_keep - pooled_keep),
                    len(pooled_keep & yearly_keep) / len(pooled_keep | yearly_keep) if pooled_keep | yearly_keep else np.nan,
                ],
            }
        )
        comparison.to_csv(
            TAB4 / "point4_anchor_year_variable_network_pruning_strategy_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("Wrote yearly-pruned-union variable network tables.")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Target variables:", variables_raw["exposure"].nunique())
    print(f"Selection years: {min(SELECTION_YEARS)}-{max(SELECTION_YEARS)} ({len(SELECTION_YEARS)} years)")
    print("Full-year-pruned union variables:", union["exposure"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
