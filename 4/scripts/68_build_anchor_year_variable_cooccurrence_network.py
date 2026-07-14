#!/usr/bin/env python3
"""Anchor-year variable-level exposome co-occurrence networks.

* nodes are clean native exposure variables;
* node size is the variable's single-variable Delta R2 with annual per-cow milk
  yield in that anchor year;
* node border encodes the standard Point-1 year-varying ExWAS association with
  per-cow milk;
* edges are annual state-level Spearman correlations between exposure variables,
  retained at BH-FDR q < 0.05 within each year.

The full node/edge tables are written for all clean variables. A plotting
backbone (top annual single-variable nodes per year) is also written to keep the figure
legible.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx

ROOT = Path(__file__).resolve().parents[5]
STAT = ROOT / "analysis" / "statistics"
TAB1 = STAT / "1" / "tables"
TAB4 = STAT / "4" / "tables"
SUPP2_CLEAN_EXWAS_PATH = (
    ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

ANCHOR_YEARS = [2000, 2005, 2010, 2015, 2020, 2025]
PLOT_TOP_N = 999
MIN_PAIR_N = 12
EDGE_P_CUT = 0.05
REDUNDANT_R_CUT = 0.90
REDUNDANT_P_CUT = 0.05

CLEAN_CLASSES = [
    "Nature and climate",
    "Forage and pasture condition",
    "Chemical and pollution exposome",
    "Market and production-system",
]

DOMAIN_LABELS = {
    "Heat": "Heat",
    "Cold": "Cold",
    "Severe weather": "Severe weather",
    "Forage condition": "Forage",
    "Agricultural pesticides": "Pesticides",
    "Feed market": "Feed market",
    "Milk price / dairy market": "Dairy market",
    "Dairy market": "Dairy market",
    "Market demand": "Market demand",
    "Herd scale": "Herd scale",
    "Dairy scale": "Dairy scale",
}

TARGET_SUBDOMAINS = {
    "Dry heat",
    "Humid heat",
    "Night heat",
    "Dry cold",
    "Night cold",
    "Wet/snowy cold",
    "Fire events",
    "Flood events",
    "Hail events",
    "Wind events",
    "Hay condition",
    "Pasture condition",
    "Pesticide burden",
    "Pesticide diversity",
    "Grain prices",
    "Hay prices",
    "Alfalfa premium",
    "Feed price ratios",
    "Feed price index",
    "All-milk price",
    "Fluid-grade milk price",
    "Manufacturing-grade milk price",
    "Milk-feed ratio",
    "Population size",
    "Population growth",
    "Population share",
}

STATE26 = {
    "AZ", "CA", "CO", "FL", "GA", "IA", "ID", "IL", "IN", "KS", "KY", "MI",
    "MN", "MO", "NM", "NY", "OH", "OR", "PA", "SD", "TX", "UT", "VA", "VT",
    "WA", "WI",
}


def zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    sd = np.nanstd(arr, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(arr)
    return (arr - np.nanmean(arr)) / sd


def safe_label(s: str, max_len: int = 30) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def classify_subdomain(domain: str, exposure: str, fallback: str) -> str:
    """Mechanistic subdomain labels for the variable-level network.

    The Point-1 dictionary stores some domains at a deliberately conservative
    level (for example all storm-event variables under one severe-weather
    label). For the network, a finer label helps group like-with-like without
    changing the clean variable pool.
    """
    x = str(exposure).lower()
    if domain == "Heat":
        if "dry" in str(fallback).lower() or "vpd" in str(fallback).lower() or "aridity" in str(fallback).lower():
            return "Dry heat"
        if "humid" in str(fallback).lower() or "wet-bulb" in str(fallback).lower():
            return "Humid heat"
        if "night" in str(fallback).lower():
            return "Night heat"
        return fallback
    if domain == "Cold":
        if "dry" in str(fallback).lower():
            return "Dry cold"
        if "night" in str(fallback).lower() or "no-thaw" in str(fallback).lower():
            return "Night cold"
        if "wet" in str(fallback).lower() or "snow" in str(fallback).lower():
            return "Wet/snowy cold"
        return fallback
    if domain == "Severe weather":
        if "damage" in x:
            return "Damage cost"
        if "wind" in x:
            return "Wind events"
        if "hail" in x:
            return "Hail events"
        if "flood" in x:
            return "Flood events"
        if "ice" in x or "winter" in x or "cold" in x:
            return "Winter / ice / cold events"
        if "fire" in x:
            return "Fire events"
        if "heat" in x:
            return "Heat disaster events"
        return "General storm-disaster burden"
    if domain == "Forage condition":
        if "pastureland" in x:
            return "Pasture condition"
        if "hay" in x:
            return "Hay condition"
        return "Composite forage condition"
    if domain == "Feed market":
        if "corn" in x and "ratio" not in x:
            return "Grain prices"
        if "soybeans" in x and "ratio" not in x:
            return "Grain prices"
        if "alfalfa_hay_price_ratio" in x or "alfalfa_hay_price_spread" in x or "alfalfa_premium" in x:
            return "Alfalfa premium"
        if "hay_alfalfa" in x:
            return "Hay prices"
        if "hay_excl_alfalfa" in x:
            return "Hay prices"
        if "hay_price" in x:
            return "Hay prices"
        if "ratio" in x:
            return "Feed price ratios"
        if "feed_price_index" in x:
            return "Feed price index"
        return "Feed market"
    if domain == "Milk price / dairy market":
        if "feed_ratio" in x:
            return "Milk-feed ratio"
        if "fluid" in x:
            return "Fluid-grade milk price"
        if "manufacturing" in x:
            return "Manufacturing-grade milk price"
        return "All-milk price"
    if domain == "Market demand":
        if "growth" in x or "change" in x:
            return "Population growth"
        if "share" in x:
            return "Population share"
        return "Population size"
    if domain == "Herd structure / scale":
        if "concentration" in x or "share" in x:
            return "Herd spatial concentration"
        if "avg" in x or "per" in x:
            return "Farm/herd scale"
        return "Cow inventory scale"
    if domain == "Agricultural pesticides":
        if "log" in x:
            return "Pesticide burden log"
        if "intensity" in x or "rate" in x:
            return "Pesticide use intensity"
        if "compound" in x:
            return "Pesticide diversity"
        return "Pesticide burden"
    return fallback


def load_clean_variables() -> pd.DataFrame:
    clean = pd.read_excel(SUPP2_CLEAN_EXWAS_PATH, sheet_name="exwas_variables")
    clean = clean[
        clean["class"].isin(CLEAN_CLASSES)
        & clean["used_in_exwas"].astype(bool)
    ].copy()
    clean["source_class"] = clean["class"]
    clean["domain_label"] = clean["domain"].map(DOMAIN_LABELS).fillna(clean["domain"])
    clean["subdomain_label"] = clean["Subdomain"]
    clean["exposure"] = clean["variables_en"]
    clean["exposure_zh"] = clean["variables_ch"]
    clean = clean[~clean["subdomain_label"].eq("Hay condition")].copy()

    assoc_cols = [
        "source_class",
        "domain",
        "mechanistic_domain_en",
        "exposure",
        "exposure_zh",
        "construct",
        "form",
        "p",
        "beta",
        "incr_r2",
        "plot_p",
        "plot_incr_r2",
    ]
    assoc = pd.read_csv(
        TAB1 / "point1_native_only_endpoint_exwas_associations.csv",
        usecols=lambda c: c in set(assoc_cols + ["phenotype_scope", "window"]),
        low_memory=False,
    )
    assoc = assoc[
        assoc["phenotype_scope"].eq("per_cow_26")
        & assoc["window"].eq("native")
        & assoc["exposure"].isin(clean["exposure"])
    ].copy()
    assoc["plot_p_sort"] = pd.to_numeric(assoc["plot_p"].fillna(assoc["p"]), errors="coerce").fillna(np.inf)
    assoc["plot_incr_r2_sort"] = pd.to_numeric(
        assoc["plot_incr_r2"].fillna(assoc["incr_r2"]), errors="coerce"
    ).fillna(-np.inf)
    assoc = (
        assoc.sort_values(["exposure", "plot_p_sort", "plot_incr_r2_sort"], ascending=[True, True, False])
        .drop_duplicates("exposure")
        .drop(columns=["phenotype_scope", "window", "plot_p_sort", "plot_incr_r2_sort"])
    )

    metric_cols = ["exposure", "p", "beta", "incr_r2", "plot_p", "plot_incr_r2"]
    variables = clean.merge(assoc[metric_cols], on="exposure", how="left")
    variables["full_exwas_p"] = pd.to_numeric(variables["plot_p"].fillna(variables["p"]), errors="coerce")
    variables["full_exwas_p"] = variables["full_exwas_p"].fillna(1.0)
    variables["full_exwas_incr_r2"] = pd.to_numeric(
        variables["plot_incr_r2"].fillna(variables["incr_r2"]), errors="coerce"
    ).fillna(0.0)
    variables["full_exwas_neglogp"] = -np.log10(variables["full_exwas_p"].clip(lower=1e-300))
    variables["full_exwas_abs_beta"] = pd.to_numeric(variables["beta"], errors="coerce").abs().fillna(0.0)
    variables["full_exwas_rank_score"] = (
        variables["full_exwas_neglogp"].fillna(-np.inf)
        + 1e-6 * variables["full_exwas_incr_r2"].fillna(0)
        + 1e-9 * variables["full_exwas_abs_beta"].fillna(0)
    )
    return variables.sort_values(["domain", "subdomain_label", "exposure"]).reset_index(drop=True)


def load_monthly_exposure_subset(variables: pd.DataFrame) -> pd.DataFrame:
    requested = variables["exposure"].tolist()
    exp_cols = pd.read_csv(L.US_EXPOSE / "exposure_state_month_expanded.csv", nrows=0).columns.tolist()
    available = [c for c in requested if c in exp_cols]
    use_cols = ["state_alpha", "year", "month"] + available
    exp = pd.read_csv(
        L.US_EXPOSE / "exposure_state_month_expanded.csv",
        usecols=use_cols,
        low_memory=False,
    )
    return exp[exp["state_alpha"].isin(STATE26) & exp["year"].between(2000, 2025)].copy()


def prune_redundant_variables(variables: pd.DataFrame) -> pd.DataFrame:
    """Keep one milk-associated representative per highly redundant subdomain cluster."""
    exp = load_monthly_exposure_subset(variables)
    kept_rows = []
    cluster_rows = []
    edge_rows = []
    cluster_id = 0

    for subdomain, vdf in variables.groupby("subdomain_label", sort=False):
        vars_in = [v for v in vdf["exposure"].tolist() if v in exp.columns]
        g = nx.Graph()
        g.add_nodes_from(vars_in)
        for a, b in combinations(vars_in, 2):
            pair = exp[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 50 or pair[a].nunique() <= 1 or pair[b].nunique() <= 1:
                continue
            rho, p = stats.spearmanr(pair[a].to_numpy(float), pair[b].to_numpy(float))
            if np.isfinite(rho) and np.isfinite(p) and abs(rho) > REDUNDANT_R_CUT and p < REDUNDANT_P_CUT:
                g.add_edge(a, b, spearman_r=float(rho), p=float(p), n=int(len(pair)))
                edge_rows.append(
                    {
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
                ["full_exwas_rank_score", "full_exwas_incr_r2", "full_exwas_abs_beta", "exposure"],
                ascending=[False, False, False, True],
            )
            representative = comp_df.iloc[0]["exposure"]
            for _, row in comp_df.iterrows():
                cluster_rows.append(
                    {
                        **row.to_dict(),
                        "redundancy_cluster_id": cluster_id,
                        "redundancy_cluster_size": len(comp),
                        "redundancy_representative": representative,
                        "is_representative": row["exposure"] == representative,
                        "redundancy_rule": f"|Spearman rho|>{REDUNDANT_R_CUT} and p<{REDUNDANT_P_CUT}",
                    }
                )
            kept_rows.append(comp_df.iloc[0].to_dict())

    clusters = pd.DataFrame(cluster_rows)
    edges = pd.DataFrame(edge_rows)
    kept = pd.DataFrame(kept_rows).sort_values(["domain", "subdomain_label", "exposure"]).reset_index(drop=True)
    clusters.to_csv(
        TAB4 / "point4_anchor_year_variable_network_redundancy_clusters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    edges.to_csv(
        TAB4 / "point4_anchor_year_variable_network_redundancy_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        clusters.groupby(["domain_label", "subdomain_label"], dropna=False)
        .agg(
            n_input_variables=("exposure", "nunique"),
            n_representatives=("is_representative", "sum"),
            n_redundancy_clusters=("redundancy_cluster_id", "nunique"),
            max_cluster_size=("redundancy_cluster_size", "max"),
        )
        .reset_index()
        .to_csv(
            TAB4 / "point4_anchor_year_variable_network_nonredundant_subdomain_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )
    return kept


def load_annual_exposure_matrix(variables: pd.DataFrame) -> pd.DataFrame:
    requested = variables["exposure"].tolist()
    exp_cols = pd.read_csv(L.US_EXPOSE / "exposure_state_month_expanded.csv", nrows=0).columns.tolist()
    available = [c for c in requested if c in exp_cols]
    missing = sorted(set(requested) - set(available))
    if missing:
        pd.DataFrame({"missing_exposure": missing}).to_csv(
            TAB4 / "point4_anchor_year_variable_network_missing_exposures.csv",
            index=False,
            encoding="utf-8-sig",
        )

    use_cols = ["state_alpha", "year", "month"] + available
    exp = pd.read_csv(
        L.US_EXPOSE / "exposure_state_month_expanded.csv",
        usecols=use_cols,
        low_memory=False,
    )
    exp = exp[exp["state_alpha"].isin(STATE26) & exp["year"].isin(ANCHOR_YEARS)].copy()
    annual = exp.groupby(["state_alpha", "year"], dropna=False)[available].mean().reset_index()
    return annual


def load_annual_percow() -> pd.DataFrame:
    milk = pd.read_csv(L.US_MILK / "state_month_panel.csv", low_memory=False)
    d = milk.loc[
        milk["state_alpha"].isin(STATE26)
        & milk["year"].isin(ANCHOR_YEARS)
        & (milk["milk_cows_head"] > 0)
        & (milk["milk_per_cow_lb"] > 0),
        ["state_alpha", "year", "month", "milk_cows_head", "milk_per_cow_lb"],
    ].copy()
    out = (
        d.groupby(["state_alpha", "year"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "milk_per_cow_lb": np.average(g["milk_per_cow_lb"], weights=g["milk_cows_head"]),
                    "milk_cows_head": g["milk_cows_head"].mean(),
                    "milk_months": len(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return out


def weighted_single_r2(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < MIN_PAIR_N or np.nanstd(x[ok]) <= 1e-10 or np.nanstd(y[ok]) <= 1e-10:
        return np.nan
    yy = zscore(np.log(y[ok]))
    xx = zscore(x[ok])
    ww = w[ok] / np.nanmean(w[ok])
    X = np.column_stack([np.ones(ok.sum()), xx])
    sw = np.sqrt(ww)
    Xw = X * sw[:, None]
    yw = yy * sw
    beta = np.linalg.pinv(Xw.T @ Xw) @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    sse = float(resid @ resid)
    sst = float(((yw - np.average(yw)) ** 2).sum())
    return max(0.0, 1 - sse / sst) if sst > 0 else np.nan


def load_exwas() -> pd.DataFrame:
    exwas = pd.read_csv(TAB1 / "point1_chord_signal_yearly_variable_associations.csv", low_memory=False)
    exwas = exwas[
        exwas["phenotype_scope"].eq("per_cow_26")
        & exwas["year"].isin(ANCHOR_YEARS)
    ].copy()
    exwas["exwas_direction"] = np.where(
        exwas["beta"] > 0,
        "Positive milk association",
        np.where(exwas["beta"] < 0, "Negative milk association", "No direction"),
    )
    exwas["exwas_neglogp"] = -np.log10(exwas["p"].clip(lower=1e-300))
    return exwas[["year", "exposure", "beta", "se", "p", "exwas_direction", "exwas_neglogp"]]


def build_year_nodes(year_panel: pd.DataFrame, variables: pd.DataFrame, exwas: pd.DataFrame, year: int) -> pd.DataFrame:
    y = year_panel["milk_per_cow_lb"].to_numpy(float)
    w = year_panel["milk_cows_head"].to_numpy(float)
    rows = []
    for _, v in variables.iterrows():
        exposure = v["exposure"]
        if exposure not in year_panel.columns:
            continue
        x = pd.to_numeric(year_panel[exposure], errors="coerce").to_numpy(float)
        rows.append(
            {
                **v.to_dict(),
                "year": year,
                "n_states_exposure_year": int(np.isfinite(x).sum()),
                "single_variable_delta_r2": weighted_single_r2(y, x, w),
                "node_size_method": "single_variable_delta_r2",
            }
        )
    nodes = pd.DataFrame(rows)
    nodes = nodes.merge(exwas[exwas["year"].eq(year)], on=["year", "exposure"], how="left")
    nodes = nodes.rename(
        columns={
            "beta_y": "beta",
            "p_y": "p",
            "beta_x": "full_period_beta",
            "p_x": "full_period_p",
        }
    )
    nodes["single_variable_r2_pct"] = 100 * nodes["single_variable_delta_r2"].clip(lower=0)
    nodes["exposure_label"] = nodes["exposure"].map(lambda s: safe_label(str(s).replace("daymet_dairy_weighted_", "")))
    return nodes


def build_year_edges(year_panel: pd.DataFrame, variables: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    var_list = [v for v in variables["exposure"].tolist() if v in year_panel.columns]
    for a, b in combinations(var_list, 2):
        d = year_panel[["state_alpha", a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < MIN_PAIR_N or d[a].nunique() <= 1 or d[b].nunique() <= 1:
            continue
        rho, p = stats.spearmanr(d[a].to_numpy(float), d[b].to_numpy(float))
        if np.isfinite(rho) and np.isfinite(p):
            rows.append(
                {
                    "year": year,
                    "exposure_a": a,
                    "exposure_b": b,
                    "spearman_r": float(rho),
                    "p": float(p),
                    "n_states_pair": int(len(d)),
                    "edge_direction": "Positive co-occurrence" if rho > 0 else "Negative co-occurrence",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("p").reset_index(drop=True)
    m = len(out)
    ranks = np.arange(1, m + 1, dtype=float)
    q = out["p"].to_numpy(float) * m / ranks
    out["q_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
    return out[out["q_bh"] < EDGE_P_CUT].reset_index(drop=True)


def add_layout(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    layout_rows = []
    for year, ndf in nodes.groupby("year", sort=False):
        show = (
            ndf.sort_values(["single_variable_delta_r2", "exwas_neglogp"], ascending=False)
            .head(PLOT_TOP_N)
            .copy()
        )
        show_set = set(show["exposure"])
        edf = edges[
            edges["year"].eq(year)
            & edges["exposure_a"].isin(show_set)
            & edges["exposure_b"].isin(show_set)
        ].copy()
        g = nx.Graph()
        for exp in show["exposure"]:
            g.add_node(exp)
        for _, e in edf.iterrows():
            g.add_edge(e["exposure_a"], e["exposure_b"], weight=abs(float(e["spearman_r"])))
        pos = nx.spring_layout(g, seed=year, k=0.55, iterations=300, weight="weight")
        for exp, (x, y) in pos.items():
            layout_rows.append({"year": year, "exposure": exp, "x": x, "y": y, "plot_backbone": True})
    layout = pd.DataFrame(layout_rows)
    return nodes.merge(layout, on=["year", "exposure"], how="left")


def main() -> int:
    variables_raw = load_clean_variables()
    variables = prune_redundant_variables(variables_raw)
    annual_x = load_annual_exposure_matrix(variables)
    annual_y = load_annual_percow()
    panel = annual_y.merge(annual_x, on=["state_alpha", "year"], how="inner")
    exwas = load_exwas()

    node_rows = []
    edge_rows = []
    summaries = []
    for year in ANCHOR_YEARS:
        yp = panel[panel["year"].eq(year)].copy()
        nodes = build_year_nodes(yp, variables, exwas, year)
        edges = build_year_edges(yp, variables, year)
        node_rows.append(nodes)
        edge_rows.append(edges)
        summaries.append(
            {
                "year": year,
                "n_states": int(yp["state_alpha"].nunique()),
                "n_variables": int(nodes["exposure"].nunique()),
                "n_edges_bh_fdr05": int(len(edges)),
                "mean_abs_edge_r": float(edges["spearman_r"].abs().mean()) if len(edges) else np.nan,
                "shown_top_n": PLOT_TOP_N,
            }
        )

    nodes_all = pd.concat(node_rows, ignore_index=True)
    edges_all = pd.concat(edge_rows, ignore_index=True)
    nodes_layout = add_layout(nodes_all, edges_all)
    shown = nodes_layout[nodes_layout["plot_backbone"].eq(True)][["year", "exposure"]].drop_duplicates()
    edges_plot = edges_all.merge(shown.rename(columns={"exposure": "exposure_a"}), on=["year", "exposure_a"], how="inner")
    edges_plot = edges_plot.merge(shown.rename(columns={"exposure": "exposure_b"}), on=["year", "exposure_b"], how="inner")

    nodes_layout.to_csv(
        TAB4 / "point4_anchor_year_variable_network_nodes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        nodes_layout.drop_duplicates(["domain_label", "subdomain_label", "exposure"])
        .groupby(["domain_label", "subdomain_label"], dropna=False)
        .agg(n_variables=("exposure", "nunique"))
        .reset_index()
        .sort_values(["domain_label", "subdomain_label"])
        .to_csv(
            TAB4 / "point4_anchor_year_variable_network_subdomain_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    )
    edges_all.to_csv(
        TAB4 / "point4_anchor_year_variable_network_edges.csv",
        index=False,
        encoding="utf-8-sig",
    )
    edges_plot.to_csv(
        TAB4 / "point4_anchor_year_variable_network_edges_plot_backbone.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summaries).to_csv(
        TAB4 / "point4_anchor_year_variable_network_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Wrote variable-level anchor-year co-occurrence network tables.")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("Target subdomain variables before pruning:", variables_raw["exposure"].nunique())
    print("Nonredundant representative variables:", variables["exposure"].nunique())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
