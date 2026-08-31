#!/usr/bin/env python3
"""Nonredundant state-exposure trends for two stages.

Within each domain, select variables that are not obvious transformed duplicates
and keep pairwise |r| < 0.7 on annual state values. Then estimate per-state
linear trends for 2000-2014 and 2015-2024.
"""

from __future__ import annotations

from pathlib import Path
import re
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
OUT_SELECTED = TAB / "point2_nonredundant_exposure_variable_selection.csv"
OUT_STATE_TRENDS = TAB / "point2_nonredundant_state_exposure_trends_by_stage.csv"
OUT_VARIABLE_SUMMARY = TAB / "point2_nonredundant_variable_trend_summary_by_stage.csv"
OUT_DOMAIN_SUMMARY = TAB / "point2_nonredundant_domain_trend_summary_by_stage.csv"

DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
STAGES = {
    "2000_2014": (2000, 2014),
    "2015_2024": (2015, 2024),
}
MAX_ABS_CORR = 0.70


def canonical_definition(exposure: str) -> str:
    x = exposure
    x = re.sub(r"_state_month_(anomaly|robust_z)$", "", x)
    x = re.sub(r"_index_2000_2004$", "", x)
    x = re.sub(r"_index_2000$", "", x)
    x = re.sub(r"_log1p$", "", x)
    x = re.sub(r"^market_log_", "market_", x)
    x = re.sub(r"^herd_log_", "herd_", x)
    x = re.sub(r"_pct$", "", x)
    return x


def variable_priority(row: pd.Series) -> tuple:
    p = np.nanmin([row.get("per_cow_p", np.nan), row.get("total_p", np.nan), row.get("cows_p", np.nan)])
    p = p if np.isfinite(p) and p > 0 else 1.0
    exposure = row["exposure"]
    transformed_penalty = int(
        any(
            tag in exposure
            for tag in ["state_month_anomaly", "state_month_robust_z", "index_2000", "log1p", "market_log_"]
        )
    )
    return (transformed_penalty, p, len(exposure), exposure)


def selected_candidates() -> pd.DataFrame:
    d = pd.read_csv(DECOMP)
    d["domain_label"] = d["domain_plot"].replace({"Forage condition": "Forage"})
    d = d[
        d["domain_label"].isin(DOMAIN_ORDER)
        & ((d["total_p"] < 0.05) | (d["per_cow_p"] < 0.05) | (d["cows_p"] < 0.05))
    ].copy()
    d["definition_group"] = d["exposure"].map(canonical_definition)
    d["priority_key"] = d.apply(variable_priority, axis=1)
    # Keep one representative of obvious transformed duplicates before correlation filtering.
    reps = (
        d.sort_values(["domain_label", "definition_group", "priority_key"])
        .groupby(["domain_label", "definition_group"], as_index=False)
        .head(1)
        .copy()
    )
    return reps


def annual_panel(exposures: list[str]) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel().copy()
    cols = ["state_alpha", "region", "year", *exposures]
    return (
        panel[cols]
        .replace([np.inf, -np.inf], np.nan)
        .groupby(["state_alpha", "region", "year"], as_index=False)
        .mean(numeric_only=True)
    )


def pairwise_corr_vector(annual: pd.DataFrame, exposure: str) -> pd.Series:
    return pd.to_numeric(annual[exposure], errors="coerce")


def select_by_correlation(candidates: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, g in candidates.groupby("domain_label", sort=False):
        g = g.sort_values("priority_key").copy()
        kept: list[str] = []
        vectors = {x: pairwise_corr_vector(annual, x) for x in g["exposure"] if x in annual.columns}
        for _, row in g.iterrows():
            exposure = row["exposure"]
            if exposure not in vectors:
                continue
            max_corr = 0.0
            blocking = ""
            for kept_exposure in kept:
                pair = pd.concat([vectors[exposure], vectors[kept_exposure]], axis=1).dropna()
                if len(pair) < 30:
                    corr = np.nan
                else:
                    corr = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                if np.isfinite(corr) and abs(corr) > max_corr:
                    max_corr = abs(corr)
                    blocking = kept_exposure
            if max_corr < MAX_ABS_CORR:
                kept.append(exposure)
                status = "kept"
            else:
                status = "dropped_corr_ge_0.7"
            out = row.drop(labels=["priority_key"]).to_dict()
            out.update(
                {
                    "selection_status": status,
                    "max_abs_corr_to_kept": max_corr if kept else np.nan,
                    "blocking_kept_exposure": blocking,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def slope_direction(s: pd.DataFrame, exposure: str, start: int, end: int) -> dict:
    d = s[(s["year"] >= start) & (s["year"] <= end)][["year", exposure]].dropna().sort_values("year")
    if d["year"].nunique() < 5:
        return {
            "n_years": int(d["year"].nunique()),
            "slope_per_year": np.nan,
            "direction": "missing",
            "fitted_start": np.nan,
            "fitted_end": np.nan,
            "observed_start": np.nan,
            "observed_end": np.nan,
        }
    x = d["year"].to_numpy(float)
    y = d[exposure].to_numpy(float)
    coef = np.polyfit(x, y, 1)
    slope = float(coef[0])
    if slope > 1e-12:
        direction = "up"
    elif slope < -1e-12:
        direction = "down"
    else:
        direction = "flat"
    return {
        "n_years": int(d["year"].nunique()),
        "slope_per_year": slope,
        "direction": direction,
        "fitted_start": float(coef[0] * start + coef[1]),
        "fitted_end": float(coef[0] * end + coef[1]),
        "observed_start": float(d.loc[d["year"].idxmin(), exposure]),
        "observed_end": float(d.loc[d["year"].idxmax(), exposure]),
    }


def build_trends(annual: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    kept = selected[selected["selection_status"].eq("kept")].copy()
    meta = kept.set_index("exposure")
    rows = []
    for exposure in kept["exposure"]:
        for (state, region), s in annual.groupby(["state_alpha", "region"]):
            for stage, (start, end) in STAGES.items():
                res = slope_direction(s, exposure, start, end)
                rows.append(
                    {
                        "stage": stage,
                        "stage_start": start,
                        "stage_end": end,
                        "state_alpha": state,
                        "region": region,
                        "domain_label": meta.loc[exposure, "domain_label"],
                        "domain_plot": meta.loc[exposure, "domain_plot"],
                        "mechanistic_domain_en": meta.loc[exposure, "mechanistic_domain_en"],
                        "definition_group": meta.loc[exposure, "definition_group"],
                        "exposure": exposure,
                        **res,
                    }
                )
    out = pd.DataFrame(rows)
    out["domain_label"] = pd.Categorical(out["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    return out.sort_values(["stage", "domain_label", "exposure", "state_alpha"])


def summarize(trends: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = trends[trends["direction"].ne("missing")].copy()
    variable = (
        ok.groupby(
            ["stage", "domain_label", "mechanistic_domain_en", "exposure"],
            observed=True,
            as_index=False,
        )
        .agg(
            n_states=("state_alpha", "nunique"),
            n_up=("direction", lambda x: int((x == "up").sum())),
            n_down=("direction", lambda x: int((x == "down").sum())),
            n_flat=("direction", lambda x: int((x == "flat").sum())),
            median_slope_per_year=("slope_per_year", "median"),
            median_fitted_change=("fitted_end", "median"),
            median_start=("fitted_start", "median"),
        )
    )
    variable["median_fitted_change"] = (
        ok.groupby(["stage", "domain_label", "mechanistic_domain_en", "exposure"], observed=True)
        .apply(lambda g: (g["fitted_end"] - g["fitted_start"]).median(), include_groups=False)
        .to_numpy()
    )
    variable["frac_up"] = variable["n_up"] / variable["n_states"]
    variable["frac_down"] = variable["n_down"] / variable["n_states"]
    variable["dominant_direction"] = np.select(
        [variable["frac_up"] >= 0.6, variable["frac_down"] >= 0.6],
        ["mostly_up", "mostly_down"],
        default="mixed",
    )

    domain = (
        ok.groupby(["stage", "domain_label"], observed=True, as_index=False)
        .agg(
            n_state_exposure_pairs=("exposure", "size"),
            n_exposures=("exposure", "nunique"),
            n_states=("state_alpha", "nunique"),
            n_up=("direction", lambda x: int((x == "up").sum())),
            n_down=("direction", lambda x: int((x == "down").sum())),
            n_flat=("direction", lambda x: int((x == "flat").sum())),
            median_slope_per_year=("slope_per_year", "median"),
        )
    )
    domain["frac_up"] = domain["n_up"] / domain["n_state_exposure_pairs"]
    domain["frac_down"] = domain["n_down"] / domain["n_state_exposure_pairs"]
    return variable, domain


def main() -> int:
    candidates = selected_candidates()
    available = candidates["exposure"].tolist()
    annual = annual_panel(available)
    selected = select_by_correlation(candidates, annual)
    kept = selected[selected["selection_status"].eq("kept")]["exposure"].tolist()
    trends = build_trends(annual, selected)
    variable, domain = summarize(trends)

    selected.to_csv(OUT_SELECTED, index=False)
    trends.to_csv(OUT_STATE_TRENDS, index=False)
    variable.to_csv(OUT_VARIABLE_SUMMARY, index=False)
    domain.to_csv(OUT_DOMAIN_SUMMARY, index=False)

    print(f"Wrote {OUT_SELECTED}")
    print(f"Wrote {OUT_STATE_TRENDS}")
    print(f"Wrote {OUT_VARIABLE_SUMMARY}")
    print(f"Wrote {OUT_DOMAIN_SUMMARY}")
    print(f"Candidate variables: {len(candidates)}; kept nonredundant variables: {len(kept)}")
    print("\nKept variables by domain")
    print(selected[selected["selection_status"].eq("kept")].groupby("domain_label", observed=True)["exposure"].nunique().to_string())
    print("\nDomain summary")
    print(domain.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
