#!/usr/bin/env python3
"""Annual sparse exposome incremental R2 beyond Census region.

This follows the variance-explained logic of phenome-exposome atlas analyses:

1. start from the clean curated macro-exposome variables;
2. use the annual union of yearly redundancy-pruned representatives;
3. rank those nonredundant variables by annual single-variable Delta R2;
4. keep at most one representative per subdomain;
5. fit a sparse multiexposure model using the top K variables;
6. calculate incremental R2 beyond a Census-region baseline;
7. test the aggregate incremental R2 with a nested F-test.

The analysis is descriptive and annual cross-sectional: it asks how much
between-state variation in annual per-cow milk yield is explained by the most
informative yearly-pruned clean macro-exposome components after accounting for
region.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

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

STAT = base.STAT
TAB4 = base.TAB4
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

YEARS = list(range(2000, 2026))
TOP_K = 5
MIN_STATES = 14
BREED_PATH = (
    base.ROOT
    / "data"
    / "us_milk"
    / "processed"
    / "genomics"
    / "state_year_adaptive_heat_genetic_index_webconnect_enriched_2003_2020.csv"
)
BREED_BASELINE_COLS = ["cdcb_breed_heat_background_state_z"]
HERD_SCALE_BASELINE_COLS = ["log_milk_cows_head_baseline"]
BASELINES = {
    "Region": {"include_region": True, "covariates": []},
    "Region + breed context": {"include_region": True, "covariates": BREED_BASELINE_COLS},
    "Region + herd scale": {"include_region": True, "covariates": HERD_SCALE_BASELINE_COLS},
    "Region + breed context + herd scale": {
        "include_region": True,
        "covariates": BREED_BASELINE_COLS + HERD_SCALE_BASELINE_COLS,
    },
    "No region": {"include_region": False, "covariates": []},
    "No region + breed context": {"include_region": False, "covariates": BREED_BASELINE_COLS},
    "No region + herd scale": {"include_region": False, "covariates": HERD_SCALE_BASELINE_COLS},
    "No region + breed context + herd scale": {
        "include_region": False,
        "covariates": BREED_BASELINE_COLS + HERD_SCALE_BASELINE_COLS,
    },
    "No region + herd scale + available breed context": {
        "include_region": False,
        "covariates": BREED_BASELINE_COLS + HERD_SCALE_BASELINE_COLS,
        "resolve_by_year": True,
    },
}


def zscore(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    sd = np.nanstd(arr, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return np.zeros_like(arr)
    return (arr - np.nanmean(arr)) / sd


def weighted_r2(
    d: pd.DataFrame,
    exposures: tuple[str, ...],
    include_region: bool = True,
    baseline_covariates: tuple[str, ...] = tuple(),
) -> float:
    cols = ["milk_per_cow_lb", "milk_cows_head", "region"] + list(baseline_covariates) + list(exposures)
    use = d[cols].replace([np.inf, -np.inf], np.nan).copy()
    use = use.dropna(subset=["milk_per_cow_lb", "milk_cows_head", "region"] + list(baseline_covariates))
    if len(use) < MIN_STATES or use["milk_per_cow_lb"].nunique() <= 1:
        return np.nan

    y = zscore(np.log(use["milk_per_cow_lb"].to_numpy(float)))
    pieces = [pd.Series(1.0, index=use.index, name="intercept")]
    if include_region:
        pieces.append(pd.get_dummies(use["region"], prefix="region", drop_first=True, dtype=float))
    for covariate in baseline_covariates:
        x = pd.to_numeric(use[covariate], errors="coerce")
        med = x.median(skipna=True)
        x = x.fillna(med if np.isfinite(med) else 0).to_numpy(float)
        zx = zscore(x)
        if np.nanstd(zx) > 1e-10:
            pieces.append(pd.Series(zx, index=use.index, name=covariate))
    for exposure in exposures:
        x = pd.to_numeric(use[exposure], errors="coerce")
        med = x.median(skipna=True)
        x = x.fillna(med if np.isfinite(med) else 0).to_numpy(float)
        zx = zscore(x)
        if np.nanstd(zx) > 1e-10:
            pieces.append(pd.Series(zx, index=use.index, name=exposure))
    X = pd.concat(pieces, axis=1).to_numpy(float)
    w = use["milk_cows_head"].to_numpy(float)
    sw = np.sqrt(w / np.nanmean(w))
    Xw = X * sw[:, None]
    yw = y * sw
    beta = np.linalg.pinv(Xw.T @ Xw) @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    sse = float(resid @ resid)
    sst = float(((yw - np.average(yw)) ** 2).sum())
    return max(0.0, 1 - sse / sst) if sst > 0 else np.nan


def weighted_nested_incremental_r2_test(
    d: pd.DataFrame,
    exposures: tuple[str, ...],
    include_region: bool = True,
    baseline_covariates: tuple[str, ...] = tuple(),
) -> dict:
    cols = ["milk_per_cow_lb", "milk_cows_head", "region"] + list(baseline_covariates) + list(exposures)
    use = d[cols].replace([np.inf, -np.inf], np.nan).copy()
    use = use.dropna(subset=["milk_per_cow_lb", "milk_cows_head", "region"] + list(baseline_covariates))
    if len(use) < MIN_STATES or use["milk_per_cow_lb"].nunique() <= 1:
        return {
            "base_r2": np.nan,
            "full_r2": np.nan,
            "incremental_r2": np.nan,
            "p": np.nan,
            "f": np.nan,
            "df_num": np.nan,
            "df_den": np.nan,
        }

    y = zscore(np.log(use["milk_per_cow_lb"].to_numpy(float)))
    pieces = [pd.Series(1.0, index=use.index, name="intercept")]
    if include_region:
        pieces.append(pd.get_dummies(use["region"], prefix="region", drop_first=True, dtype=float))
    for covariate in baseline_covariates:
        x = pd.to_numeric(use[covariate], errors="coerce")
        med = x.median(skipna=True)
        x = x.fillna(med if np.isfinite(med) else 0).to_numpy(float)
        zx = zscore(x)
        if np.nanstd(zx) > 1e-10:
            pieces.append(pd.Series(zx, index=use.index, name=covariate))
    X0 = pd.concat(pieces, axis=1)

    exposure_pieces = []
    for exposure in exposures:
        x = pd.to_numeric(use[exposure], errors="coerce")
        med = x.median(skipna=True)
        x = x.fillna(med if np.isfinite(med) else 0).to_numpy(float)
        zx = zscore(x)
        if np.nanstd(zx) > 1e-10:
            exposure_pieces.append(pd.Series(zx, index=use.index, name=exposure))
    if exposure_pieces:
        X1 = pd.concat([X0] + exposure_pieces, axis=1)
    else:
        X1 = X0.copy()

    w = use["milk_cows_head"].to_numpy(float)
    sw = np.sqrt(w / np.nanmean(w))
    yw = y * sw
    sst = float(((yw - np.average(yw)) ** 2).sum())
    if sst <= 0:
        return {
            "base_r2": np.nan,
            "full_r2": np.nan,
            "incremental_r2": np.nan,
            "p": np.nan,
            "f": np.nan,
            "df_num": np.nan,
            "df_den": np.nan,
        }

    X0w = X0.to_numpy(float) * sw[:, None]
    X1w = X1.to_numpy(float) * sw[:, None]
    beta0 = np.linalg.pinv(X0w.T @ X0w) @ (X0w.T @ yw)
    beta1 = np.linalg.pinv(X1w.T @ X1w) @ (X1w.T @ yw)
    resid0 = yw - X0w @ beta0
    resid1 = yw - X1w @ beta1
    sse0 = float(resid0 @ resid0)
    sse1 = float(resid1 @ resid1)
    rank0 = int(np.linalg.matrix_rank(X0w))
    rank1 = int(np.linalg.matrix_rank(X1w))
    df_num = rank1 - rank0
    df_den = len(use) - rank1
    base_r2 = max(0.0, 1 - sse0 / sst)
    full_r2 = max(0.0, 1 - sse1 / sst)
    incr = max(0.0, full_r2 - base_r2)
    f_stat = np.nan
    p_value = np.nan
    if df_num > 0 and df_den > 0 and sse1 > 0 and sse0 >= sse1:
        f_stat = ((sse0 - sse1) / df_num) / (sse1 / df_den)
        p_value = float(stats.f.sf(f_stat, df_num, df_den))
    return {
        "base_r2": base_r2,
        "full_r2": full_r2,
        "incremental_r2": incr,
        "p": p_value,
        "f": f_stat,
        "df_num": df_num,
        "df_den": df_den,
    }


def load_variable_pool() -> pd.DataFrame:
    path = TAB4 / "point4_anchor_year_variable_network_yearly_pruned_union_dictionary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 70_build_anchor_year_variable_cooccurrence_network_yearly_pruned_union.py first."
        )
    vars_df = pd.read_csv(path, low_memory=False)
    return vars_df.drop_duplicates("exposure").reset_index(drop=True)


def load_yearly_scores(vars_df: pd.DataFrame) -> pd.DataFrame:
    path = TAB4 / "point4_nonredundant_variable_yearly_single_r2_skyline_herd_breed_adjusted_clean_macro_exwas_full.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 73_build_and_make_nonredundant_variable_yearly_r2_skyline.py first."
        )
    scores = pd.read_csv(path, low_memory=False)
    scores = scores[scores["exposure"].isin(vars_df["exposure"])].copy()
    scores = scores.merge(
        vars_df[["exposure", "domain_label", "subdomain_label", "exposure_zh"]].drop_duplicates("exposure"),
        on="exposure",
        how="left",
        suffixes=("", "_dict"),
    )
    for col in ["domain_label", "subdomain_label", "exposure_zh"]:
        alt = f"{col}_dict"
        if alt in scores.columns:
            scores[col] = scores[col].fillna(scores[alt])
    return scores


def choose_top_k(scores: pd.DataFrame, year: int, top_k: int = TOP_K) -> pd.DataFrame:
    d = scores[scores["year"].eq(year)].copy()
    d["single_variable_delta_r2"] = pd.to_numeric(d["single_variable_delta_r2"], errors="coerce").fillna(0)
    d["p"] = pd.to_numeric(d["p"], errors="coerce").fillna(1)
    d = d.sort_values(
        ["subdomain_label", "single_variable_delta_r2", "p", "exposure"],
        ascending=[True, False, True, True],
    )
    sub_reps = d.groupby("subdomain_label", as_index=False).head(1)
    out = (
        sub_reps.sort_values(["single_variable_delta_r2", "p", "exposure"], ascending=[False, True, True])
        .head(top_k)
        .reset_index(drop=True)
    )
    out["selection_rank"] = np.arange(1, len(out) + 1)
    return out


def load_annual_panel(vars_df: pd.DataFrame) -> pd.DataFrame:
    old_years = list(base.ANCHOR_YEARS)
    base.ANCHOR_YEARS = YEARS
    try:
        annual_y = base.load_annual_percow()
        annual_x = base.load_annual_exposure_matrix(vars_df)
    finally:
        base.ANCHOR_YEARS = old_years
    panel = annual_y.merge(annual_x, on=["state_alpha", "year"], how="inner")
    panel["region"] = panel["state_alpha"].map(L.CENSUS_REGION)
    panel["log_milk_cows_head_baseline"] = np.log(pd.to_numeric(panel["milk_cows_head"], errors="coerce"))
    if BREED_PATH.exists():
        breed = pd.read_csv(BREED_PATH, low_memory=False)
        keep = ["state_alpha", "year"] + [c for c in BREED_BASELINE_COLS if c in breed.columns]
        panel = panel.merge(breed[keep].drop_duplicates(["state_alpha", "year"]), on=["state_alpha", "year"], how="left")
    return panel


def resolve_year_covariates(d: pd.DataFrame, covariates: tuple[str, ...]) -> tuple[str, ...]:
    resolved = []
    for covariate in covariates:
        if covariate not in d.columns:
            continue
        values = pd.to_numeric(d[covariate], errors="coerce")
        if values.notna().sum() >= MIN_STATES and values.nunique(dropna=True) > 1:
            resolved.append(covariate)
    return tuple(resolved)


def selected_variable_contribution_for_year(
    panel: pd.DataFrame,
    selected: pd.DataFrame,
    year: int,
    baseline_label: str,
    include_region: bool,
    baseline_covariates: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Fit the annual top-five model and report selected-variable statistics.

    The aggregate contribution is evaluated with a nested F-test comparing the
    baseline annual model with the model containing all selected exposures. For
    variable-level reporting, the script carries forward each selected exposure's
    annual single-variable incremental R2 from the skyline analysis; no
    multivariable contribution decomposition is performed.
    """
    d = panel[panel["year"].eq(year)].copy()
    exposures = tuple([x for x in selected["exposure"].tolist() if x in d.columns])
    nested = weighted_nested_incremental_r2_test(
        d,
        exposures,
        include_region=include_region,
        baseline_covariates=baseline_covariates,
    )
    rows = []
    for exposure in exposures:
        meta = selected[selected["exposure"].eq(exposure)].iloc[0].to_dict()
        rows.append(
            {
                "year": year,
                "exposure": exposure,
                "exposure_zh": meta.get("exposure_zh", ""),
                "domain_label": meta.get("domain_label", ""),
                "subdomain_label": meta.get("subdomain_label", ""),
                "selection_rank": meta.get("selection_rank", np.nan),
                "single_variable_delta_r2": meta.get("single_variable_delta_r2", np.nan),
                "single_variable_beta": meta.get("beta", np.nan),
                "single_variable_p": meta.get("p", np.nan),
                "selected_variable_incremental_r2": meta.get("single_variable_delta_r2", np.nan),
                "base_model_r2": nested["base_r2"],
                "full_model_r2": nested["full_r2"],
                "combined_incremental_r2": nested["incremental_r2"],
                "combined_incremental_r2_p": nested["p"],
                "combined_incremental_r2_f": nested["f"],
                "combined_incremental_r2_df_num": nested["df_num"],
                "combined_incremental_r2_df_den": nested["df_den"],
                "n_states": d["state_alpha"].nunique(),
                "n_states_model": int(
                    d.dropna(subset=["milk_per_cow_lb", "milk_cows_head", "region"] + list(baseline_covariates))[
                        "state_alpha"
                    ].nunique()
                ),
                "n_selected": len(exposures),
                "baseline": baseline_label,
                "include_region_baseline": include_region,
                "baseline_covariates": ";".join(baseline_covariates),
                "top_k": TOP_K,
            }
        )
    summary = {
        "year": year,
        "base_model_r2": nested["base_r2"],
        "full_model_r2": nested["full_r2"],
        "combined_incremental_r2": nested["incremental_r2"],
        "combined_incremental_r2_p": nested["p"],
        "combined_incremental_r2_f": nested["f"],
        "combined_incremental_r2_df_num": nested["df_num"],
        "combined_incremental_r2_df_den": nested["df_den"],
        "n_states": d["state_alpha"].nunique(),
        "n_states_model": int(
            d.dropna(subset=["milk_per_cow_lb", "milk_cows_head", "region"] + list(baseline_covariates))[
                "state_alpha"
            ].nunique()
        ),
        "n_selected": len(exposures),
        "baseline": baseline_label,
        "include_region_baseline": include_region,
        "baseline_covariates": ";".join(baseline_covariates),
        "top_k": TOP_K,
    }
    return pd.DataFrame(rows), summary


def main() -> int:
    vars_df = load_variable_pool()
    scores = load_yearly_scores(vars_df)
    panel = load_annual_panel(vars_df)

    selected_rows = []
    contrib_rows = []
    summary_rows = []
    for year in YEARS:
        selected = choose_top_k(scores, year)
        selected_rows.append(selected.assign(year=year))
        for baseline_label, baseline_spec in BASELINES.items():
            covariates = tuple(baseline_spec["covariates"])
            if baseline_spec.get("resolve_by_year", False):
                covariates = resolve_year_covariates(panel[panel["year"].eq(year)].copy(), covariates)
            include_region = bool(baseline_spec["include_region"])
            contrib, summary = selected_variable_contribution_for_year(panel, selected, year, baseline_label, include_region, covariates)
            contrib_rows.append(contrib)
            summary_rows.append(summary)

    selected_out = pd.concat(selected_rows, ignore_index=True)
    contrib_out = pd.concat(contrib_rows, ignore_index=True)
    summary_out = pd.DataFrame(summary_rows)
    selected_out.to_csv(
        TAB4 / "point4_annual_region_adjusted_sparse_exposome_selected_variables.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrib_out.to_csv(
        TAB4 / "point4_annual_region_adjusted_sparse_exposome_selected_variable_contribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_out.to_csv(
        TAB4 / "point4_annual_region_adjusted_sparse_exposome_incremental_r2.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Wrote annual region-adjusted sparse exposome incremental R2.")
    print(summary_out.assign(incr_pct=100 * summary_out["combined_incremental_r2"]).round(3).to_string(index=False))
    print(
        contrib_out.groupby("domain_label")["selected_variable_incremental_r2"]
        .sum()
        .mul(100)
        .sort_values(ascending=False)
        .round(2)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
