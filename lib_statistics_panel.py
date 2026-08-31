"""Shared statistics-module panel loader and fixed-effect estimator.

This local helper keeps the statistics story modules self-contained. It reads
only the organized project data directories and does not import legacy analysis
helpers.
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
US_MILK = ROOT / "data" / "us_milk" / "processed"
US_EXPOSE = ROOT / "data" / "us_expose_new" / "processed"
if not US_EXPOSE.exists():
    US_EXPOSE = ROOT / "data" / "us_expose" / "processed"

CENSUS_REGION = {
    "OH": "Midwest", "IN": "Midwest", "IL": "Midwest", "MI": "Midwest", "WI": "Midwest",
    "MN": "Midwest", "IA": "Midwest", "MO": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "NE": "Midwest", "KS": "Midwest",
    "PA": "Northeast", "NY": "Northeast", "NJ": "Northeast", "CT": "Northeast",
    "RI": "Northeast", "MA": "Northeast", "VT": "Northeast", "NH": "Northeast", "ME": "Northeast",
    "DE": "South", "MD": "South", "VA": "South", "WV": "South", "NC": "South", "SC": "South",
    "GA": "South", "FL": "South", "KY": "South", "TN": "South", "AL": "South", "MS": "South",
    "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    "MT": "West", "ID": "West", "WY": "West", "CO": "West", "NM": "West", "AZ": "West",
    "UT": "West", "NV": "West", "WA": "West", "OR": "West", "CA": "West", "AK": "West", "HI": "West",
}

US_MILK_TABLES = ROOT / "data" / "us_milk" / "tables"
FIFTY_STATES = set(CENSUS_REGION)


def build_derived_milk_panel_50() -> pd.DataFrame:
    """Return 50-state monthly-equivalent milk panel.

    The outcome is total milk production divided by milk cows. Quarter-total
    records are divided by 3 before per-cow calculation; sparse quarterly
    state-years are expanded to quarter months. Official per-cow values are
    retained only for QC/provenance.
    """
    cols = [
        "state_alpha", "year", "month", "milk_per_cow_lb",
        "milk_production_lb", "milk_cows_head",
    ]
    raw = pd.read_csv(US_MILK_TABLES / "analysis_1_2_4_state_month_milk_phenotypes.csv", usecols=cols, low_memory=False)
    raw = raw[raw["state_alpha"].isin(FIFTY_STATES)].copy()
    raw = raw[(raw["year"] >= 2000) & (raw["year"] <= 2025)].copy()
    raw["milk_per_cow_lb_official_qc"] = raw["milk_per_cow_lb"]
    raw["milk_production_lb_raw"] = raw["milk_production_lb"]
    raw["derived_total_per_cow_lb_raw"] = raw["milk_production_lb_raw"] / raw["milk_cows_head"]
    ratio = raw["derived_total_per_cow_lb_raw"] / raw["milk_per_cow_lb_official_qc"]
    raw["quarterly_total_record"] = (
        raw["month"].isin([1, 4, 7, 10])
        & raw["derived_total_per_cow_lb_raw"].notna()
        & (raw["milk_per_cow_lb_official_qc"].isna() | ratio.gt(1.5))
    )
    raw["milk_production_lb_monthly_equiv"] = raw["milk_production_lb_raw"].where(
        ~raw["quarterly_total_record"], raw["milk_production_lb_raw"] / 3.0
    )
    raw["milk_per_cow_derived_50_lb"] = raw["milk_production_lb_monthly_equiv"] / raw["milk_cows_head"]

    rows = []
    for (_, _), g in raw.sort_values(["state_alpha", "year", "month"]).groupby(["state_alpha", "year"], sort=False):
        available = g.loc[g["milk_per_cow_derived_50_lb"].notna(), "month"].astype(int).tolist()
        is_quarterly_year = bool(available) and len(available) <= 4 and set(available).issubset({1, 4, 7, 10})
        if is_quarterly_year:
            for _, row in g[g["milk_per_cow_derived_50_lb"].notna()].iterrows():
                start = int(row["month"])
                for month in range(start, min(start + 3, 13)):
                    r = row.copy()
                    r["month"] = month
                    r["production_source_resolution"] = "quarterly_allocated_average"
                    r["cow_source_resolution"] = "quarterly_average"
                    rows.append(r)
        else:
            h = g.copy()
            h["production_source_resolution"] = np.where(h["quarterly_total_record"], "quarterly_allocated_average", "monthly")
            h["cow_source_resolution"] = np.where(h["quarterly_total_record"], "quarterly_average", "monthly")
            rows.extend([r for _, r in h.iterrows()])
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["milk_per_cow_derived_50_lb", "milk_cows_head"]).copy()
    out = out[(out["milk_production_lb_monthly_equiv"] > 0) & (out["milk_cows_head"] > 0)].copy()
    out = out.sort_values(["state_alpha", "year", "month", "production_source_resolution"]).drop_duplicates(["state_alpha", "year", "month"], keep="first")
    out["milk_per_cow_lb"] = out["milk_per_cow_derived_50_lb"]
    out["milk_per_cow_kg"] = out["milk_per_cow_lb"] * 0.45359237
    out["milk_production_lb"] = out["milk_production_lb_monthly_equiv"]
    out["milk_production_million_kg"] = out["milk_production_lb"] * 0.45359237 / 1e6
    out["milk_production_from_cows_million_kg"] = out["milk_per_cow_lb"] * out["milk_cows_head"] * 0.45359237 / 1e6
    out["implied_milk_per_cow_lb"] = out["milk_per_cow_lb"]
    out["region"] = out["state_alpha"].map(CENSUS_REGION)
    return out


def load_panel() -> pd.DataFrame:
    key = ["state_alpha", "year", "month"]
    panel = build_derived_milk_panel_50()
    exposure_path = US_EXPOSE / "exposure_state_month_expanded.csv"
    if not exposure_path.exists():
        exposure_path = US_EXPOSE / "exposure_state_month_highres.csv"
    exposure = pd.read_csv(exposure_path, low_memory=False)
    keep = key + [c for c in exposure.columns if c not in key]
    exposure = exposure[keep]
    overlap = [c for c in exposure.columns if c in panel.columns and c not in key]
    if overlap:
        panel = panel.drop(columns=overlap)
    panel = panel.merge(exposure, on=key, how="left")
    for col in list(panel.columns):
        prefix = "daymet_dairy_weighted_"
        if col.startswith(prefix):
            alias = col[len(prefix):]
            if alias not in panel.columns:
                panel[alias] = panel[col]
    panel["region"] = panel["state_alpha"].map(CENSUS_REGION)
    return panel


def transform_y(values: np.ndarray, log: bool | str = "auto") -> tuple[np.ndarray, str]:
    values = values.astype(float)
    use_log = (log is True) or (log == "auto" and np.nanmin(values) > 0)
    return (np.log(values), "log") if use_log else (values, "none")


def _dummies(series: pd.Series, prefix: str, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(series.astype(str), prefix=prefix, drop_first=drop_first, dtype=float)


def fixed_effects(df: pd.DataFrame, spec: str = "twoway") -> pd.DataFrame:
    pieces = [pd.Series(1.0, index=df.index, name="intercept")]
    pieces.append(_dummies(df["state_alpha"], "state"))
    if spec == "twoway":
        ym = df["year"].astype(int).astype(str) + "_" + df["month"].astype(int).astype(str).str.zfill(2)
        pieces.append(_dummies(ym, "ym"))
    elif spec == "sm_yeartrend":
        pieces.append(_dummies(df["month"], "month"))
        year = df["year"].astype(float)
        pieces.append(((year - year.mean()) / year.std(ddof=0)).rename("year_scaled"))
    elif spec == "state_year":
        pieces.append(_dummies(df["year"], "year"))
    else:
        raise ValueError(f"unknown spec {spec}")
    return pd.concat(pieces, axis=1)


def absorb(m: np.ndarray, fe: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(fe, m, rcond=None)
    return m - fe @ coef


def cluster_robust_ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray, absorbed_params: int = 0) -> dict:
    y = y.astype(float)
    x = x.astype(float)
    n, k = x.shape
    k_full = k + absorbed_params
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    meat = np.zeros((k, k))
    uniq = np.unique(clusters)
    g = len(uniq)
    for cl in uniq:
        mask = clusters == cl
        score = x[mask].T @ resid[mask]
        meat += np.outer(score, score)
    correction = (g / (g - 1)) * ((n - 1) / (n - k_full)) if g > 1 and n > k_full else 1.0
    cov_cluster = correction * (xtx_inv @ meat @ xtx_inv)

    sigma2 = float(resid @ resid) / max(n - k_full, 1)
    cov_classical = sigma2 * xtx_inv
    return {
        "beta": beta,
        "se_cluster": np.sqrt(np.clip(np.diag(cov_cluster), 0, None)),
        "se_classical": np.sqrt(np.clip(np.diag(cov_classical), 0, None)),
        "n": n,
        "k": k_full,
        "n_clusters": g,
        "resid": resid,
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / sd if sd > 0 else x * 0.0


def fit_exposure(
    df: pd.DataFrame,
    y_col: str,
    exposures: list[str] | str,
    spec: str = "twoway",
    cluster_col: str = "state_alpha",
    log: bool | str = "auto",
    standardize: bool = True,
    extra_covariates: list[str] | None = None,
    weight_col: str | None = None,
) -> dict:
    try:
        from scipy import stats
    except ModuleNotFoundError:
        stats = None

    if isinstance(exposures, str):
        exposures = [exposures]
    extra_covariates = extra_covariates or []

    needed = ["state_alpha", "year", "month", y_col] + exposures + extra_covariates
    if weight_col:
        needed.append(weight_col)
    data = df[needed].replace([np.inf, -np.inf], np.nan).dropna()
    if weight_col:
        data = data[data[weight_col] > 0]
    if len(data) < 60 or data[cluster_col].nunique() < 3:
        return {"status": "too_few", "n": len(data), "results": {}}

    y, ytrans = transform_y(data[y_col].to_numpy(), log=log)
    y = _standardize(y) if standardize else y
    cov_cols = exposures + extra_covariates
    z = np.column_stack([
        _standardize(data[c].to_numpy().astype(float)) if standardize else data[c].to_numpy().astype(float)
        for c in cov_cols
    ])
    fe = fixed_effects(data, spec=spec).to_numpy()
    clusters = data[cluster_col].to_numpy()

    if weight_col:
        w = data[weight_col].to_numpy().astype(float)
        sw = np.sqrt(w / np.mean(w))
        y = y * sw
        z = z * sw[:, None]
        fe = fe * sw[:, None]

    y_r = absorb(y.reshape(-1, 1), fe).ravel()
    z_r = absorb(z, fe)
    fit = cluster_robust_ols(y_r, z_r, clusters, absorbed_params=fe.shape[1])
    df_resid = max(fit["n_clusters"] - 1, 1)
    ss_total = float(y @ y)
    ss_base = float(y_r @ y_r)
    ss_full = float(fit["resid"] @ fit["resid"])
    incr_r2 = (ss_base - ss_full) / ss_total if ss_total > 0 else np.nan

    results = {}
    for i, name in enumerate(cov_cols):
        b = float(fit["beta"][i])
        se_c = float(fit["se_cluster"][i])
        se_o = float(fit["se_classical"][i])
        t_c = b / se_c if se_c > 0 else np.nan
        if not np.isfinite(t_c):
            p_c = np.nan
        elif stats is not None:
            p_c = 2 * stats.t.sf(abs(t_c), df_resid)
        else:
            p_c = 2 * (1 - 0.5 * (1 + erf(abs(t_c) / sqrt(2))))
        results[name] = {
            "beta": b,
            "se_cluster": se_c,
            "t_cluster": t_c,
            "p_cluster": p_c,
            "se_classical": se_o,
            "t_classical": (b / se_o if se_o > 0 else np.nan),
            "se_inflation": (se_c / se_o if se_o > 0 else np.nan),
        }
    return {
        "status": "ok",
        "n": fit["n"],
        "n_clusters": fit["n_clusters"],
        "y_transform": ytrans,
        "spec": spec,
        "results": results,
        "incr_r2": incr_r2,
        "weighted": weight_col is not None,
        "exposure_of_interest": exposures[0],
    }
