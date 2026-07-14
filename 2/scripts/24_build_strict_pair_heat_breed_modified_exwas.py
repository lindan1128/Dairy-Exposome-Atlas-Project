#!/usr/bin/env python3
"""Strict-paired heat sensitivity for Point 2 breed-buffering interactions.

This ports the original breed-modified ExWAS strict-pair heat interaction
analysis into Point 2 so the mini summary figure and supplementary table can be
reproduced without depending on the restored Point 3 directory.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[5]
POINT = ROOT / "analysis" / "statistics" / "2"
TAB = POINT / "tables"
TAB.mkdir(parents=True, exist_ok=True)

STAT = ROOT / "analysis" / "statistics"
sys.path.insert(0, str(STAT))

VAR_AUDIT = ROOT / "analysis" / "statistics" / "2" / "tables" / "point2_heat_clean_full_paired_variable_audit.csv"
EXP = ROOT / "data" / "us_expose_new" / "processed" / "exposure_state_month_expanded.csv"
MILK = ROOT / "data" / "us_milk" / "processed" / "state_month_panel.csv"
K2 = ROOT / "data" / "us_milk" / "processed" / "genomics" / "cdcb_dhi_k2_state_year_breed_mix_2000_2020.csv"

OUT_ASSOC = TAB / "point2_strict_pair_heat_breed_modified_exwas_interactions.csv"
OUT_SUMMARY = TAB / "point2_strict_pair_heat_breed_modified_exwas_summary.csv"
OUT_INTERP = TAB / "point2_strict_pair_heat_breed_modified_exwas_interpretation.md"

BREED_CONTEXTS = {
    "breed_heat_background_z": {
        "column": "cdcb_dhi_breed_heat_background_z",
        "label": "Observed K-2 breed heat background",
        "expected_buffering_sign": 1,
        "role": "primary",
    },
}


def zscore_arr(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if np.isfinite(sd) and sd > 0 else x * np.nan


def zscore_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    sd = x.std(skipna=True)
    return (x - x.mean(skipna=True)) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=s.index)


def bh_q(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce").to_numpy(float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return pd.Series(q, index=pvals.index)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.minimum(adj, 1)
    return pd.Series(q, index=pvals.index)


def fe_design(df: pd.DataFrame) -> pd.DataFrame:
    year = pd.to_numeric(df["year"], errors="coerce").astype(float)
    pieces = [
        pd.Series(1.0, index=df.index, name="intercept"),
        pd.get_dummies(df["state_alpha"], prefix="state", drop_first=True, dtype=float),
        pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=float),
        ((year - year.mean()) / year.std(ddof=0)).rename("year_scaled"),
    ]
    return pd.concat(pieces, axis=1)


def absorb(m: np.ndarray, fe: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(fe, m, rcond=None)
    return m - fe @ coef


def cluster_fit_interaction(df: pd.DataFrame, y: str, x: str, g: str) -> dict[str, float | str | int]:
    d = df[["state_alpha", "year", "month", y, x, g]].replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "status": "ok",
        "n": int(len(d)),
        "n_states": int(d["state_alpha"].nunique()),
        "beta_exposure": np.nan,
        "beta_breed_context": np.nan,
        "beta_interaction": np.nan,
        "se_interaction": np.nan,
        "p_interaction": np.nan,
        "incr_r2_interaction": np.nan,
    }
    if len(d) < 120 or d["state_alpha"].nunique() < 8 or d[x].std() <= 0 or d[g].std() <= 0:
        out["status"] = "insufficient_variation"
        return out

    try:
        yy = zscore_arr(np.log(pd.to_numeric(d[y], errors="coerce").to_numpy(float)))
        xx = zscore_arr(pd.to_numeric(d[x], errors="coerce").to_numpy(float))
        gg = zscore_arr(pd.to_numeric(d[g], errors="coerce").to_numpy(float))
        ii = xx * gg
        z = np.column_stack([xx, gg, ii])
        fe = fe_design(d).to_numpy(float)
        if not (np.isfinite(yy).all() and np.isfinite(z).all() and np.isfinite(fe).all()):
            out["status"] = "nonfinite_design"
            return out
        yy_r = absorb(yy.reshape(-1, 1), fe).ravel()
        z_r = absorb(z, fe)
        beta, *_ = np.linalg.lstsq(z_r, yy_r, rcond=None)
        resid = yy_r - z_r @ beta
        xtx_inv = np.linalg.pinv(z_r.T @ z_r)
        meat = np.zeros((z_r.shape[1], z_r.shape[1]))
        groups = d["state_alpha"].to_numpy()
        uniq = np.unique(groups)
        for cl in uniq:
            mask = groups == cl
            score = z_r[mask, :].T @ resid[mask]
            meat += np.outer(score, score)
        n, k = z_r.shape
        k_full = k + fe.shape[1]
        scale = (len(uniq) / (len(uniq) - 1)) * ((n - 1) / (n - k_full)) if len(uniq) > 1 and n > k_full else 1.0
        vcov = scale * xtx_inv @ meat @ xtx_inv
        se_i = float(np.sqrt(max(vcov[2, 2], 0)))
        t_i = float(beta[2] / se_i) if se_i > 0 else np.nan
        p_i = float(2 * stats.t.sf(abs(t_i), df=max(len(uniq) - 1, 1))) if np.isfinite(t_i) else np.nan

        z_no_i = z_r[:, :2]
        beta_no_i, *_ = np.linalg.lstsq(z_no_i, yy_r, rcond=None)
        resid_no_i = yy_r - z_no_i @ beta_no_i
        ss_base = float(yy_r @ yy_r)
        incr = (float(resid_no_i @ resid_no_i) - float(resid @ resid)) / ss_base if ss_base > 0 else np.nan
        out.update({
            "beta_exposure": float(beta[0]),
            "beta_breed_context": float(beta[1]),
            "beta_interaction": float(beta[2]),
            "se_interaction": se_i,
            "p_interaction": p_i,
            "incr_r2_interaction": float(incr),
        })
    except Exception as exc:
        out["status"] = f"fit_failed:{type(exc).__name__}"
    return out


def build_strict_meta() -> pd.DataFrame:
    audit = pd.read_csv(VAR_AUDIT, low_memory=False)
    d = audit[audit["in_strict_paired_heat"].astype(str).str.lower().eq("true")].copy()
    d = d.rename(
        columns={
            "domain": "domain_en",
            "variables_en": "exposure",
            "variables_ch": "exposure_zh",
            "strict_paired_subform": "strict_pair_form",
        }
    )
    d["domain_zh"] = "热应激"
    d["mechanistic_domain_zh"] = np.where(
        d["strict_pair_form"].eq("Humid paired heat"),
        "湿热/湿球温度",
        "干热/VPD",
    )
    d["exposure_family"] = np.where(
        d["strict_pair_form"].eq("Humid paired heat"),
        "strict_humid_paired_heat",
        "strict_dry_paired_heat",
    )
    cols = [
        "domain_en",
        "domain_zh",
        "mechanistic_domain_en",
        "mechanistic_domain_zh",
        "exposure",
        "exposure_zh",
        "construct",
        "form",
        "strict_pair_form",
        "strict_pair_group",
        "strict_pair_label",
        "exposure_family",
    ]
    return d[cols].drop_duplicates("exposure").sort_values(["strict_pair_form", "strict_pair_group", "exposure"])


def load_panel(exposures: list[str]) -> pd.DataFrame:
    header = pd.read_csv(EXP, nrows=0).columns.tolist()
    exposures = [x for x in exposures if x in header]
    milk = pd.read_csv(MILK, low_memory=False)
    overlap = [c for c in exposures if c in milk.columns]
    if overlap:
        milk = milk.drop(columns=overlap)
    exp = pd.read_csv(EXP, usecols=["state_alpha", "year", "month"] + exposures, low_memory=False)
    panel = milk[["state_alpha", "year", "month", "milk_per_cow_lb"]].merge(
        exp, on=["state_alpha", "year", "month"], how="inner"
    )
    panel = panel[panel["milk_per_cow_lb"].notna()].copy()

    k2 = pd.read_csv(K2, low_memory=False)
    if "calving_year" in k2.columns and "year" not in k2.columns:
        k2 = k2.rename(columns={"calving_year": "year"})
    keep = ["state_alpha", "year"] + [v["column"] for v in BREED_CONTEXTS.values()]
    k2 = k2[keep].copy()
    for name, meta in BREED_CONTEXTS.items():
        k2[name] = zscore_series(k2[meta["column"]])
    panel = panel.merge(k2[["state_alpha", "year"] + list(BREED_CONTEXTS)], on=["state_alpha", "year"], how="inner")
    return panel[panel["year"].between(2003, 2020)].copy()


def main() -> None:
    meta = build_strict_meta()
    exposures = meta["exposure"].tolist()
    panel = load_panel(exposures)
    meta_map = meta.set_index("exposure")

    rows = []
    for exposure in exposures:
        if exposure not in panel.columns:
            continue
        for context, cmeta in BREED_CONTEXTS.items():
            fit = cluster_fit_interaction(panel, "milk_per_cow_lb", exposure, context)
            r = meta_map.loc[exposure]
            beta_i = fit["beta_interaction"]
            expected_buffering = (
                np.isfinite(beta_i)
                and np.sign(beta_i) == cmeta["expected_buffering_sign"]
            )
            rows.append(
                {
                    "breed_context": context,
                    "breed_context_label": cmeta["label"],
                    "breed_context_role": cmeta["role"],
                    "expected_buffering_sign": cmeta["expected_buffering_sign"],
                    "domain": r["domain_en"],
                    "domain_zh": r["domain_zh"],
                    "mechanistic_domain_en": r["mechanistic_domain_en"],
                    "mechanistic_domain_zh": r["mechanistic_domain_zh"],
                    "exposure": exposure,
                    "exposure_zh": r["exposure_zh"],
                    "construct": r["construct"],
                    "form": r["form"],
                    "strict_pair_form": r["strict_pair_form"],
                    "strict_pair_group": r["strict_pair_group"],
                    "strict_pair_label": r["strict_pair_label"],
                    "exposure_family": r["exposure_family"],
                    "main_model": "log(milk per cow)_z ~ strict-pair heat exposure_z * observed breed_context_z + state FE + month FE + year trend; state-cluster SE",
                    **fit,
                    "interaction_direction": "positive" if np.isfinite(beta_i) and beta_i > 0 else ("negative" if np.isfinite(beta_i) and beta_i < 0 else "zero"),
                    "expected_buffering": bool(expected_buffering),
                }
            )

    assoc = pd.DataFrame(rows)
    assoc["q_bh_context"] = assoc.groupby("breed_context", group_keys=False)["p_interaction"].apply(bh_q)
    assoc["p_lt_001"] = assoc["p_interaction"] < 0.01
    assoc["fdr_q05"] = assoc["q_bh_context"] < 0.05
    assoc["signed_neglog10p_interaction"] = np.sign(assoc["beta_interaction"].fillna(0)) * -np.log10(
        assoc["p_interaction"].clip(lower=1e-300)
    )
    assoc.to_csv(OUT_ASSOC, index=False)

    ok = assoc[assoc["status"].eq("ok")].copy()
    summary = (
        ok.groupby(["breed_context", "breed_context_label", "strict_pair_form"], as_index=False)
        .agg(
            n_tests=("exposure", "nunique"),
            n_p_lt_001=("p_lt_001", "sum"),
            n_fdr_q05=("fdr_q05", "sum"),
            n_expected_buffering=("expected_buffering", "sum"),
            median_abs_interaction=("beta_interaction", lambda x: float(np.nanmedian(np.abs(x)))),
            median_interaction=("beta_interaction", "median"),
            median_incr_r2=("incr_r2_interaction", "median"),
        )
        .sort_values(["breed_context", "strict_pair_form"])
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    primary = summary[summary["breed_context"].eq("breed_heat_background_z")].copy()
    OUT_INTERP.write_text(
        "# Strict-paired Point 2 breed-buffering sensitivity\n\n"
        "## Model\n"
        "`log(milk per cow)_z ~ strict-pair heat exposure_z * breed_context_z + state FE + month FE + year trend`, "
        "with state-clustered SE; 2003-2020 observed K-2 breed context.\n\n"
        "## Primary breed heat background summary\n\n"
        + primary.to_string(index=False)
        + "\n\nReading rule: for heat pressure variables, a positive interaction with breed heat background/Jersey share indicates buffering; "
        "the analysis is restricted to the breed heat-background index.\n"
    )

    print("wrote", OUT_ASSOC, assoc.shape)
    print("wrote", OUT_SUMMARY, summary.shape)
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
