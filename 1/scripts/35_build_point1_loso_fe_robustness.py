#!/usr/bin/env python3
"""Build Point 1 leave-one-state-out and FE-vibration robustness tables."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402

TAB = POINT / "tables"
US_MILK = ROOT / "data" / "us_milk" / "processed"
US_EXPOSE = ROOT / "data" / "us_expose_new" / "processed"
KEY = ["state_alpha", "year", "month"]
ENDPOINTS = {
    "total_26": "milk_production_lb_total26",
    "per_cow_26": "milk_per_cow_lb",
}
EXCLUDED_DOMAINS = {
    "Drought", "Wildfire smoke", "Air pollution", "Industrial chemicals",
    "Production system context",
}
OUT_LOSO = TAB / "point1_loso_beta_robustness.csv"
OUT_FE = TAB / "point1_alt_fe_manhattan_robustness.csv"


def weighted_group_demean(mat: np.ndarray, codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = mat.copy()
    n_groups = int(codes.max()) + 1
    den = np.bincount(codes, weights=weights, minlength=n_groups)
    den = np.where(den == 0, np.nan, den)
    for j in range(out.shape[1]):
        num = np.bincount(codes, weights=weights * out[:, j], minlength=n_groups)
        out[:, j] = out[:, j] - (num / den)[codes]
    return out


def weighted_two_way_within(mat: np.ndarray, state: np.ndarray, time: np.ndarray, weights: np.ndarray) -> np.ndarray:
    state_codes, _ = pd.factorize(state, sort=True)
    time_codes, _ = pd.factorize(time, sort=True)
    out = mat.astype(float).copy()
    for _ in range(30):
        prev = out.copy()
        out = weighted_group_demean(out, state_codes, weights)
        out = weighted_group_demean(out, time_codes, weights)
        if np.nanmax(np.abs(out - prev)) < 1e-8:
            break
    return out


def fast_beta(df: pd.DataFrame, y_col: str, x_col: str, keep_mask: np.ndarray | None = None) -> float:
    needed = ["state_alpha", "year", "month", y_col, x_col, "milk_cows_head"]
    if keep_mask is None:
        data = df.loc[:, needed]
    else:
        data = df.loc[keep_mask, needed]
    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    data = data[data["milk_cows_head"] > 0]
    if len(data) < 60 or data["state_alpha"].nunique() < 3:
        return np.nan
    y, _ = L.transform_y(data[y_col].to_numpy(), log="auto")
    x = data[x_col].to_numpy(dtype=float)
    y = L._standardize(y)
    x = L._standardize(x)
    w = data["milk_cows_head"].to_numpy(dtype=float)
    time = data["year"].astype(int).astype(str) + "_" + data["month"].astype(int).astype(str).str.zfill(2)
    resid = weighted_two_way_within(
        np.column_stack([y, x]),
        data["state_alpha"].astype(str).to_numpy(),
        time.to_numpy(),
        w,
    )
    y_r = resid[:, 0]
    x_r = resid[:, 1]
    den = float(x_r @ x_r)
    return float((x_r @ y_r) / den) if den > 0 else np.nan


def load_panel() -> pd.DataFrame:
    milk = pd.read_csv(US_MILK / "state_month_panel.csv", low_memory=False)
    exp = pd.read_csv(US_EXPOSE / "exposure_state_month_expanded.csv", low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    panel = milk.merge(exp, on=KEY, how="left")
    per_cow_states = sorted(panel.loc[panel["milk_per_cow_lb"].notna(), "state_alpha"].dropna().unique())
    panel["milk_production_lb_total26"] = np.where(
        panel["state_alpha"].isin(per_cow_states), panel["milk_production_lb"], np.nan
    )
    return panel


def load_clean_assoc() -> pd.DataFrame:
    assoc = pd.read_csv(TAB / "point1_native_only_endpoint_exwas_associations.csv", low_memory=False)
    assoc["domain"] = np.where(
        (assoc["domain"].eq("Pandemic shock")) & (assoc["mechanistic_domain_en"].eq("COVID")),
        "COVID",
        np.where(
            (assoc["domain"].eq("Pandemic shock")) & (assoc["mechanistic_domain_en"].eq("HPAI")),
            "HPAI",
            np.where(assoc["domain"].eq("Dairy market"), "Milk price / dairy market", assoc["domain"]),
        ),
    )
    assoc = assoc[
        assoc["window"].eq("native")
        & assoc["phenotype_scope"].isin(["total_26", "per_cow_26"])
        & ~assoc["domain"].isin(EXCLUDED_DOMAINS)
        & np.isfinite(assoc["plot_p"])
    ].copy()
    if "measurement_support_variable" in assoc.columns:
        assoc = assoc[~assoc["measurement_support_variable"].fillna(False).astype(bool)].copy()
    return assoc


def by_fdr(pvals: pd.Series) -> pd.Series:
    p = pvals.to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return pd.Series(out, index=pvals.index)
    c_m = np.sum(1.0 / np.arange(1, m + 1))
    order = np.argsort(p[ok])
    ranked = p[ok][order]
    q = ranked * m * c_m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    idx = np.where(ok)[0][order]
    out[idx] = q
    return pd.Series(out, index=pvals.index)


def tier_from_p(p: float, q: float, n_tests: int) -> str:
    if np.isfinite(p) and p < 0.05 / n_tests:
        return "Bonferroni"
    if np.isfinite(q) and q < 0.05:
        return "BY-FDR"
    if np.isfinite(p) and p < 0.05:
        return "P<0.05"
    return "n.s."


def build_loso(panel: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta_cols = ["phenotype_scope", "domain", "source_class", "exposure", "exposure_zh", "plot_beta"]
    clean = assoc[meta_cols].drop_duplicates(["phenotype_scope", "exposure"])
    for endpoint, y_col in ENDPOINTS.items():
        endpoint_assoc = clean[clean["phenotype_scope"].eq(endpoint)].copy()
        states = sorted(panel.loc[panel[y_col].notna(), "state_alpha"].dropna().unique())
        for i, (_, r) in enumerate(endpoint_assoc.iterrows(), 1):
            betas = []
            for state in states:
                keep = panel["state_alpha"].ne(state).to_numpy()
                b = fast_beta(panel, y_col, r["exposure"], keep_mask=keep)
                if np.isfinite(b):
                    betas.append(b)
            arr = np.asarray(betas, dtype=float)
            n_loso = len(arr)
            rows.append({
                "phenotype_scope": endpoint,
                "domain": r["domain"],
                "source_class": r["source_class"],
                "exposure": r["exposure"],
                "exposure_zh": r.get("exposure_zh", ""),
                "main_beta": r["plot_beta"],
                "loso_n": n_loso,
                "loso_beta_mean": float(np.nanmean(arr)) if n_loso else np.nan,
                "loso_beta_median": float(np.nanmedian(arr)) if n_loso else np.nan,
                "loso_beta_lo": float(np.nanquantile(arr, 0.025)) if n_loso else np.nan,
                "loso_beta_hi": float(np.nanquantile(arr, 0.975)) if n_loso else np.nan,
                "loso_same_sign_share": float(np.mean(np.sign(arr) == np.sign(r["plot_beta"]))) if n_loso and np.isfinite(r["plot_beta"]) else np.nan,
            })
            if i % 25 == 0:
                print(f"  LOSO {endpoint}: {i}/{len(endpoint_assoc)} exposures", flush=True)
    return pd.DataFrame(rows)


def fit_alt_missing(panel: pd.DataFrame, endpoint: str, exposure: str, spec: str) -> tuple[float, float, float]:
    y_col = ENDPOINTS[endpoint]
    if exposure not in panel.columns:
        return np.nan, np.nan, np.nan
    fit = L.fit_exposure(panel, y_col, exposure, spec=spec, weight_col="milk_cows_head")
    if fit.get("status") != "ok":
        return np.nan, np.nan, np.nan
    res = fit["results"][exposure]
    return float(res["beta"]), float(res["p_cluster"]), float(fit.get("incr_r2", np.nan))


def build_alt_fe(assoc: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    spec_map = {
        "state_month_yeartrend": ("state + month FE + linear year trend", "beta_sm_yeartrend", "p_sm_yeartrend"),
        "state_year_month": ("state + year FE + month FE", "beta_state_year", "p_state_year"),
    }
    base_cols = [
        "phenotype_scope", "domain", "source_class", "exposure", "exposure_zh",
        "plot_incr_r2", "plot_beta",
    ]
    for endpoint, endpoint_assoc in assoc.groupby("phenotype_scope"):
        n_tests = endpoint_assoc["exposure"].nunique()
        for spec_id, (spec_label, beta_col, p_col) in spec_map.items():
            tmp = endpoint_assoc[base_cols + [beta_col, p_col]].copy()
            tmp = tmp.rename(columns={beta_col: "beta", p_col: "p"})
            missing = tmp["beta"].isna() | tmp["p"].isna()
            if missing.any():
                spec_for_fit = "sm_yeartrend" if spec_id == "state_month_yeartrend" else "state_year"
                for idx in tmp.index[missing]:
                    b, p, r2 = fit_alt_missing(panel, endpoint, tmp.at[idx, "exposure"], spec_for_fit)
                    tmp.at[idx, "beta"] = b
                    tmp.at[idx, "p"] = p
                    if np.isfinite(r2):
                        tmp.at[idx, "plot_incr_r2"] = r2
            tmp["spec_id"] = spec_id
            tmp["spec_label"] = spec_label
            tmp["q_by"] = by_fdr(tmp["p"])
            tmp["signal_tier"] = [tier_from_p(p, q, n_tests) for p, q in zip(tmp["p"], tmp["q_by"])]
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    panel = load_panel()
    assoc = load_clean_assoc()
    loso = build_loso(panel, assoc)
    fe = build_alt_fe(assoc, panel)
    loso.to_csv(OUT_LOSO, index=False, encoding="utf-8-sig")
    fe.to_csv(OUT_FE, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_LOSO} rows={len(loso)}")
    print(f"Wrote {OUT_FE} rows={len(fe)}")
    print(loso.groupby("phenotype_scope").agg(n=("exposure", "count"), same=("loso_same_sign_share", "mean")).to_string())
    print(fe.groupby(["phenotype_scope", "spec_id"]).agg(n=("exposure", "count"), p05=("p", lambda x: float((x < 0.05).mean()))).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
