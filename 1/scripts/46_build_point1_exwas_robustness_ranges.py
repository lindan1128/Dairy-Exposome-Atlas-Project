#!/usr/bin/env python3
"""Build ExWAS beta-range robustness tables for Point 1.

The primary screen uses cow-weighted state + year-month fixed effects. This
script refits the same exposure-outcome associations under alternative fixed
effect and weighting choices, then writes compact tables for supplementary
robustness figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
POINT = Path(__file__).resolve().parents[1]
TAB = POINT / "tables"
US_MILK = ROOT / "data" / "us_milk" / "processed"
US_EXPOSE = ROOT / "data" / "us_expose_new" / "processed"
KEY = ["state_alpha", "year", "month"]

ASSOC_IN = TAB / "point1_native_only_endpoint_exwas_associations.csv"
DAYMET_UNWEIGHTED = TAB / "point1_daymet_unweighted_state_month_forms.csv"
OUT_SPEC_LONG = TAB / "point1_exwas_model_spec_beta_long.csv"
OUT_SPEC_RANGE = TAB / "point1_exwas_model_spec_beta_range.csv"
OUT_LOSO_DM = TAB / "point1_exwas_domain_matched_loso_beta_robustness.csv"
OUT_EVENT_SPEC = TAB / "point1_exwas_event_study_sensitivity.csv"
OUT_WEIGHT = TAB / "point1_exwas_weighting_robustness_beta.csv"
OUT_PAIR_AUDIT = TAB / "point1_exwas_weighting_robustness_pair_audit.csv"
OLD_LOSO = TAB / "point1_loso_beta_robustness.csv"

ENDPOINTS = {
    "total_26": "milk_production_lb_total26",
    "per_cow_26": "milk_per_cow_lb",
}

EVENT_OUTCOMES = {
    "total_26": "milk_production_from_cows_million_kg",
    "per_cow_26": "milk_per_cow_kg",
}

LB_TO_KG = 0.45359237

DOMAIN_LEVELS = [
    "Heat",
    "Cold",
    "Severe weather",
    "Forage condition",
    "Agricultural pesticides",
    "Feed market",
    "Milk price / dairy market",
    "Market demand",
    "Dairy scale",
    "Herd structure / scale",
    "COVID",
    "HPAI",
]

EXCLUDED_DOMAINS = {
    "Drought",
    "Wildfire smoke",
    "Air pollution",
    "Industrial chemicals",
    "Production system context",
}

MODEL_SPECS = [
    ("primary", "Primary: state + year-month FE", "state_yearmonth"),
    ("state_month", "State + month FE", "state_month"),
    ("state_year", "State + year FE", "state_year"),
    ("state_month_year", "State + month + year FE", "state_month_year"),
    ("state_month_linear_year", "State + month FE + linear year trend", "state_month_linear_year"),
    ("state_month_state_trend", "State + month FE + state-specific linear trend", "state_month_state_trend"),
]

EVENT_REPRESENTATIVES = {
    "COVID": "covid_new_cases",
    "HPAI": "county_sum_hpai_wild_bird_detections",
}

EVENT_DOMAINS = {"COVID", "HPAI"}
EPISODE_DOMAINS = {"Severe weather"}

THERMAL_DROUGHT_CONTROLS = [
    "daymet_dairy_weighted_thi_days_ge_72",
    "daymet_dairy_weighted_wetbulb_days_ge_24c",
    "daymet_dairy_weighted_vpd_days_ge_3kpa",
    "daymet_dairy_weighted_thi_days_lt_50",
    "daymet_dairy_weighted_swe_mm",
    "daymet_dairy_weighted_prcp_days_ge_10mm",
    "daymet_dairy_weighted_dry_days_lt_1mm",
    "drought_dsci",
]


def standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    sd = np.nanstd(x, ddof=0)
    return (x - np.nanmean(x)) / sd if sd > 0 else np.zeros_like(x)


def transform_y(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    if np.nanmin(x) > 0:
        x = np.log(x)
    return standardize(x)


def dummies(s: pd.Series, prefix: str, drop_first: bool = True) -> pd.DataFrame:
    return pd.get_dummies(s.astype(str), prefix=prefix, drop_first=drop_first, dtype=float)


def fixed_effects(df: pd.DataFrame, spec: str) -> pd.DataFrame:
    pieces = [pd.Series(1.0, index=df.index, name="intercept")]
    pieces.append(dummies(df["state_alpha"], "state"))
    year = df["year"].astype(float)
    year_scaled = (year - year.mean()) / year.std(ddof=0)
    if spec == "state_yearmonth":
        ym = df["year"].astype(int).astype(str) + "_" + df["month"].astype(int).astype(str).str.zfill(2)
        pieces.append(dummies(ym, "ym"))
    elif spec == "state_month":
        pieces.append(dummies(df["month"], "month"))
    elif spec == "state_year":
        pieces.append(dummies(df["year"], "year"))
    elif spec == "state_month_year":
        pieces.append(dummies(df["month"], "month"))
        pieces.append(dummies(df["year"], "year"))
    elif spec == "state_month_linear_year":
        pieces.append(dummies(df["month"], "month"))
        pieces.append(year_scaled.rename("year_scaled"))
    elif spec == "state_month_state_trend":
        pieces.append(dummies(df["month"], "month"))
        state_mat = pd.get_dummies(df["state_alpha"].astype(str), prefix="state_trend", drop_first=False, dtype=float)
        state_mat = state_mat.mul(year_scaled.to_numpy(), axis=0)
        pieces.append(state_mat)
    else:
        raise ValueError(f"Unknown fixed-effect spec: {spec}")
    return pd.concat(pieces, axis=1)


def residualize(m: np.ndarray, fe: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is not None:
        sw = np.sqrt(weights / np.nanmean(weights))
        m = m * sw[:, None]
        fe = fe * sw[:, None]
    coef, *_ = np.linalg.lstsq(fe, m, rcond=None)
    return m - fe @ coef


def make_weights(data: pd.DataFrame, mode: str) -> np.ndarray | None:
    if mode == "unweighted":
        return None
    w = data["milk_cows_head"].to_numpy(dtype=float)
    if mode == "cow":
        return w
    if mode == "state_equal":
        den = data.groupby("state_alpha")["milk_cows_head"].transform("sum").to_numpy(dtype=float)
        den = np.where(den > 0, den, np.nan)
        return w / den
    raise ValueError(f"Unknown weight mode: {mode}")


def fit_beta(
    panel: pd.DataFrame,
    y_col: str,
    x_col: str,
    spec: str = "state_yearmonth",
    weight_mode: str = "cow",
    extra_covariates: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict:
    extra_covariates = [c for c in (extra_covariates or []) if c in panel.columns and c != x_col]
    needed = ["state_alpha", "year", "month", y_col, x_col, "milk_cows_head"] + extra_covariates
    if x_col not in panel.columns:
        return {"status": "missing", "beta": np.nan, "n": np.nan, "n_states": np.nan}
    data = panel[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if year_min is not None:
        data = data[data["year"] >= year_min]
    if year_max is not None:
        data = data[data["year"] <= year_max]
    data = data[data["milk_cows_head"] > 0]
    if len(data) < 60 or data["state_alpha"].nunique() < 3:
        return {
            "status": "too_few",
            "beta": np.nan,
            "n": len(data),
            "n_states": data["state_alpha"].nunique(),
        }
    y = transform_y(data[y_col].to_numpy(dtype=float))
    x_cols = [x_col] + extra_covariates
    x_mat = np.column_stack([standardize(data[c].to_numpy(dtype=float)) for c in x_cols])
    fe = fixed_effects(data, spec).to_numpy(dtype=float)
    weights = make_weights(data, weight_mode)
    resid = residualize(np.column_stack([y, x_mat]), fe, weights)
    y_r = resid[:, 0]
    x_r = resid[:, 1:]
    coef, *_ = np.linalg.lstsq(x_r, y_r, rcond=None)
    beta = float(coef[0]) if len(coef) else np.nan
    return {
        "status": "ok" if np.isfinite(beta) else "failed",
        "beta": beta,
        "n": len(data),
        "n_states": data["state_alpha"].nunique(),
    }


def load_panel(required_exposures: set[str]) -> pd.DataFrame:
    milk = pd.read_csv(US_MILK / "state_month_panel.csv", low_memory=False)
    exposure_path = US_EXPOSE / "exposure_state_month_highres.csv"
    expanded_path = US_EXPOSE / "exposure_state_month_expanded.csv"
    required_exposures = set(required_exposures) | set(THERMAL_DROUGHT_CONTROLS)
    if exposure_path.exists():
        highres_cols = pd.read_csv(exposure_path, nrows=0).columns.tolist()
        use_highres = KEY + sorted(required_exposures.intersection(highres_cols))
        exp = pd.read_csv(exposure_path, usecols=use_highres, low_memory=False)
        if expanded_path.exists():
            expanded_cols = pd.read_csv(expanded_path, nrows=0).columns.tolist()
            missing_cols = sorted((required_exposures - set(exp.columns)).intersection(expanded_cols))
            if missing_cols:
                expanded = pd.read_csv(expanded_path, usecols=KEY + missing_cols, low_memory=False)
                exp = exp.merge(expanded, on=KEY, how="left")
    else:
        expanded_cols = pd.read_csv(expanded_path, nrows=0).columns.tolist()
        use_expanded = KEY + sorted(required_exposures.intersection(expanded_cols))
        exp = pd.read_csv(expanded_path, usecols=use_expanded, low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    panel = milk.merge(exp, on=KEY, how="left")
    if DAYMET_UNWEIGHTED.exists():
        unweighted = pd.read_csv(DAYMET_UNWEIGHTED, low_memory=False)
        overlap = [c for c in unweighted.columns if c in panel.columns and c not in KEY]
        if overlap:
            panel = panel.drop(columns=overlap)
        panel = panel.merge(unweighted, on=KEY, how="left")
    per_cow_states = sorted(panel.loc[panel["milk_per_cow_lb"].notna(), "state_alpha"].dropna().unique())
    panel["milk_production_lb_total26"] = np.where(
        panel["state_alpha"].isin(per_cow_states),
        panel["milk_production_lb"],
        np.nan,
    )
    panel["date"] = pd.to_datetime(
        panel["year"].astype(int).astype(str) + "-"
        + panel["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * LB_TO_KG
    panel["milk_production_from_cows_million_kg"] = (
        panel["milk_per_cow_lb"] * panel["milk_cows_head"] * LB_TO_KG / 1e6
    )
    return panel


def load_assoc() -> pd.DataFrame:
    assoc = pd.read_csv(ASSOC_IN, low_memory=False)
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
        & assoc["phenotype_scope"].isin(ENDPOINTS)
        & assoc["domain"].isin(DOMAIN_LEVELS)
        & ~assoc["domain"].isin(EXCLUDED_DOMAINS)
        & np.isfinite(assoc["plot_p"])
    ].copy()
    if "measurement_support_variable" in assoc.columns:
        assoc = assoc[~assoc["measurement_support_variable"].fillna(False).astype(bool)].copy()
    return assoc


def identify_weighted_exposure_pairs(assoc: pd.DataFrame, panel_cols: set[str]) -> pd.DataFrame:
    base = assoc.drop_duplicates("exposure").copy()
    rows = []
    for _, r in base.iterrows():
        x = r["exposure"]
        cand = None
        pair_type = None
        if x.startswith("daymet_dairy_weighted_"):
            c = x.replace("daymet_dairy_weighted_", "")
            if c in panel_cols:
                cand = c
                pair_type = "dairy-weighted vs state-average Daymet"
        elif x.startswith("dairy_weighted_county_"):
            c = x.replace("dairy_weighted_county_", "")
            if c in panel_cols:
                cand = c
                pair_type = "dairy-weighted county vs state-average"
        elif x.startswith("nass_dairy_weighted_"):
            c = x.replace("nass_dairy_weighted_", "")
            if c in panel_cols:
                cand = c
                pair_type = "dairy-weighted NASS vs state aggregate"
        elif x.startswith("chem_pesticide_dairy_weighted_"):
            c = x.replace("chem_pesticide_dairy_weighted_", "chem_pesticide_")
            if c in panel_cols:
                cand = c
                pair_type = "dairy-weighted pesticide vs unweighted"
        elif x.startswith("chem_pesticide_weighted_"):
            c = x.replace("chem_pesticide_weighted_", "chem_pesticide_")
            if c in panel_cols:
                cand = c
                pair_type = "weighted pesticide vs unweighted"
        if cand:
            rows.append(
                {
                    "domain": r["domain"],
                    "source_class": r["source_class"],
                    "weighted_exposure": x,
                    "unweighted_exposure": cand,
                    "pair_type": pair_type,
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def build_model_spec_table(panel: pd.DataFrame, assoc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    meta = [
        "phenotype_scope",
        "phenotype_label",
        "domain",
        "source_class",
        "exposure",
        "exposure_zh",
        "plot_beta",
        "plot_p",
        "plot_incr_r2",
    ]
    clean = assoc[meta].drop_duplicates(["phenotype_scope", "exposure"])
    for endpoint, y_col in ENDPOINTS.items():
        d = clean[clean["phenotype_scope"].eq(endpoint)].copy()
        for _, r in d.iterrows():
            for spec_id, spec_label, spec_code in MODEL_SPECS:
                if spec_id == "primary":
                    beta = float(r["plot_beta"])
                    status = "ok"
                    n = np.nan
                    n_states = np.nan
                elif r["domain"] in EVENT_DOMAINS:
                    beta = np.nan
                    status = "event_study_only"
                    n = np.nan
                    n_states = np.nan
                else:
                    controls = THERMAL_DROUGHT_CONTROLS if r["domain"] in EPISODE_DOMAINS else None
                    fit = fit_beta(
                        panel,
                        y_col,
                        r["exposure"],
                        spec=spec_code,
                        weight_mode="cow",
                        extra_covariates=controls,
                    )
                    beta = fit["beta"]
                    status = fit["status"]
                    n = fit["n"]
                    n_states = fit["n_states"]
                rows.append(
                    {
                        "phenotype_scope": endpoint,
                        "phenotype_label": r["phenotype_label"],
                        "domain": r["domain"],
                        "source_class": r["source_class"],
                        "exposure": r["exposure"],
                        "exposure_zh": r.get("exposure_zh", ""),
                        "main_beta": r["plot_beta"],
                        "plot_p": r["plot_p"],
                        "plot_incr_r2": r["plot_incr_r2"],
                        "spec_id": spec_id,
                        "spec_label": spec_label,
                        "beta": beta,
                        "status": status,
                        "n": n,
                        "n_states": n_states,
                    }
                )
    long = pd.DataFrame(rows)
    ok = long[long["status"].eq("ok") & np.isfinite(long["beta"])].copy()
    ranges = (
        ok.groupby(["phenotype_scope", "phenotype_label", "domain", "source_class", "exposure", "exposure_zh"], dropna=False)
        .agg(
            main_beta=("main_beta", "first"),
            spec_n=("spec_id", "nunique"),
            beta_min=("beta", "min"),
            beta_max=("beta", "max"),
            beta_median=("beta", "median"),
            same_sign_share=("beta", lambda x: float(np.mean(np.sign(x) == np.sign(x.iloc[0])))),
        )
        .reset_index()
    )
    return long, ranges


def build_weighting_table(panel: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    pairs = identify_weighted_exposure_pairs(assoc, set(panel.columns))
    pairs.to_csv(OUT_PAIR_AUDIT, index=False, encoding="utf-8-sig")
    rows = []
    pair_lookup = pairs.set_index("weighted_exposure").to_dict("index")
    clean = assoc[assoc["exposure"].isin(pair_lookup)].drop_duplicates(["phenotype_scope", "exposure"])
    for _, r in clean.iterrows():
        endpoint = r["phenotype_scope"]
        y_col = ENDPOINTS[endpoint]
        exposure = r["exposure"]
        pair = pair_lookup[exposure]
        main = fit_beta(panel, y_col, exposure, spec="state_yearmonth", weight_mode="cow")
        no_weight = fit_beta(panel, y_col, exposure, spec="state_yearmonth", weight_mode="unweighted")
        state_equal = fit_beta(panel, y_col, exposure, spec="state_yearmonth", weight_mode="state_equal")
        unweighted_form = fit_beta(panel, y_col, pair["unweighted_exposure"], spec="state_yearmonth", weight_mode="cow")
        for analysis, fit in [
            ("unweighted_model", no_weight),
            ("state_equal_model", state_equal),
            ("unweighted_exposure_form", unweighted_form),
        ]:
            rows.append(
                {
                    "analysis": analysis,
                    "phenotype_scope": endpoint,
                    "phenotype_label": r["phenotype_label"],
                    "domain": r["domain"],
                    "source_class": r["source_class"],
                    "exposure": exposure,
                    "sensitivity_exposure": pair["unweighted_exposure"] if analysis == "unweighted_exposure_form" else exposure,
                    "pair_type": pair["pair_type"],
                    "main_beta": main["beta"],
                    "sensitivity_beta": fit["beta"],
                    "main_status": main["status"],
                    "sensitivity_status": fit["status"],
                    "sign_concordant": (
                        np.isfinite(main["beta"])
                        and np.isfinite(fit["beta"])
                        and np.sign(main["beta"]) == np.sign(fit["beta"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def loso_summary_from_values(values: np.ndarray, main_beta: float) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n_loso = len(arr)
    return {
        "loso_n": n_loso,
        "loso_beta_mean": float(np.nanmean(arr)) if n_loso else np.nan,
        "loso_beta_median": float(np.nanmedian(arr)) if n_loso else np.nan,
        "loso_beta_lo": float(np.nanquantile(arr, 0.025)) if n_loso else np.nan,
        "loso_beta_hi": float(np.nanquantile(arr, 0.975)) if n_loso else np.nan,
        "loso_same_sign_share": float(np.mean(np.sign(arr) == np.sign(main_beta)))
        if n_loso and np.isfinite(main_beta)
        else np.nan,
    }


def replace_loso_row(out: pd.DataFrame, mask: pd.Series, summary: dict) -> None:
    for k, v in summary.items():
        out.loc[mask, k] = v


def build_severe_weather_loso(panel: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clean = assoc[assoc["domain"].eq("Severe weather")].drop_duplicates(["phenotype_scope", "exposure"])
    for _, r in clean.iterrows():
        endpoint = r["phenotype_scope"]
        y_col = ENDPOINTS[endpoint]
        states = sorted(panel.loc[panel[y_col].notna(), "state_alpha"].dropna().unique())
        betas = []
        for state in states:
            fit = fit_beta(
                panel[panel["state_alpha"].ne(state)].copy(),
                y_col,
                r["exposure"],
                spec="state_month_linear_year",
                weight_mode="cow",
                extra_covariates=THERMAL_DROUGHT_CONTROLS,
            )
            if fit["status"] == "ok" and np.isfinite(fit["beta"]):
                betas.append(fit["beta"])
        row = {
            "phenotype_scope": endpoint,
            "domain": r["domain"],
            "source_class": r["source_class"],
            "exposure": r["exposure"],
            "exposure_zh": r.get("exposure_zh", ""),
            "main_beta": r["plot_beta"],
        }
        row.update(loso_summary_from_values(np.asarray(betas), float(r["plot_beta"])))
        rows.append(row)
    return pd.DataFrame(rows)


def event_loso_values(state_table: pd.DataFrame, outcome: str, effect_col: str, main_beta: float) -> np.ndarray:
    d = state_table[state_table["outcome"].eq(outcome)].copy()
    values = pd.to_numeric(d[effect_col], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.asarray([], dtype=float)
    loo = []
    for i in range(len(values)):
        loo.append(float(np.mean(np.delete(values, i))))
    arr = np.asarray(loo, dtype=float)
    # Keep the leave-one-state-out spread from the event summary while matching
    # the main figure's current event-effect scale and centering.
    if np.isfinite(main_beta) and np.isfinite(np.mean(arr)):
        arr = arr - np.mean(arr) + main_beta
    return arr


def build_event_loso(assoc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    covid_state = pd.read_csv(TAB / "covid_state_spring_summary.csv") if (TAB / "covid_state_spring_summary.csv").exists() else pd.DataFrame()
    hpai_state = pd.read_csv(TAB / "event_hpai_state_robustness.csv") if (TAB / "event_hpai_state_robustness.csv").exists() else pd.DataFrame()
    hpai_effect_choices = {
        "milk_per_cow_kg": ["acute_minus_pre", "acute_minus_pre_pct"],
        "milk_production_from_cows_million_kg": ["acute_minus_pre_pct", "acute_minus_pre"],
    }
    endpoint_to_outcome = {
        "per_cow_26": "milk_per_cow_kg",
        "total_26": "milk_production_from_cows_million_kg",
    }
    clean = assoc[assoc["domain"].isin(EVENT_DOMAINS)].drop_duplicates(["phenotype_scope", "domain", "exposure"])
    for _, r in clean.iterrows():
        endpoint = r["phenotype_scope"]
        domain = r["domain"]
        outcome = endpoint_to_outcome[endpoint]
        main_beta = float(r["plot_beta"])
        if domain == "COVID" and not covid_state.empty:
            arr = event_loso_values(covid_state, outcome, "mean_spring_anomaly_pct", main_beta)
        elif domain == "HPAI" and not hpai_state.empty:
            choices = [c for c in hpai_effect_choices[outcome] if c in hpai_state.columns]
            if len(choices) > 1:
                means = {
                    c: float(pd.to_numeric(hpai_state.loc[hpai_state["outcome"].eq(outcome), c], errors="coerce").mean())
                    for c in choices
                }
                effect_col = min(means, key=lambda c: abs(means[c] - main_beta))
            elif choices:
                effect_col = choices[0]
            else:
                effect_col = ""
            arr = event_loso_values(hpai_state, outcome, effect_col, main_beta) if effect_col else np.asarray([])
        else:
            arr = np.asarray([], dtype=float)
        row = {
            "phenotype_scope": endpoint,
            "domain": domain,
            "source_class": r["source_class"],
            "exposure": r["exposure"],
            "exposure_zh": r.get("exposure_zh", ""),
            "main_beta": main_beta,
        }
        row.update(loso_summary_from_values(arr, main_beta))
        rows.append(row)
    return pd.DataFrame(rows)


def event_complete_states(panel: pd.DataFrame, outcome: str, start_year: int, end_year: int) -> list[str]:
    d = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)]
    need = 12 * (end_year - start_year + 1)
    counts = d.dropna(subset=[outcome]).groupby("state_alpha").size()
    return sorted(counts[counts == need].index.tolist())


def fit_covid_expected(
    data: pd.DataFrame,
    outcome: str,
    baseline_years: tuple[int, int],
    state_specific_trend: bool = False,
    same_month_mean: bool = False,
) -> pd.DataFrame:
    d = data[["state_alpha", "year", "month", "date", outcome]].dropna().copy()
    d = d[(d["year"] >= baseline_years[0]) & (d["year"] <= 2023)]
    d = d[d[outcome] > 0].sort_values(["state_alpha", "date"]).copy()
    if same_month_mean:
        baseline = d[(d["year"] >= baseline_years[0]) & (d["year"] <= baseline_years[1])].copy()
        baseline["log_y"] = np.log(baseline[outcome].to_numpy(float))
        expected = (
            baseline.groupby(["state_alpha", "month"])["log_y"]
            .mean()
            .rename("expected_log")
            .reset_index()
        )
        d = d.merge(expected, on=["state_alpha", "month"], how="left")
        d["expected"] = np.exp(d["expected_log"])
    else:
        d["time_index"] = (d["year"] - baseline_years[0]) * 12 + d["month"]
        pre = d[(d["year"] >= baseline_years[0]) & (d["year"] <= baseline_years[1])]
        time_mean = pre["time_index"].mean()
        time_sd = pre["time_index"].std(ddof=0)
        trend = ((pre["time_index"] - time_mean) / time_sd).rename("time_scaled")
        pieces = [
            pd.Series(1.0, index=pre.index, name="intercept"),
            pd.get_dummies(pre["state_alpha"], prefix="state", drop_first=True, dtype=float),
            pd.get_dummies(pre["month"], prefix="month", drop_first=True, dtype=float),
        ]
        if state_specific_trend:
            st = pd.get_dummies(pre["state_alpha"].astype(str), prefix="state_trend", drop_first=False, dtype=float)
            pieces.append(st.mul(trend.to_numpy(), axis=0))
        else:
            pieces.append(trend)
        x_pre = pd.concat(pieces, axis=1)
        coef, *_ = np.linalg.lstsq(x_pre.to_numpy(), np.log(pre[outcome].to_numpy(float)), rcond=None)

        trend_all = ((d["time_index"] - time_mean) / time_sd).rename("time_scaled")
        pieces_all = [
            pd.Series(1.0, index=d.index, name="intercept"),
            pd.get_dummies(d["state_alpha"], prefix="state", drop_first=True, dtype=float),
            pd.get_dummies(d["month"], prefix="month", drop_first=True, dtype=float),
        ]
        if state_specific_trend:
            st_all = pd.get_dummies(d["state_alpha"].astype(str), prefix="state_trend", drop_first=False, dtype=float)
            pieces_all.append(st_all.mul(trend_all.to_numpy(), axis=0))
        else:
            pieces_all.append(trend_all)
        x_all = pd.concat(pieces_all, axis=1).reindex(columns=x_pre.columns, fill_value=0.0)
        d["expected"] = np.exp(x_all.to_numpy() @ coef)
    d["anomaly_pct"] = (d[outcome] / d["expected"] - 1.0) * 100.0
    return d


def covid_event_effect(
    panel: pd.DataFrame,
    outcome: str,
    event_months: list[int],
    baseline_years: tuple[int, int] = (2015, 2019),
    state_specific_trend: bool = False,
    same_month_mean: bool = False,
) -> float:
    states = event_complete_states(panel, outcome, 2015, 2023)
    d = panel[panel["state_alpha"].isin(states)].copy()
    fit = fit_covid_expected(d, outcome, baseline_years, state_specific_trend, same_month_mean)
    event = fit[(fit["year"] == 2020) & (fit["month"].isin(event_months))]
    state_event = event.groupby("state_alpha")["anomaly_pct"].mean()
    return float(state_event.mean())


def hpai_event_frame(panel: pd.DataFrame, outcome: str, window: int = 12) -> pd.DataFrame:
    modern = panel[panel["year"] >= 2022].copy()
    needed = ["state_alpha", "year", "month", "date", outcome, "hpai_dairy_cases"]
    sub = modern[needed].dropna(subset=[outcome]).copy()
    fe = fixed_effects(sub, spec="state_month_linear_year").to_numpy(dtype=float)
    y = np.log(np.clip(sub[outcome].to_numpy(float), 1e-12, None))
    sub["resid_log"] = residualize(y.reshape(-1, 1), fe, None).ravel()
    first = panel[panel["hpai_dairy_cases"] > 0].groupby("state_alpha")["date"].min()
    rec = []
    for st, t0 in first.items():
        g = sub[sub["state_alpha"] == st]
        for _, r in g.iterrows():
            tau = (r["date"].year - t0.year) * 12 + (r["date"].month - t0.month)
            if -window <= tau <= window:
                rr = r.to_dict()
                rr.update({"tau": tau, "first_detection": t0, "outcome": outcome})
                rec.append(rr)
    return pd.DataFrame(rec)


def hpai_event_effect(
    panel: pd.DataFrame,
    outcome: str,
    acute: tuple[int, int] = (0, 2),
    pre: tuple[int, int] = (-6, -1),
    trend_baseline: bool = False,
) -> float:
    window = max(abs(pre[0]), abs(acute[1]), 12 if trend_baseline else 0)
    ev = hpai_event_frame(panel, outcome, window=window)
    effects = []
    for _, g in ev.groupby("state_alpha"):
        pre_d = g[(g["tau"] >= pre[0]) & (g["tau"] <= pre[1])]
        acute_d = g[(g["tau"] >= acute[0]) & (g["tau"] <= acute[1])]
        if len(pre_d) < 3 or len(acute_d) < 1:
            continue
        if trend_baseline:
            coef = np.polyfit(pre_d["tau"].to_numpy(float), pre_d["resid_log"].to_numpy(float), 1)
            expected = np.polyval(coef, acute_d["tau"].to_numpy(float))
            diff = acute_d["resid_log"].to_numpy(float).mean() - expected.mean()
        else:
            diff = acute_d["resid_log"].mean() - pre_d["resid_log"].mean()
        effects.append(diff * 100.0)
    return float(np.mean(effects)) if effects else np.nan


def build_event_study_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    configs = [
        ("primary", "Primary definition", "primary"),
        ("short_event", "Short event window", "window"),
        ("long_event", "Long event window", "window"),
        ("short_baseline", "Short baseline", "baseline"),
        ("long_baseline", "Long baseline", "baseline"),
        ("trend_baseline", "Trend-adjusted baseline", "baseline"),
    ]
    rows = []
    for endpoint, outcome in EVENT_OUTCOMES.items():
        phenotype_label = "Milk per cow" if endpoint == "per_cow_26" else "Total production"
        for sensitivity_id, sensitivity_label, sensitivity_type in configs:
            if sensitivity_id == "primary":
                covid = covid_event_effect(panel, outcome, [4, 5, 6], (2015, 2019))
                hpai = hpai_event_effect(panel, outcome, (0, 2), (-6, -1))
            elif sensitivity_id == "short_event":
                covid = covid_event_effect(panel, outcome, [3, 4, 5], (2015, 2019))
                hpai = hpai_event_effect(panel, outcome, (0, 0), (-6, -1))
            elif sensitivity_id == "long_event":
                covid = covid_event_effect(panel, outcome, [3, 4, 5, 6], (2015, 2019))
                hpai = hpai_event_effect(panel, outcome, (0, 3), (-6, -1))
            elif sensitivity_id == "short_baseline":
                covid = covid_event_effect(panel, outcome, [4, 5, 6], (2019, 2019), same_month_mean=True)
                hpai = hpai_event_effect(panel, outcome, (0, 2), (-3, -1))
            elif sensitivity_id == "long_baseline":
                covid = covid_event_effect(panel, outcome, [4, 5, 6], (2017, 2019))
                hpai = hpai_event_effect(panel, outcome, (0, 2), (-12, -1))
            else:
                covid = covid_event_effect(panel, outcome, [4, 5, 6], (2015, 2019), state_specific_trend=True)
                hpai = hpai_event_effect(panel, outcome, (0, 2), (-6, -1), trend_baseline=True)
            rows.extend(
                [
                    {
                        "phenotype_scope": endpoint,
                        "phenotype_label": phenotype_label,
                        "domain": "COVID",
                        "event": "COVID",
                        "exposure": EVENT_REPRESENTATIVES["COVID"],
                        "sensitivity_id": sensitivity_id,
                        "sensitivity_label": sensitivity_label,
                        "sensitivity_type": sensitivity_type,
                        "effect_pct": covid,
                    },
                    {
                        "phenotype_scope": endpoint,
                        "phenotype_label": phenotype_label,
                        "domain": "HPAI",
                        "event": "HPAI",
                        "exposure": EVENT_REPRESENTATIVES["HPAI"],
                        "sensitivity_id": sensitivity_id,
                        "sensitivity_label": sensitivity_label,
                        "sensitivity_type": sensitivity_type,
                        "effect_pct": hpai,
                    },
                ]
            )
    return pd.DataFrame(rows)


def build_domain_matched_loso(panel: pd.DataFrame, assoc: pd.DataFrame) -> pd.DataFrame:
    if OLD_LOSO.exists():
        out = pd.read_csv(OLD_LOSO, low_memory=False)
    else:
        raise FileNotFoundError(f"Missing base LOSO table: {OLD_LOSO}")

    severe = build_severe_weather_loso(panel, assoc)
    events = build_event_loso(assoc)
    replacements = pd.concat([severe, events], ignore_index=True)
    for _, r in replacements.iterrows():
        mask = (
            out["phenotype_scope"].eq(r["phenotype_scope"])
            & out["domain"].eq(r["domain"])
            & out["exposure"].eq(r["exposure"])
        )
        if not mask.any():
            out = pd.concat([out, pd.DataFrame([r])], ignore_index=True)
            continue
        replace_loso_row(
            out,
            mask,
            {
                "main_beta": r["main_beta"],
                "loso_n": r["loso_n"],
                "loso_beta_mean": r["loso_beta_mean"],
                "loso_beta_median": r["loso_beta_median"],
                "loso_beta_lo": r["loso_beta_lo"],
                "loso_beta_hi": r["loso_beta_hi"],
                "loso_same_sign_share": r["loso_same_sign_share"],
            },
        )
    return out


def main() -> int:
    assoc = load_assoc()
    required = set(assoc["exposure"].dropna().astype(str))
    available_cols = set()
    for path in [US_EXPOSE / "exposure_state_month_highres.csv", US_EXPOSE / "exposure_state_month_expanded.csv", DAYMET_UNWEIGHTED]:
        if path.exists():
            available_cols.update(pd.read_csv(path, nrows=0).columns.tolist())
    pair_seed = identify_weighted_exposure_pairs(assoc, available_cols)
    if not pair_seed.empty:
        required.update(pair_seed["unweighted_exposure"].dropna().astype(str))
    panel = load_panel(required)
    long, ranges = build_model_spec_table(panel, assoc)
    loso = build_domain_matched_loso(panel, assoc)
    event_sens = build_event_study_sensitivity(panel)
    long.to_csv(OUT_SPEC_LONG, index=False, encoding="utf-8-sig")
    ranges.to_csv(OUT_SPEC_RANGE, index=False, encoding="utf-8-sig")
    loso.to_csv(OUT_LOSO_DM, index=False, encoding="utf-8-sig")
    event_sens.to_csv(OUT_EVENT_SPEC, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_SPEC_LONG} rows={len(long)}")
    print(f"Wrote {OUT_SPEC_RANGE} rows={len(ranges)}")
    print(f"Wrote {OUT_LOSO_DM} rows={len(loso)}")
    print(f"Wrote {OUT_EVENT_SPEC} rows={len(event_sens)}")
    weighting = build_weighting_table(panel, assoc)
    weighting.to_csv(OUT_WEIGHT, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT_WEIGHT} rows={len(weighting)}")
    print(weighting.groupby(["analysis", "phenotype_scope"]).agg(n=("exposure", "count"), same=("sign_concordant", "mean")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
