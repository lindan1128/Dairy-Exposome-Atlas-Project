#!/usr/bin/env python3
"""Native-only endpoint preference summary for the clean curated exposome pool.

The model estimates come from the same Point 1 ExWAS association table, but
multiple-testing tiers are recalculated within the clean curated native-only
pool defined in Supplementary Data 2.
"""

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
KEY = ["state_alpha", "year", "month"]

ASSOC = TAB / "point1_endpoint_exwas_associations.csv"
PRECLEAN_ASSOC = TAB / "point1_preclean_native_only_endpoint_exwas_associations.csv"
OUT_ASSOC = TAB / "point1_native_only_endpoint_exwas_associations.csv"
OUT_DOMAIN = TAB / "point1_native_only_class_domain_endpoint_preference.csv"
OUT_CLASS = TAB / "point1_native_only_class_endpoint_summary.csv"
CLEAN_DICT = (
    ROOT
    / "data"
    / "us_expose_new"
    / "suppl_data"
    / "supplementary_data_2_clean_curated_macro_exposome_exwas_variables.xlsx"
)
US_MILK = ROOT / "data" / "us_milk" / "processed"
US_EXPOSE_NEW = ROOT / "data" / "us_expose_new" / "processed"
OUT_AUDIT = TAB / "point1_clean_curated_exwas_variable_filter_audit.csv"

EXCLUDE_EXPOSURES = {
    # Outcome-derived structural proxy; excluded from the clean native-only
    # exposome screen so total-production panels are not dominated by itself.
    "herd_log_total_milk_production_lb",
}

CLASS_MAP = {
    "Heat": "Nature and climate",
    "Cold": "Nature and climate",
    "Drought": "Nature and climate",
    "Severe weather": "Nature and climate",
    "Wildfire smoke": "Nature and climate",
    "Forage condition": "Forage and pasture condition",
    "Air pollution": "Chemical and pollution exposome",
    "Agricultural pesticides": "Chemical and pollution exposome",
    "Industrial chemicals": "Chemical and pollution exposome",
    "Feed market": "Market and production-system",
    "Milk price / dairy market": "Market and production-system",
    "Dairy market": "Market and production-system",
    "Market demand": "Market and production-system",
    "Herd structure / scale": "Market and production-system",
    "Dairy scale": "Market and production-system",
    "Production system context": "Market and production-system",
    "HPAI": "Epidemic and infectious shocks",
    "COVID": "Epidemic and infectious shocks",
    "Pandemic shock": "Epidemic and infectious shocks",
}

CLASS_ORDER = [
    "Nature and climate",
    "Forage and pasture condition",
    "Chemical and pollution exposome",
    "Market and production-system",
    "Epidemic and infectious shocks",
]

DOMAIN_ORDER = [
    "Heat",
    "Cold",
    "Drought",
    "Severe weather",
    "Wildfire smoke",
    "Forage condition",
    "Air pollution",
    "Agricultural pesticides",
    "Industrial chemicals",
    "Feed market",
    "Milk price / dairy market",
    "Dairy market",
    "Market demand",
    "Dairy scale",
    "Herd structure / scale",
    "Production system context",
    "COVID",
    "HPAI",
    "Pandemic shock",
]

PHENO_ORDER = ["milk_per_cow_lb", "milk_production_lb_total26", "milk_production_lb"]
PHENO_LABEL = {
    "milk_per_cow_lb": "per_cow_26",
    "milk_production_lb_total26": "total_26",
    "milk_production_lb": "total_50",
}
PHENOTYPE_LONG_LABEL = {
    "milk_per_cow_lb": ("per_cow_26", "Milk per cow: 26 states"),
    "milk_production_lb_total26": ("total_26", "Total production: same 26 states"),
    "milk_production_lb": ("total_50", "Total production: all available states"),
}


def load_panel_for_missing_exposures() -> pd.DataFrame:
    milk = pd.read_csv(US_MILK / "state_month_panel.csv", low_memory=False)
    exp = pd.read_csv(US_EXPOSE_NEW / "exposure_state_month_expanded.csv", low_memory=False)
    overlap = [c for c in exp.columns if c in milk.columns and c not in KEY]
    if overlap:
        milk = milk.drop(columns=overlap)
    panel = milk.merge(exp, on=KEY, how="left")
    per_cow_states = sorted(panel.loc[panel["milk_per_cow_lb"].notna(), "state_alpha"].dropna().unique())
    panel["milk_production_lb_total26"] = np.where(
        panel["state_alpha"].isin(per_cow_states),
        panel["milk_production_lb"],
        np.nan,
    )
    return panel


def apply_clean_dictionary_metadata(df: pd.DataFrame, clean_dict: pd.DataFrame) -> pd.DataFrame:
    """Use the clean curated dictionary as the source of display metadata."""

    meta_cols = [
        "variables_en",
        "variables_ch",
        "domain",
        "domain_ch",
        "Subdomain",
        "mechanistic_domain_en",
        "mechanistic_domain_ch",
        "construct",
        "window",
        "form",
        "is_dairy_weighted_exposure",
        "n_nonmissing_exposure_months",
        "n_states_exposure",
        "year_min_exposure",
        "year_max_exposure",
        "exwas_method_used",
    ]
    meta = clean_dict[[c for c in meta_cols if c in clean_dict.columns]].copy()
    meta = meta.rename(
        columns={
            "variables_en": "exposure",
            "variables_ch": "_clean_exposure_zh",
            "domain": "_clean_domain",
            "domain_ch": "_clean_domain_zh",
            "Subdomain": "_clean_subdomain",
            "mechanistic_domain_en": "_clean_mechanistic_domain_en",
            "mechanistic_domain_ch": "_clean_mechanistic_domain_zh",
            "construct": "_clean_construct",
            "window": "_clean_window",
            "form": "_clean_form",
            "is_dairy_weighted_exposure": "_clean_is_dairy_weighted_exposure",
            "n_nonmissing_exposure_months": "_clean_n_nonmissing_exposure_months",
            "n_states_exposure": "_clean_n_states_exposure",
            "year_min_exposure": "_clean_year_min_exposure",
            "year_max_exposure": "_clean_year_max_exposure",
            "exwas_method_used": "_clean_exwas_method_used",
        }
    )
    out = df.merge(meta, on="exposure", how="left")
    direct_map = {
        "domain": "_clean_domain",
        "domain_zh": "_clean_domain_zh",
        "mechanistic_domain_en": "_clean_mechanistic_domain_en",
        "mechanistic_domain_zh": "_clean_mechanistic_domain_zh",
        "exposure_zh": "_clean_exposure_zh",
        "construct": "_clean_construct",
        "window": "_clean_window",
        "form": "_clean_form",
        "is_dairy_weighted_exposure": "_clean_is_dairy_weighted_exposure",
        "n_nonmissing_exposure_months": "_clean_n_nonmissing_exposure_months",
        "n_states_exposure": "_clean_n_states_exposure",
        "year_min_exposure": "_clean_year_min_exposure",
        "year_max_exposure": "_clean_year_max_exposure",
        "domain_matched_model": "_clean_exwas_method_used",
    }
    for col, clean_col in direct_map.items():
        if col in out.columns and clean_col in out.columns:
            out[col] = out[clean_col].combine_first(out[col])
    assoc_map = {
        "construct_assoc": "_clean_construct",
        "window_assoc": "_clean_window",
        "form_assoc": "_clean_form",
        "n_nonmissing_exposure_months_assoc": "_clean_n_nonmissing_exposure_months",
        "n_states_exposure_assoc": "_clean_n_states_exposure",
        "year_min_exposure_assoc": "_clean_year_min_exposure",
        "year_max_exposure_assoc": "_clean_year_max_exposure",
        "domain_assoc": "_clean_domain",
    }
    for col, clean_col in assoc_map.items():
        if col in out.columns and clean_col in out.columns:
            out[col] = out[clean_col].combine_first(out[col])
    drop_cols = [c for c in out.columns if c.startswith("_clean_")]
    return out.drop(columns=drop_cols)


def recompute_changed_exposures(df: pd.DataFrame, clean_dict: pd.DataFrame, exposures: set[str]) -> pd.DataFrame:
    """Refit association rows whose underlying exposure definition changed."""

    if not exposures:
        return df
    panel = load_panel_for_missing_exposures()
    meta = clean_dict.set_index("variables_en")
    out = df.copy()
    for exposure in sorted(exposures):
        if exposure not in panel.columns or exposure not in meta.index:
            continue
        m = meta.loc[exposure]
        for phenotype in PHENO_ORDER:
            if phenotype not in panel.columns:
                continue
            mask = out["exposure"].eq(exposure) & out["phenotype"].eq(phenotype)
            if not mask.any():
                continue
            fits = {
                spec: L.fit_exposure(
                    panel,
                    phenotype,
                    exposure,
                    spec=spec,
                    weight_col="milk_cows_head",
                )
                for spec in ["twoway", "sm_yeartrend", "state_year"]
            }
            fit = fits["twoway"]
            res = fit.get("results", {}).get(exposure, {})
            beta = res.get("beta", np.nan)
            p = res.get("p_cluster", np.nan)
            se = res.get("se_cluster", np.nan)
            beta_specs = [fits[s].get("results", {}).get(exposure, {}).get("beta", np.nan) for s in fits]
            p_specs = [fits[s].get("results", {}).get(exposure, {}).get("p_cluster", np.nan) for s in fits]
            same_sign = int(
                np.nansum([
                    np.isfinite(b) and np.isfinite(beta) and np.sign(b) == np.sign(beta)
                    for b in beta_specs
                ])
            )
            sig_specs = int(np.nansum([np.isfinite(x) and x < 0.05 for x in p_specs]))
            direction_stable = same_sign >= 3
            effect_direction = (
                "negative" if np.isfinite(beta) and beta < 0
                else "positive" if np.isfinite(beta) and beta > 0
                else "zero"
            )
            updates = {
                "domain_zh": m.get("domain_ch"),
                "domain": m.get("domain"),
                "mechanistic_domain_zh": m.get("mechanistic_domain_ch"),
                "mechanistic_domain_en": m.get("mechanistic_domain_en"),
                "exposure_zh": m.get("variables_ch"),
                "construct": m.get("construct"),
                "window": m.get("window"),
                "form": m.get("form"),
                "is_dairy_weighted_exposure": bool(m.get("is_dairy_weighted_exposure")),
                "n_nonmissing_exposure_months": m.get("n_nonmissing_exposure_months"),
                "n_states_exposure": m.get("n_states_exposure"),
                "year_min_exposure": m.get("year_min_exposure"),
                "year_max_exposure": m.get("year_max_exposure"),
                "status": fit.get("status"),
                "beta": beta,
                "se": se,
                "p": p,
                "n": fit.get("n"),
                "n_clusters": fit.get("n_clusters"),
                "incr_r2": fit.get("incr_r2"),
                "se_inflation": res.get("se_inflation", np.nan),
                "beta_twoway": beta_specs[0],
                "p_twoway": p_specs[0],
                "beta_sm_yeartrend": beta_specs[1],
                "p_sm_yeartrend": p_specs[1],
                "beta_state_year": beta_specs[2],
                "p_state_year": p_specs[2],
                "n_specs": 3,
                "n_specs_sig_p05": sig_specs,
                "n_specs_same_sign": same_sign,
                "construct_assoc": m.get("construct"),
                "window_assoc": m.get("window"),
                "form_assoc": m.get("form"),
                "n_nonmissing_exposure_months_assoc": m.get("n_nonmissing_exposure_months"),
                "n_states_exposure_assoc": m.get("n_states_exposure"),
                "year_min_exposure_assoc": m.get("year_min_exposure"),
                "year_max_exposure_assoc": m.get("year_max_exposure"),
                "spec_stable": direction_stable,
                "robust_expanded": direction_stable,
                "domain_assoc": m.get("domain"),
                "plot_pool": True,
                "domain_matched_model": m.get("exwas_method_used"),
                "domain_matched_recomputed": True,
                "direction_usable_for_main": direction_stable,
                "measurement_support_variable": False,
                "plot_beta": beta if direction_stable else np.nan,
                "plot_p": p if direction_stable else 1.0,
                "plot_incr_r2": fit.get("incr_r2") if direction_stable else 0.0,
                "plot_stat_source": "refit after clean dictionary definition update",
                "domain_matched_direction_stable": direction_stable,
                "robustness_score_0_4": 4 if direction_stable else 3,
                "direction_stable": direction_stable,
                "suggestive_stable": bool(np.isfinite(p) and p < 0.05 and direction_stable),
                "is_signal": bool(np.isfinite(p) and p < 0.05 and direction_stable),
                "effect_direction": effect_direction,
                "source_class": CLASS_MAP.get(m.get("domain")),
            }
            for col, val in updates.items():
                if col in out.columns:
                    out.loc[mask, col] = val
    return out


def append_missing_sparse_event_rows(df: pd.DataFrame, clean_dict: pd.DataFrame, missing_vars: list[str]) -> pd.DataFrame:
    """Add sparse-event variables that require endpoint-specific estimates."""

    supported = [v for v in missing_vars if v == "hpai_dairy_cases"]
    if not supported:
        return df
    panel = load_panel_for_missing_exposures()
    rows = []
    template_cols = df.columns.tolist()
    meta = clean_dict.set_index("variables_en")
    for exposure in supported:
        if exposure not in panel.columns:
            continue
        panel[exposure] = panel[exposure].fillna(0)
        m = meta.loc[exposure]
        for phenotype in PHENO_ORDER:
            if phenotype not in panel.columns:
                continue
            fits = {
                spec: L.fit_exposure(
                    panel,
                    phenotype,
                    exposure,
                    spec=spec,
                    weight_col="milk_cows_head",
                )
                for spec in ["twoway", "sm_yeartrend", "state_year"]
            }
            fit = fits["twoway"]
            res = fit.get("results", {}).get(exposure, {})
            beta = res.get("beta", np.nan)
            p = res.get("p_cluster", np.nan)
            se = res.get("se_cluster", np.nan)
            beta_specs = [fits[s].get("results", {}).get(exposure, {}).get("beta", np.nan) for s in fits]
            p_specs = [fits[s].get("results", {}).get(exposure, {}).get("p_cluster", np.nan) for s in fits]
            same_sign = int(
                np.nansum([
                    np.isfinite(b) and np.isfinite(beta) and np.sign(b) == np.sign(beta)
                    for b in beta_specs
                ])
            )
            sig_specs = int(np.nansum([np.isfinite(x) and x < 0.05 for x in p_specs]))
            scope, label = PHENOTYPE_LONG_LABEL[phenotype]
            row = {c: np.nan for c in template_cols}
            row.update(
                {
                    "domain_zh": m.get("domain_ch"),
                    "domain": "HPAI",
                    "mechanistic_domain_zh": m.get("mechanistic_domain_ch"),
                    "mechanistic_domain_en": m.get("mechanistic_domain_en"),
                    "exposure_zh": m.get("variables_ch"),
                    "exposure": exposure,
                    "construct": m.get("construct"),
                    "window": m.get("window"),
                    "form": m.get("form"),
                    "is_dairy_weighted_exposure": bool(m.get("is_dairy_weighted_exposure")),
                    "n_nonmissing_exposure_months": m.get("n_nonmissing_exposure_months"),
                    "n_states_exposure": m.get("n_states_exposure"),
                    "year_min_exposure": m.get("year_min_exposure"),
                    "year_max_exposure": m.get("year_max_exposure"),
                    "is_weighted": True,
                    "cons": m.get("construct"),
                    "phenotype": phenotype,
                    "status": fit.get("status"),
                    "beta": beta,
                    "se": se,
                    "p": p,
                    "n": fit.get("n"),
                    "n_clusters": fit.get("n_clusters"),
                    "incr_r2": fit.get("incr_r2"),
                    "se_inflation": res.get("se_inflation", np.nan),
                    "beta_twoway": beta_specs[0],
                    "p_twoway": p_specs[0],
                    "beta_sm_yeartrend": beta_specs[1],
                    "p_sm_yeartrend": p_specs[1],
                    "beta_state_year": beta_specs[2],
                    "p_state_year": p_specs[2],
                    "n_specs": 3,
                    "n_specs_sig_p05": sig_specs,
                    "n_specs_same_sign": same_sign,
                    "construct_assoc": m.get("construct"),
                    "window_assoc": m.get("window"),
                    "form_assoc": m.get("form"),
                    "n_nonmissing_exposure_months_assoc": m.get("n_nonmissing_exposure_months"),
                    "n_states_exposure_assoc": m.get("n_states_exposure"),
                    "year_min_exposure_assoc": m.get("year_min_exposure"),
                    "year_max_exposure_assoc": m.get("year_max_exposure"),
                    "spec_stable": same_sign >= 3,
                    "robust_expanded": same_sign >= 3,
                    "domain_assoc": "HPAI",
                    "plot_pool": True,
                    "domain_matched_model": m.get("exwas_method_used"),
                    "domain_matched_recomputed": False,
                    "direction_usable_for_main": same_sign >= 3,
                    "measurement_support_variable": False,
                    "plot_beta": beta if same_sign >= 3 else np.nan,
                    "plot_p": p if same_sign >= 3 else 1.0,
                    "plot_incr_r2": fit.get("incr_r2") if same_sign >= 3 else 0.0,
                    "plot_stat_source": "sparse-event ordinary FE screen; HPAI event study remains primary",
                    "domain_matched_direction_stable": same_sign >= 3,
                    "phenotype_scope": scope,
                    "phenotype_label": label,
                    "robustness_score_0_4": 4 if same_sign >= 3 else 3,
                    "direction_stable": same_sign >= 3,
                    "suggestive_stable": bool(np.isfinite(p) and p < 0.05 and same_sign >= 3),
                    "is_signal": bool(np.isfinite(p) and p < 0.05 and same_sign >= 3),
                    "effect_direction": (
                        "negative" if np.isfinite(beta) and beta < 0
                        else "positive" if np.isfinite(beta) and beta > 0
                        else "zero"
                    ),
                    "source_class": "Epidemic and infectious shocks",
                }
            )
            rows.append(row)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows, columns=template_cols)], ignore_index=True)
    return df


def by_adjust(p: pd.Series) -> pd.Series:
    """Benjamini-Yekutieli adjusted p-values."""

    p = pd.to_numeric(p, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    vals = p[ok].to_numpy(float)
    m = len(vals)
    if m == 0:
        return out
    c_m = np.sum(1 / np.arange(1, m + 1))
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * m * c_m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    idx = p[ok].index.to_numpy()[order]
    out.loc[idx] = adj
    return out


def load_clean_dictionary() -> pd.DataFrame:
    if not CLEAN_DICT.exists():
        raise FileNotFoundError(f"Missing clean curated dictionary: {CLEAN_DICT}")
    d = pd.read_excel(CLEAN_DICT, sheet_name="exwas_variables")
    d = d[d["used_in_exwas"].fillna(False).astype(bool)].copy()
    d["variables_en"] = d["variables_en"].astype(str)
    return d


def load_source_associations() -> tuple[pd.DataFrame, Path]:
    for path in [ASSOC, PRECLEAN_ASSOC, OUT_ASSOC]:
        if path.exists():
            return pd.read_csv(path, low_memory=False), path
    raise FileNotFoundError(
        f"Could not find any source association table among {ASSOC}, {PRECLEAN_ASSOC}, {OUT_ASSOC}"
    )


def signal_tier(row: pd.Series) -> str:
    if bool(row["native_bonferroni_sig"]):
        return "Bonferroni"
    if bool(row["native_by_fdr_sig"]):
        return "BY-FDR"
    if row["plot_p"] < 0.05 and row.get("n_specs_same_sign", 0) >= 3:
        return "p<0.05 + direction-stable"
    return "n.s."


def summarize(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, s in df.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        p = pd.to_numeric(s["plot_p"], errors="coerce")
        r2 = pd.to_numeric(s["plot_incr_r2"], errors="coerce")
        row = dict(zip(groups, keys))
        row.update(
            {
                "n_variables": s["exposure"].nunique(),
                "n_constructs": s["construct"].nunique(),
                "n_p05": int((p < 0.05).sum()),
                "n_p01": int((p < 0.01).sum()),
                "n_by": int(s["native_by_fdr_sig"].fillna(False).sum()),
                "n_bonf": int(s["native_bonferroni_sig"].fillna(False).sum()),
                "share_p01": float((p < 0.01).mean()),
                "share_by": float(s["native_by_fdr_sig"].fillna(False).mean()),
                "median_r2": float(np.nanmedian(r2)) if np.isfinite(r2).any() else np.nan,
                "max_r2": float(np.nanmax(r2)) if np.isfinite(r2).any() else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def endpoint_pattern(row: pd.Series) -> str:
    pc = row.get("n_by_per_cow_26", 0), row.get("n_p01_per_cow_26", 0), row.get("max_r2_per_cow_26", 0)
    t26 = row.get("n_by_total_26", 0), row.get("n_p01_total_26", 0), row.get("max_r2_total_26", 0)
    t50 = row.get("n_by_total_50", 0), row.get("n_p01_total_50", 0), row.get("max_r2_total_50", 0)
    if pc[0] > max(t26[0], t50[0]):
        return "per-cow stronger"
    if max(t26[0], t50[0]) > pc[0]:
        return "total stronger"
    if pc[1] > max(t26[1], t50[1]):
        return "per-cow leaning"
    if max(t26[1], t50[1]) > pc[1]:
        return "total leaning"
    if pc[2] > max(t26[2], t50[2]) * 1.5:
        return "per-cow weak/leaning"
    if max(t26[2], t50[2]) > pc[2] * 1.5:
        return "total weak/leaning"
    return "mixed/weak"


def main() -> int:
    clean_dict = load_clean_dictionary()
    clean_vars = set(clean_dict["variables_en"])
    df, source_path = load_source_associations()
    df = df[df["window"].eq("native")].copy()
    df = df[~df["exposure"].isin(EXCLUDE_EXPOSURES)].copy()
    source_vars = set(df["exposure"].astype(str))
    initially_missing_from_assoc = sorted(clean_vars - source_vars)
    df = append_missing_sparse_event_rows(df, clean_dict, initially_missing_from_assoc)
    df = recompute_changed_exposures(df, clean_dict, {"herd_dairy_county_concentration_proxy"})
    df = apply_clean_dictionary_metadata(df, clean_dict)
    source_vars = set(df["exposure"].astype(str))
    missing_from_assoc = sorted(clean_vars - source_vars)
    extra_in_source = sorted(source_vars - clean_vars)
    audit = pd.DataFrame(
        {
            "source_association_table": [str(source_path)],
            "clean_dictionary": [str(CLEAN_DICT)],
            "n_clean_dictionary_variables": [len(clean_vars)],
            "n_source_native_variables": [len(source_vars)],
            "n_clean_variables_available_in_source": [len(clean_vars & source_vars)],
            "n_clean_variables_initially_missing_from_source": [len(initially_missing_from_assoc)],
            "initially_missing_clean_variables": [";".join(initially_missing_from_assoc)],
            "n_clean_variables_missing_from_source": [len(missing_from_assoc)],
            "missing_clean_variables": [";".join(missing_from_assoc)],
            "n_source_variables_excluded_by_clean_dictionary": [len(extra_in_source)],
        }
    )
    audit.to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")
    df = df[df["exposure"].isin(clean_vars)].copy()
    df["source_class"] = df["domain"].map(CLASS_MAP)
    missing = sorted(df.loc[df["source_class"].isna(), "domain"].dropna().unique())
    if missing:
        raise SystemExit(f"Missing class mapping for: {missing}")

    df["plot_p"] = pd.to_numeric(df["plot_p"], errors="coerce")
    df["plot_incr_r2"] = pd.to_numeric(df["plot_incr_r2"], errors="coerce")

    pieces = []
    for pheno, s in df.groupby("phenotype", dropna=False):
        s = s.copy()
        m = s["plot_p"].notna().sum()
        s["native_bonferroni_p"] = np.minimum(s["plot_p"] * m, 1.0)
        s["native_bonferroni_sig"] = s["native_bonferroni_p"] < 0.05
        s["native_by_fdr_p"] = by_adjust(s["plot_p"])
        s["native_by_fdr_sig"] = s["native_by_fdr_p"] < 0.05
        pieces.append(s)
    native = pd.concat(pieces, ignore_index=True)
    native["native_signal_tier"] = native.apply(signal_tier, axis=1)
    native.to_csv(OUT_ASSOC, index=False, encoding="utf-8-sig")

    class_summary = summarize(native, ["source_class", "phenotype"])
    class_summary["phenotype_scope"] = class_summary["phenotype"].map(PHENO_LABEL)
    class_summary["class_order"] = class_summary["source_class"].map({x: i for i, x in enumerate(CLASS_ORDER)})
    class_summary["phenotype_order"] = class_summary["phenotype"].map({x: i for i, x in enumerate(PHENO_ORDER)})
    class_summary = class_summary.sort_values(["class_order", "phenotype_order"]).drop(
        columns=["class_order", "phenotype_order"]
    )
    class_summary.to_csv(OUT_CLASS, index=False, encoding="utf-8-sig")

    domain = summarize(native, ["source_class", "domain", "phenotype"])
    domain["phenotype_scope"] = domain["phenotype"].map(PHENO_LABEL)
    wide = domain.pivot(
        index=["source_class", "domain"],
        columns="phenotype_scope",
        values=["n_variables", "n_p05", "n_p01", "n_by", "n_bonf", "max_r2", "median_r2"],
    )
    wide.columns = ["_".join([str(x) for x in col if x]) for col in wide.columns]
    wide = wide.reset_index()
    wide["native_endpoint_pattern"] = wide.apply(endpoint_pattern, axis=1)
    wide["class_order"] = wide["source_class"].map({x: i for i, x in enumerate(CLASS_ORDER)})
    wide["domain_order"] = wide["domain"].map({x: i for i, x in enumerate(DOMAIN_ORDER)})
    wide = wide.sort_values(["class_order", "domain_order"]).drop(columns=["class_order", "domain_order"])
    wide.to_csv(OUT_DOMAIN, index=False, encoding="utf-8-sig")

    print(
        "Clean curated native-only association table: "
        f"{native['exposure'].nunique()} variables x {native['phenotype'].nunique()} phenotypes "
        f"(dictionary={len(clean_vars)}, missing_from_source={len(missing_from_assoc)})"
    )
    if missing_from_assoc:
        print("Missing clean variables from source association table:")
        for v in missing_from_assoc:
            print(f"  - {v}")
    print(f"Wrote {OUT_ASSOC}")
    print(f"Wrote {OUT_DOMAIN}")
    print(f"Wrote {OUT_CLASS}")
    print(f"Wrote {OUT_AUDIT}")
    print("\nNative-only domain endpoint preference:")
    print(wide[["source_class", "domain", "native_endpoint_pattern"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
