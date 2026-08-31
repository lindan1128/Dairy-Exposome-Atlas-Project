#!/usr/bin/env python3
"""Audit agreement between the retained HGB analysis and RF robustness run.

This script does not modify HGB or RF results. It places their independently
generated held-out metrics and domain-SHAP summaries on the same keys so that
the predictive and explanatory agreement is explicit and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parents[1]
TABLES = HERE / "tables"


def forecast_agreement() -> pd.DataFrame:
    sources = {
        "hgb": (
            TABLES / "point3_point4aligned_multihorizon_performance_by_year_scope.csv",
            TABLES / "point3_point4aligned_multihorizon_paired_rmse_tests.csv",
            "history_hgb_nested",
            "history_exposure_selected_hgb_nested",
        ),
        "rf": (
            TABLES / "supp_point3_rf_multihorizon_performance_by_year_scope.csv",
            TABLES / "supp_point3_rf_multihorizon_paired_rmse_tests.csv",
            "history_rf_nested",
            "history_exposure_selected_rf_nested",
        ),
    }
    parts = []
    for name, (performance_path, tests_path, history_model, exposure_model) in sources.items():
        performance = pd.read_csv(performance_path)
        overall = performance.loc[performance.scope.eq("overall")]
        summary = overall.groupby(["horizon_months", "model"], as_index=False).agg(
            rmse_lb_per_cow=("target_milk_rmse_lb_per_cow", "mean"),
            n_outer_test_years=("test_year", "nunique"),
        )
        pivot = summary.pivot(index="horizon_months", columns="model", values="rmse_lb_per_cow").reset_index()
        result = pd.DataFrame({
            "horizon_months": pivot.horizon_months,
            f"{name}_history_rmse_lb_per_cow": pivot[history_model],
            f"{name}_history_exposure_rmse_lb_per_cow": pivot[exposure_model],
        })
        result[f"{name}_rmse_reduction_lb_per_cow"] = (
            result[f"{name}_history_rmse_lb_per_cow"] - result[f"{name}_history_exposure_rmse_lb_per_cow"]
        )
        tests = pd.read_csv(tests_path).loc[:, ["horizon_months", "rmse_reduction_kg_per_cow", "paired_t_q_bh_fdr"]]
        tests = tests.rename(columns={
            "rmse_reduction_kg_per_cow": f"{name}_rmse_reduction_kg_per_cow",
            "paired_t_q_bh_fdr": f"{name}_paired_rmse_q_bh_fdr",
        })
        parts.append(result.merge(tests, on="horizon_months", how="left", validate="one_to_one"))
    joined = parts[0].merge(parts[1], on="horizon_months", how="inner", validate="one_to_one")
    joined["same_rmse_improvement_direction"] = (
        np.sign(joined.hgb_rmse_reduction_lb_per_cow) == np.sign(joined.rf_rmse_reduction_lb_per_cow)
    )
    return joined


def regional_agreement() -> pd.DataFrame:
    hgb = pd.read_csv(TABLES / "point3_point4aligned_regional_class_net_shap_by_phase.csv")
    rf = pd.read_csv(TABLES / "supp_point3_rf_regional_class_net_shap_by_phase.csv")
    keys = ["region", "phase", "class_label"]
    merged = hgb[keys + ["share_pct"]].merge(
        rf[keys + ["share_pct"]], on=keys, suffixes=("_hgb", "_rf"), validate="one_to_one"
    )
    rows = []
    for (region, phase), data in merged.groupby(["region", "phase"], sort=True):
        rho, p_value = stats.spearmanr(data.share_pct_hgb, data.share_pct_rf)
        rows.append({
            "region": region,
            "phase": phase,
            "n_classes": len(data),
            "spearman_rho_class_shares": rho,
            "spearman_p_class_shares": p_value,
            "hgb_top_class": data.loc[data.share_pct_hgb.idxmax(), "class_label"],
            "rf_top_class": data.loc[data.share_pct_rf.idxmax(), "class_label"],
        })
    return pd.DataFrame(rows)


def spatial_agreement() -> pd.DataFrame:
    hgb = pd.read_csv(TABLES / "point3_multivariable_class_contribution_latlon_spearman.csv")
    rf = pd.read_csv(TABLES / "supp_point3_rf_class_contribution_latlon_spearman.csv")
    keys = ["scope", "phase", "class_label", "coordinate"]
    columns = keys + ["spearman_rho", "spearman_q_bh_fdr"]
    merged = hgb[columns].merge(rf[columns], on=keys, suffixes=("_hgb", "_rf"), validate="one_to_one")
    merged["same_gradient_direction"] = np.sign(merged.spearman_rho_hgb) == np.sign(merged.spearman_rho_rf)
    merged["robust_in_both_models"] = (
        merged.spearman_q_bh_fdr_hgb.lt(0.05) & merged.spearman_q_bh_fdr_rf.lt(0.05)
    )
    return merged


def main() -> None:
    forecasts = forecast_agreement()
    regional = regional_agreement()
    spatial = spatial_agreement()
    forecasts.to_csv(TABLES / "supp_point3_rf_hgb_forecast_agreement.csv", index=False)
    regional.to_csv(TABLES / "supp_point3_rf_hgb_regional_class_agreement.csv", index=False)
    spatial.to_csv(TABLES / "supp_point3_rf_hgb_spatial_gradient_agreement.csv", index=False)
    print("Forecast agreement")
    print(forecasts.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nRegional domain-share agreement")
    print(regional.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nSpatial-gradient agreement")
    print(spatial.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
