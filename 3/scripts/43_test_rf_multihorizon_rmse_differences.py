#!/usr/bin/env python3
"""Test paired annual held-out RMSE differences across forecast horizons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parents[1]
TABLES = HERE / "tables"
INPUT = TABLES / "supp_point3_rf_multihorizon_performance_by_year_scope.csv"
OUTPUT = TABLES / "supp_point3_rf_multihorizon_paired_rmse_tests.csv"
LB_TO_KG = 0.45359237


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    """Return monotone BH-adjusted P values in the original order."""
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def significance_label(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return "ns"


def main() -> None:
    performance = pd.read_csv(INPUT)
    annual = performance.loc[performance.scope.eq("overall")].pivot(
        index=["horizon_months", "test_year"],
        columns="model",
        values="target_milk_rmse_lb_per_cow",
    ).reset_index()
    expected = {"history_rf_nested", "history_exposure_selected_rf_nested"}
    if not expected.issubset(annual.columns):
        raise RuntimeError("Both held-out forecast models are required for paired testing.")

    rows: list[dict[str, float | int]] = []
    for horizon, data in annual.groupby("horizon_months", sort=True):
        history = data.history_rf_nested.to_numpy(float)
        exposome = data.history_exposure_selected_rf_nested.to_numpy(float)
        difference = history - exposome  # Positive: adding the exposome lowers RMSE.
        n = len(difference)
        mean_difference = float(difference.mean())
        se_difference = float(difference.std(ddof=1) / np.sqrt(n))
        ci_half_width = float(stats.t.ppf(0.975, n - 1) * se_difference)
        test = stats.ttest_rel(history, exposome, alternative="two-sided")
        rows.append({
            "horizon_months": int(horizon),
            "n_outer_test_years": n,
            "history_rmse_kg_per_cow": float(history.mean() * LB_TO_KG),
            "history_exposome_rmse_kg_per_cow": float(exposome.mean() * LB_TO_KG),
            "rmse_reduction_kg_per_cow": mean_difference * LB_TO_KG,
            "rmse_reduction_kg_95ci_lower": (mean_difference - ci_half_width) * LB_TO_KG,
            "rmse_reduction_kg_95ci_upper": (mean_difference + ci_half_width) * LB_TO_KG,
            "rmse_reduction_vs_history_pct": float(100 * mean_difference / history.mean()),
            "paired_t_statistic": float(test.statistic),
            "paired_t_p_two_sided": float(test.pvalue),
        })
    results = pd.DataFrame(rows)
    results["paired_t_q_bh_fdr"] = benjamini_hochberg(results.paired_t_p_two_sided)
    results["fdr_significance"] = results.paired_t_q_bh_fdr.map(significance_label)
    results.to_csv(OUTPUT, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
