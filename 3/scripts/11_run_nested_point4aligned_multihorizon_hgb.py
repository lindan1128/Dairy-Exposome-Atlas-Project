#!/usr/bin/env python3
"""Nested rolling-origin 1-12 month HGB forecasts using Point 4's 186 exposures.

Each horizon is a separate forecasting task. Within every outer forecast year,
all feature screening and HGB tuning occur only in earlier rolling validation
folds. The final held-out model never uses its test year's outcomes.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from scipy.interpolate import PchipInterpolator
from scipy.stats import t as student_t



HERE = Path(__file__).resolve().parents[1]
TAB, FIG, LOG = HERE / "tables", HERE / "figures", HERE / "logs"
DEFAULT_TEST_YEARS = tuple(range(2015, 2026))
INNER_START_YEAR = 2010
RANDOM_STATE = 20260822
LB_TO_KG = 0.45359237
K_GRID = (25, 75, 150)
MODEL_GRID = tuple(
    {
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
    }
    for max_iter, learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization in itertools.product(
        (100, 120, 140, 160),
        (0.030, 0.040, 0.050),
        (5, 10, 20, 30),
        (20, 25, 30, 35),
        (0.30, 0.40, 0.50),
    )
)


def weighted_rmse(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(weight) & (weight > 0)
    return float(np.sqrt(np.average((actual[valid] - predicted[valid]) ** 2, weights=weight[valid])))


def weighted_mae(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(weight) & (weight > 0)
    return float(np.average(np.abs(actual[valid] - predicted[valid]), weights=weight[valid]))


def loss_to_milk(current_milk: np.ndarray, loss_pct: np.ndarray) -> np.ndarray:
    return np.asarray(current_milk, float) * np.exp(-np.asarray(loss_pct, float) / 100.0)


def score(data: pd.DataFrame, predicted_loss: np.ndarray, horizon: int) -> dict[str, float]:
    loss = data[f"target_loss_h{horizon}_pct"].to_numpy(float)
    milk = data[f"target_milk_h{horizon}_lb_per_cow"].to_numpy(float)
    weight = data.milk_cows_head.to_numpy(float)
    predicted_milk = loss_to_milk(data.milk_per_cow_lb.to_numpy(float), predicted_loss)
    return {
        "loss_rmse_pct_points": weighted_rmse(loss, predicted_loss, weight),
        "loss_mae_pct_points": weighted_mae(loss, predicted_loss, weight),
        "target_milk_rmse_lb_per_cow": weighted_rmse(milk, predicted_milk, weight),
        "target_milk_mae_lb_per_cow": weighted_mae(milk, predicted_milk, weight),
    }


def history_features(horizon: int, panel: pd.DataFrame) -> list[str]:
    states = sorted(column for column in panel.columns if column.startswith("state_") and column != "state_alpha")
    return [
        "log_milk_per_cow", "log_milk_per_cow_lag1", "log_milk_per_cow_lag2",
        "milk_loss_lag1", "year_centered", *states,
        *[f"h{horizon}__target_month_{month:02d}" for month in range(1, 13)],
    ]


def prepare(panel: pd.DataFrame, horizon: int, mask: pd.Series, history: list[str]) -> pd.DataFrame:
    required = [
        f"target_loss_h{horizon}_pct", f"target_milk_h{horizon}_lb_per_cow",
        "milk_cows_head", "milk_per_cow_lb", *history,
    ]
    return panel.loc[mask].replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict) -> tuple[HistGradientBoostingRegressor, np.ndarray]:
    model = HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    model.fit(
        train[features], train._target_loss.to_numpy(float),
        sample_weight=train.milk_cows_head.to_numpy(float),
    )
    return model, model.predict(test[features])


def rank_exposures(train: pd.DataFrame, history: list[str], exposures: list[str]) -> list[str]:
    """Training-only residual screen used solely to choose the HGB input set."""
    y = train._target_loss.to_numpy(float)
    weight = train.milk_cows_head.to_numpy(float)
    history_x = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(train[history])
    ridge = Ridge(alpha=10.0).fit(history_x, y, sample_weight=weight)
    residual = y - ridge.predict(history_x)
    exposure_x = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(train[exposures])
    normalized_weight = weight / weight.sum()
    residual = residual - np.sum(normalized_weight * residual)
    centered = exposure_x - np.sum(normalized_weight[:, None] * exposure_x, axis=0)
    numerator = np.sum(normalized_weight[:, None] * centered * residual[:, None], axis=0)
    denominator = np.sqrt(
        np.sum(normalized_weight[:, None] * centered ** 2, axis=0)
        * np.sum(normalized_weight * residual ** 2)
    )
    correlation = np.divide(np.abs(numerator), denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    return pd.Series(correlation, index=exposures).sort_values(ascending=False).index.tolist()


def precompute_inner_cv(
    panel: pd.DataFrame, horizon: int, history: list[str], exposures: list[str], validation_years: list[int]
) -> pd.DataFrame:
    """Fit each legal inner fold once; later outer years only aggregate its results.

    For a fixed horizon and validation year, the training rule (target date before
    that validation year) is invariant to any later outer test year. Reusing this
    result is therefore exactly equivalent to refitting the same legal fold.
    """
    rows = []
    target_year = f"target_year_h{horizon}"
    for validation_year in validation_years:
        print(f"  precomputing horizon {horizon}, inner validation year {validation_year}...", flush=True)
        train = prepare(panel, horizon, panel[target_year].lt(validation_year), history)
        valid = prepare(panel, horizon, panel[target_year].eq(validation_year), history)
        if train.empty or valid.empty:
            continue
        train["_target_loss"] = train[f"target_loss_h{horizon}_pct"]
        valid["_target_loss"] = valid[f"target_loss_h{horizon}_pct"]
        for candidate_id, params in enumerate(MODEL_GRID):
            _, prediction = fit_predict(train, valid, history, params)
            rows.append({
                "horizon_months": horizon, "inner_validation_year": validation_year,
                "model": "history_hgb_nested", "candidate_id": candidate_id,
                "k_exposure_features": 0, **params, **score(valid, prediction, horizon),
            })
        ranking = rank_exposures(train, history, exposures)
        for k in K_GRID:
            selected = ranking[:k]
            for candidate_id, params in enumerate(MODEL_GRID):
                _, prediction = fit_predict(train, valid, [*history, *selected], params)
                rows.append({
                    "horizon_months": horizon, "inner_validation_year": validation_year,
                    "model": "history_exposure_selected_hgb_nested", "candidate_id": candidate_id,
                    "k_exposure_features": k, **params, **score(valid, prediction, horizon),
                })
    return pd.DataFrame(rows)


def choose_outer_config(inner_cv: pd.DataFrame, outer_year: int, model: str, max_inner_folds: int) -> tuple[int, int, dict, list[int]]:
    years = sorted(year for year in inner_cv.inner_validation_year.unique() if year < outer_year)[-max_inner_folds:]
    candidates = inner_cv.loc[
        inner_cv.model.eq(model) & inner_cv.inner_validation_year.isin(years)
    ].groupby(["candidate_id", "k_exposure_features"], as_index=False).agg(
        rmse=("target_milk_rmse_lb_per_cow", "mean"), mae=("target_milk_mae_lb_per_cow", "mean")
    ).sort_values(["rmse", "mae", "candidate_id", "k_exposure_features"])
    if candidates.empty:
        raise RuntimeError(f"No legal inner candidates for outer year {outer_year}, model {model}.")
    winner = candidates.iloc[0]
    candidate_id, k = int(winner.candidate_id), int(winner.k_exposure_features)
    return candidate_id, k, MODEL_GRID[candidate_id], years


def state_feature_shap(
    model: HistGradientBoostingRegressor, test: pd.DataFrame, features: list[str], exposure_meta: pd.DataFrame, test_year: int
) -> pd.DataFrame:
    values = np.asarray(shap.TreeExplainer(model).shap_values(test[features]))
    if values.ndim == 3:
        values = values[:, :, 0]
    shap_df = pd.DataFrame(values, columns=features, index=test.index)
    exposure_features = [feature for feature in features if feature in set(exposure_meta.feature)]
    work = shap_df[exposure_features].copy()
    work["state_alpha"] = test.state_alpha.to_numpy()
    work["region"] = test.region.to_numpy()
    parts = []
    for (state, region), values_by_state in work.groupby(["state_alpha", "region"], sort=False):
        values_only = values_by_state[exposure_features]
        parts.append(pd.DataFrame({
            "test_year": test_year, "state_alpha": state, "region": region,
            "feature": exposure_features,
            "mean_abs_shap": np.abs(values_only).mean(axis=0).to_numpy(),
            "mean_signed_shap": values_only.mean(axis=0).to_numpy(),
            "n_state_months": len(values_by_state),
        }))
    return pd.concat(parts, ignore_index=True).merge(exposure_meta, on="feature", how="left", validate="many_to_one")


def normalize_svg_text_style(path: Path) -> None:
    """Match the explicit Arial SVG text style used by Point 2 figures."""
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(
        r"font: ([0-9.]+)px 'Arial'",
        r"font-size: \1px; font-family: 'Arial'",
        svg,
    )
    svg = svg.replace("font-size: 9px;", "font-size: 9.00px;")
    path.write_text(svg, encoding="utf-8")


def make_figure(performance: pd.DataFrame, show_legend: bool) -> None:
    """Plot annual held-out RMSE means with 95% CIs across outer test years."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "svg.fonttype": "none",
    })
    annual = (
        performance.loc[performance.scope.eq("overall")]
        .groupby(["horizon_months", "model"], as_index=False)
        .agg(
            mean_rmse=("target_milk_rmse_lb_per_cow", "mean"),
            sd_rmse=("target_milk_rmse_lb_per_cow", "std"),
            n_outer_years=("test_year", "nunique"),
        )
        .sort_values(["model", "horizon_months"])
    )
    annual["ci_half_width"] = (
        annual.apply(
            lambda row: student_t.ppf(0.975, int(row.n_outer_years) - 1)
            * row.sd_rmse / np.sqrt(row.n_outer_years),
            axis=1,
        )
    )
    annual[["mean_rmse", "ci_half_width"]] *= LB_TO_KG
    labels = {
        "history_hgb_nested": "Baseline forecast",
        "history_exposure_selected_hgb_nested": "Exposome-enhanced forecast",
    }
    colors = {
        "history_hgb_nested": "#B5C99F",
        "history_exposure_selected_hgb_nested": "#FFCC88",
    }
    figure_height = 3.9 if show_legend else 2.325
    fig, ax = plt.subplots(figsize=(8.25, figure_height), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    model_data = {model: data.copy() for model, data in annual.groupby("model", sort=False)}
    x_smooth = np.linspace(1, 12, 221)

    def smooth(data: pd.DataFrame, column: str, x_values: np.ndarray = x_smooth) -> np.ndarray:
        if len(data) == 1:
            return np.repeat(data[column].iloc[0], len(x_values))
        return PchipInterpolator(data.horizon_months.to_numpy(), data[column].to_numpy())(x_values)

    history = model_data["history_hgb_nested"]
    exposed = model_data["history_exposure_selected_hgb_nested"]
    for model in ("history_hgb_nested", "history_exposure_selected_hgb_nested"):
        data = model_data[model]
        mean = smooth(data, "mean_rmse")
        lower = PchipInterpolator(
            data.horizon_months.to_numpy(),
            (data.mean_rmse - data.ci_half_width).to_numpy(),
        )(x_smooth)
        upper = PchipInterpolator(
            data.horizon_months.to_numpy(),
            (data.mean_rmse + data.ci_half_width).to_numpy(),
        )(x_smooth)
        ax.fill_between(
            x_smooth,
            lower,
            upper,
            color=colors[model],
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            x_smooth,
            mean,
            color="#222222",
            linewidth=1.35,
            zorder=3,
        )
    connector_months = np.arange(1, 10)
    ax.vlines(
        connector_months,
        smooth(history, "mean_rmse", connector_months),
        smooth(exposed, "mean_rmse", connector_months),
        color="#222222",
        linewidth=0.70,
        zorder=2.8,
    )
    ax.set_xlim(1, 12)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Forecast horizon (months ahead)", fontsize=9)
    ax.set_ylabel("Held-out RMSE (kg/cow)", fontsize=9)
    ax.set_ylim(10, 30)
    ax.set_yticks([10, 15, 20, 25, 30])
    ax.tick_params(axis="both", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(
            handles=[
                Patch(facecolor=colors[model], edgecolor="none", alpha=0.55, label=labels[model])
                for model in ("history_hgb_nested", "history_exposure_selected_hgb_nested")
            ],
            ncol=2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            frameon=False,
            fontsize=9,
            columnspacing=1.4,
            handlelength=1.5,
        )
    suffix = "" if show_legend else "_wo_legend"
    output = FIG / f"main_point3_point4aligned_multihorizon_model_comparison{suffix}.svg"
    fig.savefig(output, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    normalize_svg_text_style(output)
    plt.close(fig)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", default="1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--test-years", default=",".join(map(str, DEFAULT_TEST_YEARS)))
    parser.add_argument("--max-inner-folds", type=int, default=5)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Redraw the figure from saved annual held-out performance.")
    args = parser.parse_args()
    horizons, test_years = parse_ints(args.horizons), parse_ints(args.test_years)
    if any(horizon not in range(1, 13) for horizon in horizons):
        raise ValueError("Horizon must be an integer from 1 to 12.")
    if args.max_inner_folds < 1:
        raise ValueError("--max-inner-folds must be positive.")
    for directory in (TAB, FIG, LOG):
        directory.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        performance = pd.read_csv(TAB / "point3_point4aligned_multihorizon_performance_by_year_scope.csv")
        make_figure(performance, show_legend=True)
        make_figure(performance, show_legend=False)
        return 0

    panel = pd.read_csv(TAB / "point3_point4aligned_multihorizon_model_panel.csv", low_memory=False)
    exposure_meta = pd.read_csv(TAB / "point3_point4aligned_multihorizon_exposure_dictionary.csv")
    exposures = exposure_meta.feature.tolist()
    if exposure_meta.exposure.nunique() != 186 or len(exposures) != 459:
        raise RuntimeError("Expected 186 Point 4 exposures expanded to 459 forecast features.")

    prediction_rows: list[dict] = []
    performance_rows: list[dict] = []
    inner_rows: list[dict] = []
    config_rows: list[dict] = []
    selected_rows: list[dict] = []
    shap_rows: list[pd.DataFrame] = []
    skipped_outer_years: list[dict[str, int]] = []
    for horizon in horizons:
        history = history_features(horizon, panel)
        validation_start = max(INNER_START_YEAR, min(test_years) - args.max_inner_folds)
        validation_years = list(range(validation_start, max(test_years)))
        inner_cv = precompute_inner_cv(panel, horizon, history, exposures, validation_years)
        inner_rows.extend(inner_cv.to_dict("records"))
        for outer_year in test_years:
            print(f"Horizon {horizon} month(s), outer forecast year {outer_year}...", flush=True)
            target_year = f"target_year_h{horizon}"
            outer_test = prepare(panel, horizon, panel[target_year].eq(outer_year), history)
            if outer_test.empty:
                skipped_outer_years.append({"horizon_months": horizon, "outer_test_year": outer_year})
                print("  skipped: no observed target-month outcome for this horizon/year.", flush=True)
                continue
            hist_id, _, hist_params, hist_years = choose_outer_config(
                inner_cv, outer_year, "history_hgb_nested", args.max_inner_folds
            )
            expo_id, winner_k, expo_params, expo_years = choose_outer_config(
                inner_cv, outer_year, "history_exposure_selected_hgb_nested", args.max_inner_folds
            )

            outer_train = prepare(panel, horizon, panel[target_year].lt(outer_year), history)
            outer_train["_target_loss"] = outer_train[f"target_loss_h{horizon}_pct"]
            outer_test["_target_loss"] = outer_test[f"target_loss_h{horizon}_pct"]
            selected = rank_exposures(outer_train, history, exposures)[:winner_k]
            hist_model, hist_prediction = fit_predict(outer_train, outer_test, history, hist_params)
            expo_model, expo_prediction = fit_predict(outer_train, outer_test, [*history, *selected], expo_params)
            prediction = outer_test[[
                "state_alpha", "region", "year", "month", "milk_cows_head", "milk_per_cow_lb",
                f"target_year_h{horizon}", f"target_month_h{horizon}",
                f"target_loss_h{horizon}_pct", f"target_milk_h{horizon}_lb_per_cow",
            ]].copy()
            prediction["horizon_months"] = horizon
            prediction["history_hgb_predicted_loss_pct"] = hist_prediction
            prediction["history_hgb_predicted_milk_lb_per_cow"] = loss_to_milk(prediction.milk_per_cow_lb, hist_prediction)
            prediction["history_exposure_selected_hgb_predicted_loss_pct"] = expo_prediction
            prediction["history_exposure_selected_hgb_predicted_milk_lb_per_cow"] = loss_to_milk(prediction.milk_per_cow_lb, expo_prediction)
            prediction_rows.extend(prediction.to_dict("records"))

            for model_name, predicted in (
                ("history_hgb_nested", hist_prediction),
                ("history_exposure_selected_hgb_nested", expo_prediction),
            ):
                scopes: list[tuple[str, pd.Series]] = [("overall", pd.Series(True, index=outer_test.index))]
                scopes.extend((f"region:{region}", outer_test.region.eq(region)) for region in sorted(outer_test.region.unique()))
                scopes.extend((f"state:{state}", outer_test.state_alpha.eq(state)) for state in sorted(outer_test.state_alpha.unique()))
                for scope, mask in scopes:
                    data = outer_test.loc[mask]
                    if data.empty:
                        continue
                    positions = np.flatnonzero(mask.to_numpy())
                    performance_rows.append({
                        "horizon_months": horizon, "test_year": outer_year, "model": model_name,
                        "scope": scope, "n": len(data), **score(data, predicted[positions], horizon),
                    })

            config_rows.extend([
                {"horizon_months": horizon, "test_year": outer_year, "model": "history_hgb_nested", "candidate_id": hist_id, "k_exposure_features": 0, "params": json.dumps(hist_params, sort_keys=True), "inner_validation_years": ",".join(map(str, hist_years)), "inner_selection_metric": "mean cow-weighted target-month milk RMSE; MAE tie-breaker"},
                {"horizon_months": horizon, "test_year": outer_year, "model": "history_exposure_selected_hgb_nested", "candidate_id": expo_id, "k_exposure_features": winner_k, "params": json.dumps(expo_params, sort_keys=True), "inner_validation_years": ",".join(map(str, expo_years)), "inner_selection_metric": "mean cow-weighted target-month milk RMSE; MAE tie-breaker"},
            ])
            selected_meta = exposure_meta.loc[exposure_meta.feature.isin(selected)].copy()
            selected_meta["horizon_months"] = horizon
            selected_meta["outer_test_year"] = outer_year
            selected_meta["k_exposure_features"] = winner_k
            selected_rows.extend(selected_meta.to_dict("records"))
            if horizon == 1 and not args.skip_shap:
                shap_rows.append(state_feature_shap(expo_model, outer_test, [*history, *selected], exposure_meta, outer_year))

    performance = pd.DataFrame(performance_rows)
    overall = performance.loc[performance.scope.eq("overall")].groupby(["horizon_months", "model"], as_index=False).agg(
        target_milk_rmse_lb_per_cow_mean=("target_milk_rmse_lb_per_cow", "mean"),
        target_milk_mae_lb_per_cow_mean=("target_milk_mae_lb_per_cow", "mean"),
        loss_rmse_pct_points_mean=("loss_rmse_pct_points", "mean"),
        loss_mae_pct_points_mean=("loss_mae_pct_points", "mean"),
        n_outer_years=("test_year", "nunique"), n_test_state_months=("n", "sum"),
    )
    baseline = overall.loc[overall.model.eq("history_hgb_nested"), ["horizon_months", "target_milk_rmse_lb_per_cow_mean", "target_milk_mae_lb_per_cow_mean"]].rename(columns={
        "target_milk_rmse_lb_per_cow_mean": "history_target_milk_rmse_lb_per_cow_mean",
        "target_milk_mae_lb_per_cow_mean": "history_target_milk_mae_lb_per_cow_mean",
    })
    overall = overall.merge(baseline, on="horizon_months", how="left", validate="many_to_one")
    overall["target_milk_rmse_reduction_vs_history_pct"] = 100 * (
        overall.history_target_milk_rmse_lb_per_cow_mean - overall.target_milk_rmse_lb_per_cow_mean
    ) / overall.history_target_milk_rmse_lb_per_cow_mean
    overall["target_milk_mae_reduction_vs_history_pct"] = 100 * (
        overall.history_target_milk_mae_lb_per_cow_mean - overall.target_milk_mae_lb_per_cow_mean
    ) / overall.history_target_milk_mae_lb_per_cow_mean

    pd.DataFrame(prediction_rows).to_csv(TAB / "point3_point4aligned_multihorizon_predictions.csv", index=False)
    performance.to_csv(TAB / "point3_point4aligned_multihorizon_performance_by_year_scope.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(TAB / "point3_point4aligned_multihorizon_inner_cv_results.csv", index=False)
    pd.DataFrame(config_rows).to_csv(TAB / "point3_point4aligned_multihorizon_configs_by_year.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(TAB / "point3_point4aligned_multihorizon_selected_features_by_year.csv", index=False)
    overall.to_csv(TAB / "point3_point4aligned_multihorizon_overall_summary.csv", index=False)
    if shap_rows:
        shap_by_year = pd.concat(shap_rows, ignore_index=True)
        shap_by_year.to_csv(TAB / "point3_point4aligned_h1_shap_state_feature_by_year.csv", index=False)
        shap_by_year.groupby([
            "state_alpha", "region", "feature", "exposure", "source_domain", "class_label", "subclass_label", "mechanistic_subclass_short", "temporal_transform"
        ], as_index=False).agg(
            mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"), n_years=("test_year", "nunique")
        ).to_csv(TAB / "point3_point4aligned_h1_shap_state_feature_summary.csv", index=False)
    make_figure(performance, show_legend=True)
    make_figure(performance, show_legend=False)
    manifest = {
        "method": "Separate nested rolling-origin continuous HGB forecasts for 1-12 month horizons, with training-only history-adjusted residual screening of Point 4-aligned exposure features.",
        "horizons_months": list(horizons), "outer_test_years": list(test_years),
        "raw_exposures": 186, "derived_exposure_features": 459, "k_grid": list(K_GRID),
        "selection_metric": "inner rolling mean cow-weighted target-month milk RMSE; MAE tie-breaker",
        "max_recent_inner_folds": args.max_inner_folds,
        "outer_train_rule": "target year strictly before the outer forecast year; test records have target_year equal to the held-out year",
        "skipped_outer_years_without_observed_target": skipped_outer_years,
        "shap": "Held-out H1 selected-HGB feature SHAP only; positive values increase predicted one-month milk loss.",
    }
    (LOG / "point3_point4aligned_multihorizon_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
