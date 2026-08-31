#!/usr/bin/env python3
"""Generate held-out RF feature SHAP for one completed Point 3 forecast horizon.

The script reuses the already recorded outer-fold configuration and selected
features. It refits only the corresponding legal outer training set, then
computes SHAP exclusively on that outer year's held-out state-months.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge



HERE = Path(__file__).resolve().parents[1]
TAB = HERE / "tables"
RANDOM_STATE = 20260822
PLOT_DOMAINS = (
    "Heat",
    "Cold",
    "Severe weather",
    "Forage",
    "Pesticides",
    "Feed market",
    "Dairy market",
    "Market demand",
)


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


def rank_exposures(train: pd.DataFrame, history: list[str], exposures: list[str]) -> list[str]:
    """Exact training-only residual screen used by the forecast script."""
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


def shap_values(
    model: RandomForestRegressor, imputer: SimpleImputer, test: pd.DataFrame, features: list[str]
) -> np.ndarray:
    values = np.asarray(shap.TreeExplainer(model).shap_values(imputer.transform(test[features])))
    return values[:, :, 0] if values.ndim == 3 else values


def state_feature_shap(
    model: RandomForestRegressor,
    imputer: SimpleImputer,
    test: pd.DataFrame,
    features: list[str],
    exposure_meta: pd.DataFrame,
    test_year: int,
) -> pd.DataFrame:
    values = shap_values(model, imputer, test, features)
    exposure_features = [feature for feature in features if feature in set(exposure_meta.feature)]
    shap_df = pd.DataFrame(values, columns=features, index=test.index)
    work = shap_df[exposure_features].copy()
    work["state_alpha"] = test.state_alpha.to_numpy()
    work["region"] = test.region.to_numpy()
    parts = []
    for (state, region), values_by_state in work.groupby(["state_alpha", "region"], sort=False):
        values_only = values_by_state[exposure_features]
        parts.append(pd.DataFrame({
            "test_year": test_year,
            "state_alpha": state,
            "region": region,
            "feature": exposure_features,
            "mean_abs_shap": np.abs(values_only).mean(axis=0).to_numpy(),
            "mean_signed_shap": values_only.mean(axis=0).to_numpy(),
            "n_state_months": len(values_by_state),
        }))
    return pd.concat(parts, ignore_index=True).merge(exposure_meta, on="feature", how="left", validate="many_to_one")


def state_month_class_shap(
    model: RandomForestRegressor,
    imputer: SimpleImputer,
    test: pd.DataFrame,
    features: list[str],
    exposure_meta: pd.DataFrame,
    test_year: int,
    horizon: int,
) -> pd.DataFrame:
    """Group exact held-out SHAP values by the eight pre-specified exposome domains.

    Every selected feature, including each selected temporal transformation, is
    summed within its domain before any sign or sample-level summary is taken.
    This preserves SHAP additivity for the fitted model. A domain absent from an
    outer model receives zero contribution for that outer held-out prediction.
    """
    values = shap_values(model, imputer, test, features)
    feature_to_domain = exposure_meta.drop_duplicates("feature").set_index("feature").class_label.to_dict()
    domain_features = {
        domain: [feature for feature in features if feature_to_domain.get(feature) == domain]
        for domain in PLOT_DOMAINS
    }
    shap_df = pd.DataFrame(values, columns=features, index=test.index)
    identity = test[["state_alpha", "region", "year", "month"]].copy()
    identity.insert(0, "horizon_months", horizon)
    identity.insert(0, "test_year", test_year)
    parts = []
    for domain, domain_columns in domain_features.items():
        signed = (
            shap_df[domain_columns].sum(axis=1)
            if domain_columns
            else pd.Series(0.0, index=test.index)
        )
        part = identity.copy()
        part["class_label"] = domain
        part["class_signed_shap"] = signed.to_numpy(float)
        part["class_abs_signed_shap"] = np.abs(signed.to_numpy(float))
        part["class_positive_shap"] = np.clip(signed.to_numpy(float), 0.0, None)
        part["class_negative_shap"] = np.clip(-signed.to_numpy(float), 0.0, None)
        part["n_selected_features_in_class"] = len(domain_columns)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def state_month_feature_shap(
    model: RandomForestRegressor,
    imputer: SimpleImputer,
    test: pd.DataFrame,
    features: list[str],
    exposure_meta: pd.DataFrame,
    test_year: int,
    horizon: int,
) -> pd.DataFrame:
    """Return exact feature-level SHAP and feature values for held-out samples."""
    values = shap_values(model, imputer, test, features)
    meta = exposure_meta.loc[exposure_meta.class_label.isin(PLOT_DOMAINS)].drop_duplicates("feature")
    feature_to_index = {feature: index for index, feature in enumerate(features)}
    exposure_features = [feature for feature in meta.feature if feature in feature_to_index]
    feature_indices = [feature_to_index[feature] for feature in exposure_features]
    n_samples, n_features = len(test), len(exposure_features)
    identity = test[["state_alpha", "region", "year", "month"]].reset_index(drop=True)
    test_x = imputer.transform(test[features])
    out = pd.DataFrame({
        "test_year": np.repeat(test_year, n_samples * n_features),
        "horizon_months": np.repeat(horizon, n_samples * n_features),
        "state_alpha": np.repeat(identity.state_alpha.to_numpy(), n_features),
        "region": np.repeat(identity.region.to_numpy(), n_features),
        "year": np.repeat(identity.year.to_numpy(), n_features),
        "month": np.repeat(identity.month.to_numpy(), n_features),
        "feature": np.tile(exposure_features, n_samples),
        "feature_value": test_x[:, feature_indices].ravel(),
        "shap_value": values[:, feature_indices].ravel(),
    })
    return out.merge(meta, on="feature", how="left", validate="many_to_one")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--write-state-month-feature", action="store_true")
    args = parser.parse_args()
    horizon = args.horizon
    if horizon not in range(1, 13):
        raise ValueError("--horizon must be an integer from 1 to 12.")

    panel = pd.read_csv(TAB / "point3_point4aligned_multihorizon_model_panel.csv", low_memory=False)
    exposure_meta = pd.read_csv(TAB / "point3_point4aligned_multihorizon_exposure_dictionary.csv")
    configs = pd.read_csv(TAB / "supp_point3_rf_multihorizon_configs_by_year.csv")
    selected = pd.read_csv(TAB / "supp_point3_rf_multihorizon_selected_features_by_year.csv")
    stored_predictions = pd.read_csv(TAB / "supp_point3_rf_multihorizon_predictions.csv")
    configs = configs.loc[
        configs.horizon_months.eq(horizon)
        & configs.model.eq("history_exposure_selected_rf_nested")
    ].sort_values("test_year")
    if configs.empty:
        raise RuntimeError(f"No completed exposure-model configurations found for horizon {horizon}.")

    history = history_features(horizon, panel)
    exposures = exposure_meta.feature.tolist()
    target_year = f"target_year_h{horizon}"
    rows = []
    domain_rows = []
    state_month_feature_rows = []
    prediction_checks = []
    for config in configs.itertuples(index=False):
        test_year = int(config.test_year)
        stored_features = selected.loc[
            selected.horizon_months.eq(horizon) & selected.outer_test_year.eq(test_year), "feature"
        ].tolist()
        if len(stored_features) != int(config.k_exposure_features):
            raise RuntimeError(f"Selected-feature record does not match the stored h{horizon}, {test_year} configuration.")
        train = prepare(panel, horizon, panel[target_year].lt(test_year), history)
        test = prepare(panel, horizon, panel[target_year].eq(test_year), history)
        if test.empty:
            raise RuntimeError(f"No held-out h{horizon} test records for {test_year}; it should not have a stored configuration.")
        train["_target_loss"] = train[f"target_loss_h{horizon}_pct"]
        features = rank_exposures(train, history, exposures)[:int(config.k_exposure_features)]
        if set(features) != set(stored_features):
            raise RuntimeError(f"Reconstructed selection set does not match stored h{horizon}, {test_year} features.")
        params = json.loads(config.params)
        all_features = [*history, *features]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        train_x = imputer.fit_transform(train[all_features])
        test_x = imputer.transform(test[all_features])
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE)
        model.fit(train_x, train._target_loss, sample_weight=train.milk_cows_head)
        predicted_loss = model.predict(test_x)
        keys = ["state_alpha", "year", "month"]
        reference = stored_predictions.loc[
            stored_predictions.horizon_months.eq(horizon) & stored_predictions[f"target_year_h{horizon}"].eq(test_year),
            [*keys, "history_exposure_selected_rf_predicted_loss_pct"],
        ]
        checked = test[keys].copy()
        checked["recomputed_loss"] = predicted_loss
        checked = checked.merge(reference, on=keys, how="left", validate="one_to_one")
        if checked.history_exposure_selected_rf_predicted_loss_pct.isna().any():
            raise RuntimeError(f"Missing stored prediction for h{horizon}, {test_year}.")
        max_abs_difference = float(np.max(np.abs(
            checked.recomputed_loss - checked.history_exposure_selected_rf_predicted_loss_pct
        )))
        if not np.allclose(
            checked.recomputed_loss,
            checked.history_exposure_selected_rf_predicted_loss_pct,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Recomputed prediction does not match stored h{horizon}, {test_year}; "
                f"maximum absolute difference={max_abs_difference:.3g}."
            )
        prediction_checks.append({
            "horizon_months": horizon, "test_year": test_year, "n_test_state_months": len(checked),
            "max_abs_loss_prediction_difference": max_abs_difference,
        })
        rows.append(state_feature_shap(model, imputer, test, all_features, exposure_meta, test_year))
        domain_rows.append(state_month_class_shap(model, imputer, test, all_features, exposure_meta, test_year, horizon))
        if args.write_state_month_feature:
            state_month_feature_rows.append(
                state_month_feature_shap(model, imputer, test, all_features, exposure_meta, test_year, horizon)
            )
        print(f"computed held-out RF H{horizon} SHAP for {test_year}", flush=True)

    raw = pd.concat(rows, ignore_index=True)
    pd.DataFrame(prediction_checks).to_csv(
        TAB / f"supp_point3_rf_h{horizon}_shap_prediction_reproducibility_check.csv", index=False
    )
    raw.to_csv(TAB / f"supp_point3_rf_h{horizon}_shap_state_feature_by_year.csv", index=False)
    pd.concat(domain_rows, ignore_index=True).to_csv(
        TAB / f"supp_point3_rf_h{horizon}_shap_state_month_class.csv", index=False
    )
    if args.write_state_month_feature:
        pd.concat(state_month_feature_rows, ignore_index=True).to_csv(
            TAB / f"supp_point3_rf_h{horizon}_shap_state_month_feature.csv", index=False
        )
    raw.groupby(
        ["state_alpha", "region", "feature", "exposure", "source_domain", "class_label", "subclass_label", "mechanistic_subclass_short", "temporal_transform"],
        as_index=False,
    ).agg(
        mean_abs_shap=("mean_abs_shap", "mean"),
        mean_signed_shap=("mean_signed_shap", "mean"),
        n_years=("test_year", "nunique"),
    ).to_csv(TAB / f"supp_point3_rf_h{horizon}_shap_state_feature_summary.csv", index=False)
    print(f"wrote RF H{horizon} held-out SHAP for {configs.test_year.nunique()} outer test years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
