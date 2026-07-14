#!/usr/bin/env python3
"""Build Point 5 state-month feature panel for regional milk-loss risk modeling."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
TAB5 = ROOT / "analysis/statistics/5/tables"
POINT4_UNION = ROOT / "analysis/statistics/4/tables/point4_anchor_year_variable_network_yearly_pruned_union_dictionary.csv"
MILK = ROOT / "data/us_milk/tables/analysis_1_2_4_state_month_milk_phenotypes.csv"
EXPO = ROOT / "data/us_expose_new/processed/exposure_state_month_expanded.csv"

FORECAST_MECHANISTIC_DOMAINS = {
    "Heat: dry (VPD/aridity)",
    "Heat: humid (wet-bulb)",
    "Heat: nighttime recovery",
    "Cold: dry",
    "Cold: nighttime / no-thaw",
    "Cold: wet / snowy",
    "Storm / disaster events",
    "Agricultural pesticide-use burden",
    "Forage condition",
    "Feed market",
    "Population-demand and market-size context",
    "Milk price, dairy market and milk-feed price ratio",
}

PRIMARY_STATES = [
    "AZ","CA","CO","ID","NM","OR","UT","WA",
    "IA","IL","IN","KS","MI","MN","MO","OH","SD","WI",
    "NY","PA","VT","FL","GA","KY","TX","VA",
]

REGION = {
    "AZ":"West","CA":"West","CO":"West","ID":"West","NM":"West","OR":"West","UT":"West","WA":"West",
    "IA":"Midwest","IL":"Midwest","IN":"Midwest","KS":"Midwest","MI":"Midwest","MN":"Midwest","MO":"Midwest","OH":"Midwest","SD":"Midwest","WI":"Midwest",
    "NY":"Northeast","PA":"Northeast","VT":"Northeast",
    "FL":"South","GA":"South","KY":"South","TX":"South","VA":"South",
}

def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean(skipna=True)) / sd

def add_temporal_features(panel: pd.DataFrame, exp: str, source_class: str) -> dict[str, pd.Series]:
    x = pd.to_numeric(panel[exp], errors="coerce").astype(float)
    by_state = x.groupby(panel["state_alpha"], sort=False)
    features = {f"current__{exp}": x}

    if str(source_class) == "Nature and climate":
        features[f"roll3__{exp}"] = by_state.transform(lambda s: s.rolling(3, min_periods=2).mean())
        features[f"delta12__{exp}"] = x - by_state.shift(12)
    else:
        features[f"yoy_change__{exp}"] = x - by_state.shift(12)
    return features

def main() -> int:
    TAB5.mkdir(parents=True, exist_ok=True)
    dic = pd.read_csv(POINT4_UNION)
    dic = dic[dic["mechanistic_domain_en"].isin(FORECAST_MECHANISTIC_DOMAINS)].copy()
    if "source_class" not in dic.columns and "class" in dic.columns:
        dic["source_class"] = dic["class"]
    if "domain_label" not in dic.columns and "domain" in dic.columns:
        dic["domain_label"] = dic["domain"]
    if "subdomain_label" not in dic.columns and "Subdomain" in dic.columns:
        dic["subdomain_label"] = dic["Subdomain"]
    dic = dic.rename(
        columns={
        }
    )
    dic = dic.dropna(subset=["exposure"]).drop_duplicates("exposure").reset_index(drop=True)
    exposures = [x for x in dic["exposure"].dropna().astype(str).unique()]
    milk_cols = ["state_alpha","year","month","milk_per_cow_lb","milk_cows_head"]
    milk = pd.read_csv(MILK, usecols=milk_cols, low_memory=False)
    milk = milk[milk["state_alpha"].isin(PRIMARY_STATES)].copy()
    milk = milk[(milk["year"] >= 2000) & (milk["year"] <= 2025)].copy()
    expo_cols = ["state_alpha","year","month"] + [c for c in exposures if c in pd.read_csv(EXPO, nrows=0).columns]
    expo = pd.read_csv(EXPO, usecols=expo_cols, low_memory=False)
    panel = milk.merge(expo, on=["state_alpha","year","month"], how="left")
    panel = panel.sort_values(["state_alpha","year","month"]).reset_index(drop=True)
    panel["region"] = panel["state_alpha"].map(REGION)
    panel["date_index"] = panel["year"] * 12 + panel["month"]
    panel["log_milk_per_cow"] = np.log(panel["milk_per_cow_lb"].clip(lower=1))
    panel["log_milk_per_cow_lag1"] = panel.groupby("state_alpha")["log_milk_per_cow"].shift(1)
    panel["next_log_milk_per_cow"] = panel.groupby("state_alpha")["log_milk_per_cow"].shift(-1)
    panel["next_year"] = panel.groupby("state_alpha")["year"].shift(-1)
    panel["next_month"] = panel.groupby("state_alpha")["month"].shift(-1)
    panel["next_loss_pct"] = -100 * (panel["next_log_milk_per_cow"] - panel["log_milk_per_cow"])

    feature_rows = []
    for _, r in dic.iterrows():
        exp = str(r["exposure"])
        if exp not in panel.columns:
            continue
        temporal = add_temporal_features(panel, exp, r.get("source_class"))
        for raw_name, values in temporal.items():
            zcol = f"x__{raw_name}"
            panel[zcol] = zscore(values)
            transform = raw_name.split("__", 1)[0]
            feature_rows.append({
                "feature": zcol,
                "exposure": exp,
                "feature_group": "exposure_node",
                "temporal_transform": transform,
                "source_class": r.get("source_class"),
                "domain_label": r.get("domain_label"),
                "subdomain_label": r.get("subdomain_label"),
                "mechanistic_domain_en": r.get("mechanistic_domain_en"),
                "mechanistic_domain_ch": r.get("mechanistic_domain_ch"),
                "definition_en": r.get("definition_en"),
                "definition_ch": r.get("definition_ch"),
                "data_source": r.get("data_source"),
                "source_url": r.get("source_url"),
            })
    meta = pd.DataFrame(feature_rows)
    meta.to_csv(TAB5 / "point5_forecast_feature_dictionary.csv", index=False)
    dic.to_csv(TAB5 / "point5_forecast_exposure_dictionary.csv", index=False)
    dic.to_csv(TAB5 / "point5_forecast_nonredundant_exposure_dictionary.csv", index=False)
    keep = ["state_alpha","region","year","month","next_year","next_month","milk_cows_head","milk_per_cow_lb","log_milk_per_cow","log_milk_per_cow_lag1","next_loss_pct"] + meta["feature"].tolist()
    panel[keep].to_csv(TAB5 / "point5_forecast_state_month_feature_panel.csv", index=False)
    print(f"Wrote {dic['exposure'].nunique()} forecast exposures, {len(meta)} features and {len(panel)} state-month rows")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
