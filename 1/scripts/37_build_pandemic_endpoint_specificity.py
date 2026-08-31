#!/usr/bin/env python3
"""Build COVID/HPAI endpoint-specificity tables for Point 1 supplement."""

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
US_EXPOSE_NEW = ROOT / "data" / "us_expose_new" / "processed"
LB_TO_KG = 0.45359237


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(2500, len(vals)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def fit_calendar_baseline(d: pd.DataFrame, outcome: str) -> pd.DataFrame:
    d = d[["state_alpha", "year", "month", "date", outcome]].dropna().copy()
    d = d[(d["year"] >= 2010) & (d["year"] <= 2025) & (d[outcome] > 0)].copy()
    d["time_index"] = (d["year"] - 2010) * 12 + d["month"]
    train = d[(d["year"] >= 2010) & (d["year"] <= 2019)].copy()
    y = np.log(train[outcome].to_numpy(float))
    tmean = train["time_index"].mean()
    tsd = train["time_index"].std(ddof=0)
    x_train = pd.concat(
        [
            pd.Series(1.0, index=train.index, name="intercept"),
            pd.get_dummies(train["state_alpha"], prefix="state", drop_first=True, dtype=float),
            pd.get_dummies(train["month"], prefix="month", drop_first=True, dtype=float),
            ((train["time_index"] - tmean) / tsd).rename("time_scaled"),
        ],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(x_train.to_numpy(), y, rcond=None)
    x_all = pd.concat(
        [
            pd.Series(1.0, index=d.index, name="intercept"),
            pd.get_dummies(d["state_alpha"], prefix="state", drop_first=True, dtype=float),
            pd.get_dummies(d["month"], prefix="month", drop_first=True, dtype=float),
            ((d["time_index"] - tmean) / tsd).rename("time_scaled"),
        ],
        axis=1,
    ).reindex(columns=x_train.columns, fill_value=0.0)
    d["expected"] = np.exp(x_all.to_numpy() @ coef)
    d["anomaly_pct"] = (d[outcome] / d["expected"] - 1.0) * 100.0
    return d


def build_covid() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(L.US_MILK / "state_month_panel.csv", low_memory=False)
    panel["date"] = pd.to_datetime(
        panel["year"].astype(int).astype(str) + "-" + panel["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * LB_TO_KG
    panel["milk_production_million_kg"] = panel["milk_production_lb"] * LB_TO_KG / 1e6
    rows = []
    state_rows = []
    outcomes = {
        "milk_per_cow_kg": "Milk per cow",
        "milk_production_million_kg": "Total production",
    }
    for outcome, label in outcomes.items():
        counts = (
            panel[(panel["year"] >= 2010) & (panel["year"] <= 2025)]
            .dropna(subset=[outcome])
            .groupby("state_alpha")
            .size()
        )
        states = sorted(counts[counts >= 12 * 12].index.tolist())
        fit = fit_calendar_baseline(panel[panel["state_alpha"].isin(states)], outcome)
        fit["outcome"] = outcome
        fit["outcome_label"] = label
        state_rows.append(fit)
        for date, g in fit.groupby("date"):
            lo, hi = bootstrap_ci(g["anomaly_pct"].to_numpy(), seed=int(date.strftime("%Y%m")))
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": label,
                    "date": date,
                    "year": date.year,
                    "month": date.month,
                    "mean_anomaly_pct": g["anomaly_pct"].mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_states": g["state_alpha"].nunique(),
                }
            )
    state_df = pd.concat(state_rows, ignore_index=True)
    profile = pd.DataFrame(rows).sort_values(["outcome", "date"])
    return state_df, profile


def build_hpai() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(L.US_MILK / "state_month_panel.csv", low_memory=False)
    if "hpai_dairy_cases" not in panel.columns:
        exp = pd.read_csv(US_EXPOSE_NEW / "exposure_state_month_expanded.csv", low_memory=False)
        panel = panel.merge(exp[["state_alpha", "year", "month", "hpai_dairy_cases"]], on=["state_alpha", "year", "month"], how="left")
    panel["date"] = pd.to_datetime(
        panel["year"].astype(int).astype(str) + "-" + panel["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * LB_TO_KG
    panel["milk_production_million_kg"] = panel["milk_production_lb"] * LB_TO_KG / 1e6
    labels = {
        "milk_per_cow_kg": "Milk per cow",
        "milk_production_million_kg": "Total production",
    }
    frames = []
    first = panel[panel["hpai_dairy_cases"] > 0].groupby("state_alpha")["date"].min()
    for outcome in labels:
        modern = panel[panel["year"] >= 2022].dropna(subset=[outcome]).copy()
        fe = L.fixed_effects(modern, spec="sm_yeartrend").to_numpy()
        ylog = np.log(np.clip(modern[outcome].to_numpy(float), 1e-12, None))
        modern["effect_pct"] = L.absorb(ylog.reshape(-1, 1), fe).ravel() * 100.0
        rec = []
        for st, t0 in first.items():
            g = modern[modern["state_alpha"] == st]
            for _, r in g.iterrows():
                tau = (r["date"].year - t0.year) * 12 + (r["date"].month - t0.month)
                if -6 <= tau <= 6:
                    rr = r[["state_alpha", "year", "month", "date", "effect_pct"]].to_dict()
                    rr.update({"tau": tau, "first_detection": t0, "outcome": outcome})
                    rec.append(rr)
        frames.append(pd.DataFrame(rec))
    ev = pd.concat(frames, ignore_index=True)
    pre = (
        ev[(ev["tau"] >= -6) & (ev["tau"] < 0)]
        .groupby(["outcome", "state_alpha"])["effect_pct"]
        .mean()
        .rename("pre_event_mean_pct")
        .reset_index()
    )
    ev = ev.merge(pre, on=["outcome", "state_alpha"], how="left")
    ev["event_centered_pct"] = ev["effect_pct"] - ev["pre_event_mean_pct"]
    rows = []
    for (outcome, tau), g in ev.groupby(["outcome", "tau"]):
        lo, hi = bootstrap_ci(g["event_centered_pct"].to_numpy(), seed=1000 + int(tau) + len(outcome))
        rows.append(
            {
                "outcome": outcome,
                "outcome_label": labels.get(outcome, outcome),
                "tau": tau,
                "mean_effect_pct": g["event_centered_pct"].mean(),
                "ci_low": lo,
                "ci_high": hi,
                "n_states": g["state_alpha"].nunique(),
                "negative_share": float((g["event_centered_pct"] < 0).mean()),
            }
        )
    return ev, pd.DataFrame(rows).sort_values(["outcome", "tau"])


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    covid_state, covid_profile = build_covid()
    hpai_state, hpai_profile = build_hpai()
    covid_state.to_csv(TAB / "point1_pandemic_covid_calendar_state_anomalies.csv", index=False)
    covid_profile.to_csv(TAB / "point1_pandemic_covid_calendar_profile.csv", index=False)
    hpai_state.to_csv(TAB / "point1_pandemic_hpai_event_state_anomalies.csv", index=False)
    hpai_profile.to_csv(TAB / "point1_pandemic_hpai_event_profile.csv", index=False)
    print(f"COVID profile rows={len(covid_profile)}, states={covid_state['state_alpha'].nunique()}")
    print(f"HPAI profile rows={len(hpai_profile)}, states={hpai_state['state_alpha'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
