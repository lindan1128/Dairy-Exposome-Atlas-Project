#!/usr/bin/env python3
"""Build unweighted Daymet analogs for Point 1 weighted-vs-unweighted sensitivity.

The clean Point 1 exposome pool intentionally keeps the dairy-weighted Daymet
constructs.  For sensitivity, we need the same daily formulas aggregated to
state-month without dairy-cow county weights.  This script reconstructs the
heat/cold forms from county daily Daymet and then averages county-month values
equally within each state-month.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
POINT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "us_expose" / "raw" / "daymet_county_points"
OUT = POINT / "tables" / "point1_daymet_unweighted_state_month_forms.csv"

sys.path.insert(0, str(ROOT / "src"))
from dairy_exposome_atlas import heatstress as H  # noqa: E402

KEY = ["state_alpha", "year", "month"]


def svp_pa(t_c: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 610.94 * np.exp((17.625 * t_c) / (t_c + 243.04))


def dewpoint_c(vp_pa: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    g = np.log(np.clip(vp_pa, 1e-6, None) / 610.94)
    return 243.04 * g / (17.625 - g)


def max_run(flags: np.ndarray) -> int:
    mx = cur = 0
    for v in flags:
        cur = cur + 1 if bool(v) else 0
        mx = max(mx, cur)
    return int(mx)


def county_month_forms(path: str) -> pd.DataFrame:
    use = [
        "year", "date", "tmax_c", "tmin_c", "vp_pa", "prcp_mm", "swe_mm",
        "state_alpha", "county_fips",
    ]
    d = pd.read_csv(path, usecols=use)
    d = d[d["year"].between(2000, 2025)].copy()
    if d.empty:
        return pd.DataFrame()

    d = d.sort_values("date")
    d["month"] = pd.to_datetime(d["date"]).dt.month
    tmean = (d["tmax_c"] + d["tmin_c"]) / 2.0
    rh = H.relative_humidity_percent(tmean, d["vp_pa"])
    thi = H.dairy_thi(tmean, rh)
    wb = H.wet_bulb_stull_c(tmean, rh)
    vpd = ((svp_pa(tmean) - d["vp_pa"]) / 1000.0).clip(lower=0)
    td = dewpoint_c(d["vp_pa"])
    drange = d["tmax_c"] - d["tmin_c"]

    hot72 = (thi >= 72).to_numpy()
    warmnight20 = (d["tmin_c"] >= 20).to_numpy()
    cold45 = (thi < 45).to_numpy()
    cold39 = (thi < 39).to_numpy()
    wet = (d["prcp_mm"] >= 1.0).to_numpy()
    dry = ((d["prcp_mm"] < 1.0) & (d["swe_mm"] < 5.0)).to_numpy()
    precip_dry = (d["prcp_mm"] < 1.0).to_numpy()
    snow = (d["swe_mm"] >= 5.0).to_numpy()
    nosnow = (d["swe_mm"] < 5.0).to_numpy()
    ice = (d["tmax_c"] <= 0.0).to_numpy()
    hard10 = (d["tmin_c"] <= -10.0).to_numpy()

    d = d.assign(
        tmean_c=tmean,
        rh_mean_pct=rh,
        thi=thi,
        wetbulb_c=wb,
        vpd_kpa_calc=vpd,
        dewpoint_c=td,
        dewpoint_depression_c=tmean - td,
        diurnal_range_c=drange,
        i_thi68=(thi >= 68).astype(int),
        i_thi72=hot72.astype(int),
        i_thi79=(thi >= 79).astype(int),
        i_thi50=(thi < 50).astype(int),
        i_thi45=cold45.astype(int),
        i_thi39=cold39.astype(int),
        i_wb22=(wb >= 22).astype(int),
        i_wb24=(wb >= 24).astype(int),
        i_wb26=(wb >= 26).astype(int),
        i_vpd2=(vpd >= 2.0).astype(int),
        i_vpd3=(vpd >= 3.0).astype(int),
        i_vpd4=(vpd >= 4.0).astype(int),
        i_swe25=(d["swe_mm"] >= 25.0).astype(int),
        i_dry30=((d["tmax_c"] >= 30) & (rh < 50)).astype(int),
        i_dry32=((d["tmax_c"] >= 32) & (rh < 40)).astype(int),
        i_t30=(d["tmax_c"] >= 30).astype(int),
        i_t32=(d["tmax_c"] >= 32).astype(int),
        i_t35=(d["tmax_c"] >= 35).astype(int),
        i_n18=(d["tmin_c"] >= 18).astype(int),
        i_n20=warmnight20.astype(int),
        i_n22=(d["tmin_c"] >= 22).astype(int),
        i_lowcool=(drange < 8).astype(int),
        thi_excess72=(thi - 72.0).clip(lower=0),
        wb_excess22=(wb - 22.0).clip(lower=0),
        vpd_excess2=(vpd - 2.0).clip(lower=0),
        i_humidhot=(hot72 & (wb >= 24).to_numpy()).astype(int),
        i_dryhot=(hot72 & (wb < 22).to_numpy()).astype(int),
        i_hotnorelief=(hot72 & warmnight20).astype(int),
        cold_excess50=(50.0 - thi).clip(lower=0),
        i_wetcold=(cold45 & wet).astype(int),
        i_wetcold50=((thi <= 50).to_numpy() & wet).astype(int),
        i_wetcold39=((thi <= 39).to_numpy() & wet).astype(int),
        i_snowcover=snow.astype(int),
        i_snowcold45=((thi <= 45).to_numpy() & snow).astype(int),
        i_drycold=(cold45 & dry).astype(int),
        i_sevdrycold=(cold39 & dry).astype(int),
        i_drycold50=((thi <= 50).to_numpy() & precip_dry).astype(int),
        i_drycold45=((thi <= 45).to_numpy() & precip_dry).astype(int),
        i_drycold39=((thi <= 39).to_numpy() & precip_dry).astype(int),
        i_nosnowcold45=((thi <= 45).to_numpy() & nosnow).astype(int),
        wetload45=np.where(wet, (45.0 - thi).clip(lower=0), 0.0),
        dryload45=np.where(precip_dry, (45.0 - thi).clip(lower=0), 0.0),
        thi_drycold=np.where(((thi <= 45).to_numpy() & precip_dry), thi, np.nan),
        i_coldnight0=(d["tmin_c"] <= 0.0).astype(int),
        i_hardfreeze10=hard10.astype(int),
        i_hardfreeze15=(d["tmin_c"] <= -15.0).astype(int),
        i_noreliefcold=((thi <= 45).to_numpy() & (d["tmax_c"] <= 0.0).to_numpy()).astype(int),
        i_lowthaw=((d["tmax_c"] <= 0.0) & (drange < 8.0)).astype(int),
        i_ice=ice.astype(int),
    )

    keys = ["state_alpha", "county_fips", "year", "month"]
    agg = d.groupby(keys).agg(
        tmax_c=("tmax_c", "mean"),
        tmin_c=("tmin_c", "mean"),
        tmean_c=("tmean_c", "mean"),
        rh_mean_pct=("rh_mean_pct", "mean"),
        thi_mean=("thi", "mean"),
        thi_max=("thi", "max"),
        thi_min=("thi", "min"),
        thi_days_ge_68=("i_thi68", "sum"),
        thi_days_ge_72=("i_thi72", "sum"),
        thi_days_ge_79=("i_thi79", "sum"),
        thi_days_lt_50=("i_thi50", "sum"),
        thi_days_lt_45=("i_thi45", "sum"),
        thi_days_lt_39=("i_thi39", "sum"),
        wetbulb_mean_c=("wetbulb_c", "mean"),
        wetbulb_max_c=("wetbulb_c", "max"),
        wetbulb_days_ge_22c=("i_wb22", "sum"),
        wetbulb_days_ge_24c=("i_wb24", "sum"),
        wetbulb_days_ge_26c=("i_wb26", "sum"),
        swe_mm=("swe_mm", "mean"),
        swe_max_mm=("swe_mm", "max"),
        swe_days_ge_25mm=("i_swe25", "sum"),
        vpd_kpa=("vpd_kpa_calc", "mean"),
        vpd_max=("vpd_kpa_calc", "max"),
        vpd_days_ge_2kpa=("i_vpd2", "sum"),
        vpd_days_ge_3kpa=("i_vpd3", "sum"),
        vpd_days_ge_4kpa=("i_vpd4", "sum"),
        vpd_heatload_ge2=("vpd_excess2", "sum"),
        dewpoint_mean_c=("dewpoint_c", "mean"),
        dewpoint_depression_mean=("dewpoint_depression_c", "mean"),
        diurnal_range_c=("diurnal_range_c", "mean"),
        dry_heat_days_t30_rh50=("i_dry30", "sum"),
        dry_heat_days_t32_rh40=("i_dry32", "sum"),
        tmax_days_ge_30c=("i_t30", "sum"),
        tmax_days_ge_32c=("i_t32", "sum"),
        tmax_days_ge_35c=("i_t35", "sum"),
        tmin_mean_c=("tmin_c", "mean"),
        tmin_max_c=("tmin_c", "max"),
        tmin_min_c=("tmin_c", "min"),
        warm_nights_tmin_ge_18c=("i_n18", "sum"),
        warm_nights_tmin_ge_20c=("i_n20", "sum"),
        warm_nights_tmin_ge_22c=("i_n22", "sum"),
        low_cooling_days=("i_lowcool", "sum"),
        thi_heatload_ge72=("thi_excess72", "sum"),
        humid_hot_days_t72wb24=("i_humidhot", "sum"),
        dry_hot_days_t72wb_lt22=("i_dryhot", "sum"),
        hot_no_relief_days_t72n20=("i_hotnorelief", "sum"),
        wetbulb_heatload_ge22=("wb_excess22", "sum"),
        cold_load_lt50=("cold_excess50", "sum"),
        wet_cold_days=("i_wetcold", "sum"),
        wet_cold_days_lt50=("i_wetcold50", "sum"),
        wet_cold_days_lt39=("i_wetcold39", "sum"),
        snow_cover_days=("i_snowcover", "sum"),
        snow_cold_days_lt45=("i_snowcold45", "sum"),
        dry_cold_days=("i_drycold", "sum"),
        severe_dry_cold_days=("i_sevdrycold", "sum"),
        dry_cold_days_lt50=("i_drycold50", "sum"),
        dry_cold_days_lt45=("i_drycold45", "sum"),
        dry_cold_days_lt39=("i_drycold39", "sum"),
        nosnow_cold_days_lt45=("i_nosnowcold45", "sum"),
        wet_cold_load_lt45=("wetload45", "sum"),
        dry_cold_load_lt45=("dryload45", "sum"),
        dry_cold_thi_mean=("thi_drycold", "mean"),
        dry_cold_thi_min=("thi_drycold", "min"),
        cold_nights_tmin_le0=("i_coldnight0", "sum"),
        hard_freeze_le10=("i_hardfreeze10", "sum"),
        hard_freeze_nights=("i_hardfreeze15", "sum"),
        cold_no_relief_t45tmax0=("i_noreliefcold", "sum"),
        low_thaw_days=("i_lowthaw", "sum"),
        ice_days=("i_ice", "sum"),
    )
    runs = d.groupby(keys).agg(
        consec_thi72_maxrun=("i_thi72", lambda s: max_run(s.to_numpy())),
        consec_warmnight20_maxrun=("i_n20", lambda s: max_run(s.to_numpy())),
        consec_thi_lt45_maxrun=("i_thi45", lambda s: max_run(s.to_numpy())),
        consec_snowcover_maxrun=("i_snowcover", lambda s: max_run(s.to_numpy())),
        consec_ice_days_maxrun=("i_ice", lambda s: max_run(s.to_numpy())),
        consec_hardfreeze10_maxrun=("i_hardfreeze10", lambda s: max_run(s.to_numpy())),
    )
    return agg.join(runs).reset_index()


def main() -> int:
    files = sorted(glob.glob(str(RAW / "*.csv")))
    rows = []
    for i, path in enumerate(files, 1):
        try:
            cm = county_month_forms(path)
        except Exception as exc:
            print(f"skip {Path(path).name}: {exc}", flush=True)
            continue
        if not cm.empty:
            rows.append(cm)
        if i % 250 == 0:
            print(f"processed {i}/{len(files)} county files", flush=True)
    if not rows:
        raise RuntimeError(f"No Daymet county forms built from {RAW}")

    county = pd.concat(rows, ignore_index=True)
    value_cols = [c for c in county.columns if c not in KEY + ["county_fips"]]
    state = (
        county.groupby(KEY, as_index=False)[value_cols]
        .mean(numeric_only=True)
        .sort_values(KEY)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"wrote {OUT} rows={len(state):,} cols={len(state.columns)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
