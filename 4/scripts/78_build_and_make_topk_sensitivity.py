#!/usr/bin/env python3
"""Top-K sensitivity of the annual aggregate exposome incremental R2 (Fig 4b).

Fig 4b combines the top-5 representative exposures per year. Here we re-run the
same annual model for K = 2..10 under the herd-scale + available-breed baseline
to check that the temporal decline in aggregate explanatory R2 is not specific
to K = 5. Reuses the model/selection functions from script 76.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STAT4 = SCRIPT_DIR.parents[0]
TAB4 = STAT4 / "tables"
FIG4 = STAT4 / "figures"
YEARS = tuple(range(2000, 2026))
KS = list(range(2, 11))
OUT_TAB = TAB4 / "point4_annual_sparse_exposome_topk_sensitivity.csv"


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "m76", SCRIPT_DIR / "76_build_annual_region_adjusted_sparse_exposome_r2.py"
    )
    m76 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m76)
    _expose_new = m76.base.ROOT / "data" / "us_expose_new" / "processed"
    if (_expose_new / "exposure_state_month_expanded.csv").exists():
        m76.L.US_EXPOSE = _expose_new
    return m76


def ols_slope(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if len(xs) < 3:
        return np.nan
    b = np.polyfit(xs, ys, 1)[0]
    return b


def main() -> int:
    if False and OUT_TAB.exists():
        df = pd.read_csv(OUT_TAB)
    else:
        m76 = load_builder()
        covs = tuple(m76.BREED_BASELINE_COLS + m76.HERD_SCALE_BASELINE_COLS)
        vars_df = m76.load_variable_pool()
        scores = m76.load_yearly_scores(vars_df)
        panel = m76.load_annual_panel(vars_df)

        rows = []
        for year in YEARS:
            dyr = panel[panel["year"].eq(year)].copy()
            cov = m76.resolve_year_covariates(dyr, covs)
            for K in KS:
                sel = m76.choose_top_k(scores, year, top_k=K)
                exposures = tuple(x for x in sel["exposure"].tolist() if x in dyr.columns)
                nested = m76.weighted_nested_incremental_r2_test(
                    dyr, exposures, include_region=False, baseline_covariates=cov
                )
                incr = nested["incremental_r2"]
                rows.append({
                    "year": year, "top_k": K, "n_selected": len(exposures),
                    "incremental_r2": incr,
                    "incremental_r2_pct": 100 * incr if np.isfinite(incr) else np.nan,
                    "p": nested["p"],
                })
        df = pd.DataFrame(rows)
        df.to_csv(OUT_TAB, index=False, encoding="utf-8-sig")

    # per-K temporal slope (is the decline consistent across K?)
    print("K   slope(%/yr)  R2_2000  R2_2010  R2_2025  sig_years(p<.05)/26")
    for K in KS:
        s = df[df.top_k.eq(K)].sort_values("year")
        slope = ols_slope(s.year, s.incremental_r2_pct)
        def yr(y):
            v = s.loc[s.year.eq(y), "incremental_r2_pct"]
            return float(v.iloc[0]) if len(v) else np.nan
        sig = int((s["p"] < 0.05).sum())
        print(f"{K:<3d} {slope:+.2f}        {yr(2000):5.1f}    {yr(2010):5.1f}    {yr(2025):5.1f}    {sig}/26")

    print(f"\nWrote {OUT_TAB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
