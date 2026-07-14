#!/usr/bin/env python3
"""Leave-one-year-out (LOYO) robustness for the Fig 2c yearly-trend analysis.

The main trend regresses per-domain median |standardized beta| on calendar year.
This script drops each year in turn, refits the same regression, and summarizes
whether the slope sign, magnitude and R2 are stable. Outputs feed a supplementary
robustness figure.
"""

from __future__ import annotations

import math
from pathlib import Path

import csv

POINT = Path(__file__).resolve().parents[1]
TAB = POINT / "tables"
IN = TAB / "point1_chord_signal_yearly_domain_summary.csv"
OUT_SUMMARY = TAB / "point1_yearly_trend_loyo_summary.csv"
OUT_LINES = TAB / "point1_yearly_trend_loyo_lines.csv"

DOMAINS = [
    "Heat", "Cold", "Severe weather", "Forage condition",
    "Agricultural pesticides", "Feed market", "Milk price / dairy market",
    "Market demand", "Dairy scale",
]
SCOPES = {"per_cow_26": "Milk per cow", "total_26": "Total production"}


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # two-sided p for slope via t distribution (normal approx for df>=8)
    if n > 2 and ss_res > 0:
        se_b = math.sqrt((ss_res / (n - 2)) / sxx)
        t = b / se_b if se_b > 0 else float("inf")
        p = math.erfc(abs(t) / math.sqrt(2))
    else:
        p = float("nan")
    return a, b, r2, p


def main():
    rows = list(csv.DictReader(open(IN, encoding="utf-8-sig")))
    summ = []
    lines = []
    for scope, label in SCOPES.items():
        for dom in DOMAINS:
            s = [r for r in rows if r["phenotype_scope"] == scope
                 and r["domain"] == dom and r["median_abs_beta"] not in ("", "NA")]
            pts = sorted((int(float(r["year"])), float(r["median_abs_beta"])) for r in s)
            if len(pts) < 8:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            a0, b0, r20, p0 = ols(xs, ys)
            loyo = []
            for i in range(len(pts)):
                xx = xs[:i] + xs[i + 1:]
                yy = ys[:i] + ys[i + 1:]
                intercept_i, slope_i, r2_i, _ = ols(xx, yy)
                loyo.append((xs[i], intercept_i, slope_i, r2_i))
                lines.append({
                    "phenotype": label, "domain": dom, "dropped_year": xs[i],
                    "slope": slope_i, "intercept": intercept_i,
                })
            sl = [l[2] for l in loyo]
            rr = [l[3] for l in loyo]
            stable = sum(1 for x in sl if (x < 0) == (b0 < 0)) / len(sl)
            summ.append({
                "phenotype": label, "domain": dom, "n_years": len(pts),
                "full_slope": b0, "full_intercept": a0, "full_r2": r20, "full_p": p0,
                "loyo_slope_min": min(sl), "loyo_slope_max": max(sl),
                "loyo_r2_min": min(rr), "loyo_r2_max": max(rr),
                "sign_stable_share": stable,
            })

    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    with open(OUT_LINES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lines[0].keys()))
        w.writeheader()
        w.writerows(lines)
    print(f"Wrote {OUT_SUMMARY} ({len(summ)} rows) and {OUT_LINES} ({len(lines)} rows)")
    for r in summ:
        if r["phenotype"] == "Milk per cow":
            print(f"  {r['domain']:24s} slope={r['full_slope']:+.5f} R2={r['full_r2']:.2f} "
                  f"LOYO slope[{r['loyo_slope_min']:+.5f},{r['loyo_slope_max']:+.5f}] "
                  f"sign_stable={100*r['sign_stable_share']:.0f}%")


if __name__ == "__main__":
    main()
