#!/usr/bin/env python3
"""Plot raw Heat/Cold exposure intensity trends for selected variables."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
TAB = POINT / "tables"
FIG = POINT / "figures"
sys.path.insert(0, str(STAT))
import lib_statistics_panel as L  # noqa: E402


SELECTION = TAB / "point2_common_sense_directional_variable_selection.csv"
OUT_ANNUAL = TAB / "point2_heat_cold_common_sense_exposure_intensity_trends_annual.csv"
OUT_SUMMARY = TAB / "point2_heat_cold_common_sense_exposure_intensity_trends_summary.csv"
OUT_STEM = "main_point2_heat_cold_common_sense_exposure_intensity_trends"

DOMAIN_ORDER = ["Heat", "Cold"]
COLORS = {"Heat": "#32a4b4", "Cold": "#33c5b2"}
LABELS = {
    "daymet_dairy_weighted_thi_days_ge_72": "THI days >=72",
    "daymet_dairy_weighted_thi_heatload_ge72": "THI heatload >=72",
    "daymet_dairy_weighted_tmax_days_ge_32c": "Tmax days >=32C",
    "daymet_dairy_weighted_vpd_days_ge_4kpa": "VPD days >=4 kPa",
    "daymet_dairy_weighted_warm_nights_tmin_ge_20c": "Warm nights Tmin >=20C",
    "daymet_dairy_weighted_wetbulb_days_ge_26c": "Wet-bulb days >=26C",
    "daymet_dairy_weighted_hard_freeze_nights": "Hard-freeze nights",
    "daymet_dairy_weighted_ice_days": "Ice days",
    "daymet_dairy_weighted_snow_cover_days": "Snow-cover days",
    "daymet_dairy_weighted_swe_max_mm": "Max SWE (mm)",
    "daymet_dairy_weighted_thi_days_lt_45": "THI days <45",
    "daymet_dairy_weighted_wet_cold_load_lt45": "Wet-cold load <45",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "svg.fonttype": "none",
    }
)


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    path.write_text(svg, encoding="utf-8")


def selected_heat_cold() -> pd.DataFrame:
    d = pd.read_csv(SELECTION)
    d = d[
        d["selection_status"].eq("kept_common_sense")
        & d["domain_label"].isin(DOMAIN_ORDER)
    ].copy()
    d["domain_label"] = pd.Categorical(d["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    return d.sort_values(["domain_label", "exposure"])


def build_annual(panel: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        exposure = row["exposure"]
        domain = row["domain_label"]
        d = panel.loc[:, ["year", "state_alpha", "month", exposure]].replace([np.inf, -np.inf], np.nan)
        d = d.dropna(subset=[exposure])
        for year, s in d.groupby("year", observed=True):
            x = s[exposure].astype(float)
            n = int(x.notna().sum())
            mean = float(x.mean())
            sd = float(x.std(ddof=1)) if n > 1 else np.nan
            se = sd / np.sqrt(n) if n > 1 else np.nan
            rows.append(
                {
                    "domain_label": domain,
                    "exposure": exposure,
                    "exposure_label": LABELS.get(exposure, exposure),
                    "year": int(year),
                    "n_state_months": n,
                    "n_states": int(s["state_alpha"].nunique()),
                    "n_months": int(s["month"].nunique()),
                    "mean_raw": mean,
                    "sd_raw": sd,
                    "se_raw": se,
                    "ci95_low_raw": mean - 1.96 * se if np.isfinite(se) else np.nan,
                    "ci95_high_raw": mean + 1.96 * se if np.isfinite(se) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["domain_label"] = pd.Categorical(out["domain_label"], categories=DOMAIN_ORDER, ordered=True)
    return out.sort_values(["domain_label", "exposure", "year"])


def slope_summary(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (domain, exposure), s in annual.groupby(["domain_label", "exposure"], observed=True):
        s = s.dropna(subset=["year", "mean_raw"]).sort_values("year")
        x = s["year"].to_numpy(float)
        y = s["mean_raw"].to_numpy(float)
        if len(s) < 5 or np.std(y) <= 1e-12:
            slope = np.nan
            endpoint_change = np.nan
        else:
            slope = float(np.polyfit(x, y, 1)[0])
            endpoint_change = float(y[-1] - y[0])
        expected = "increase" if domain == "Heat" else "decrease"
        matches = (slope > 0 and endpoint_change > 0) if domain == "Heat" else (slope < 0 and endpoint_change < 0)
        rows.append(
            {
                "domain_label": domain,
                "exposure": exposure,
                "exposure_label": LABELS.get(exposure, exposure),
                "expected_direction": expected,
                "slope_raw_per_year": slope,
                "endpoint_change_2000_2025": endpoint_change,
                "trend_matches_expected": bool(matches) if np.isfinite(slope) and np.isfinite(endpoint_change) else False,
                "mean_2000": float(y[0]) if len(y) else np.nan,
                "mean_2025": float(y[-1]) if len(y) else np.nan,
                "n_years": int(len(s)),
            }
        )
    return pd.DataFrame(rows)


def plot(annual: pd.DataFrame) -> None:
    exposures = (
        annual[["domain_label", "exposure", "exposure_label"]]
        .drop_duplicates()
        .sort_values(["domain_label", "exposure"])
        .reset_index(drop=True)
    )
    n = len(exposures)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 8.0), constrained_layout=True)
    axes = np.ravel(axes)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")
    for i, meta in exposures.iterrows():
        ax = axes[i]
        s = annual[annual["exposure"].eq(meta["exposure"])].sort_values("year")
        x = s["year"].to_numpy(float)
        y = s["mean_raw"].to_numpy(float)
        lo = s["ci95_low_raw"].to_numpy(float)
        hi = s["ci95_high_raw"].to_numpy(float)
        color = COLORS[str(meta["domain_label"])]
        ax.fill_between(x, lo, hi, color=color, alpha=0.20, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.scatter(x, y, color=color, s=9, linewidth=0, zorder=3)
        ax.set_title(str(meta["exposure_label"]), fontsize=8, pad=3)
        ax.set_xlim(2000, 2025)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
        ax.tick_params(axis="both", labelsize=7, length=2.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.7)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.supxlabel("Year", fontsize=9)
    fig.supylabel("Annual mean exposure intensity (raw units, 95% CI)", fontsize=9)
    svg = FIG / f"{OUT_STEM}.svg"
    fig.savefig(svg, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    normalize_svg_text_style(svg)
    print(f"Wrote {svg}")


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel = L.load_panel().copy()
    selected = selected_heat_cold()
    annual = build_annual(panel, selected)
    summary = slope_summary(annual)
    annual.to_csv(OUT_ANNUAL, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(annual)
    print(f"Wrote {OUT_ANNUAL}")
    print(f"Wrote {OUT_SUMMARY}")
    print(summary[["domain_label", "exposure_label", "slope_raw_per_year", "endpoint_change_2000_2025", "trend_matches_expected"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
