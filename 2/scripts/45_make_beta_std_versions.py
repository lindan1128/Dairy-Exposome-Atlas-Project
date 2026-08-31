#!/usr/bin/env python3
"""Create beta_std versions of retained Point 2 beta figures."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr


POINT = Path(__file__).resolve().parents[1]
STAT = POINT.parent
TAB = POINT / "tables"
FIG = POINT / "figures"
sys.path.insert(0, str(STAT))

import lib_statistics_panel as L  # noqa: E402


YEARLY = TAB / "point2_herd_adjusted_yearly_sensitivity.csv"
SELECTION = TAB / "point2_common_sense_expanded_kept_variables.csv"
VARIANT_LONG = TAB / "point2_four_regression_variant_yearly_sensitivity_long.csv"
LOSO_LONG = TAB / "point2_leave_one_state_out_main_model_yearly_sensitivity_long.csv"
LME_LONG = TAB / "point2_lme4_yearly_sensitivity_trial_long.csv"
SCATTER_BASE = TAB / "point2_2015_2024_exposure_intensity_vs_abs_beta_change.csv"
THREE_METRIC_VALUES = TAB / "point2_beta_std_two_stage_three_metric_by_variable.csv"
THREE_METRIC_OUTLIERS = TAB / "point2_beta_std_two_stage_three_metric_boxplots_by_class_outliers_removed.csv"

OUT_SD = TAB / "point2_milk_per_cow_yearly_log_sd_for_beta_std.csv"
OUT_LONG = TAB / "point2_herd_adjusted_yearly_sensitivity_beta_std.csv"

DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
SLOPE_DOMAIN_ORDER = ["Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand", "Heat"]
COLORS = {
    "Heat": "#32a4b4",
    "Cold": "#33c5b2",
    "Severe weather": "#d5eada",
    "Forage": "#1E7A8D",
    "Feed market": "#fbc4ab",
    "Dairy market": "#E47666",
    "Market demand": "#f09d51",
}

PANEL_SPECS = [
    ("fixest_year_month_fe", "Sensitivity", "FE + YM"),
    ("fixest_pooled_linear_time", "Sensitivity", "FE + x×year"),
    ("loso_median", "Sensitivity", "LOSO"),
    ("mgcv_gam", "Robustness", "GAM"),
    ("geepack_gee", "Robustness", "GEE"),
    ("lme4_random_intercept_slope", "Robustness", "LME"),
]

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
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
    svg = svg.replace("font-size: 9px;", "font-size: 9.00px;")
    path.write_text(svg, encoding="utf-8")


def weighted_mean_sd(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    x_arr = x.to_numpy(float)
    w_arr = w.to_numpy(float)
    keep = np.isfinite(x_arr) & np.isfinite(w_arr) & (w_arr > 0)
    x_arr = x_arr[keep]
    w_arr = w_arr[keep]
    if x_arr.size == 0 or w_arr.sum() <= 0:
        return np.nan, np.nan
    mean = float(np.sum(w_arr * x_arr) / np.sum(w_arr))
    sd = float(np.sqrt(np.sum(w_arr * (x_arr - mean) ** 2) / np.sum(w_arr)))
    return mean, sd


def build_yearly_outcome_sd() -> pd.DataFrame:
    panel = L.load_panel()
    if "milk_per_cow_kg" not in panel.columns:
        panel["milk_per_cow_kg"] = panel["milk_per_cow_lb"] * 0.45359237
    panel["log_milk_per_cow_kg"] = np.log(panel["milk_per_cow_kg"].where(panel["milk_per_cow_kg"] > 0))
    d = panel.loc[
        panel["year"].between(2000, 2024),
        ["year", "state_alpha", "month", "milk_cows_head", "log_milk_per_cow_kg"],
    ].replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d["milk_cows_head"] > 0].copy()
    rows = []
    for year, g in d.groupby("year", sort=True):
        mean, sd = weighted_mean_sd(g["log_milk_per_cow_kg"], g["milk_cows_head"])
        rows.append(
            {
                "year": int(year),
                "log_milk_per_cow_kg_weighted_mean": mean,
                "log_milk_per_cow_kg_weighted_sd": sd,
                "n_state_months": int(len(g)),
                "n_states": int(g["state_alpha"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SD, index=False)
    return out


def add_beta_std(yearly_path: Path = YEARLY) -> pd.DataFrame:
    sd = build_yearly_outcome_sd()
    yearly = pd.read_csv(yearly_path)
    out = yearly.merge(sd[["year", "log_milk_per_cow_kg_weighted_sd"]], on="year", how="left")
    out["beta_std_per_1sd_exposure"] = (
        out["beta_log_per_1sd_exposure"] / out["log_milk_per_cow_kg_weighted_sd"]
    )
    out.to_csv(OUT_LONG, index=False)
    return out


def expanded_kept_exposures() -> set[str]:
    selected = pd.read_csv(SELECTION)
    kept = selected[selected["expanded_selection_status"].str.startswith("kept_expanded", na=False)].copy()
    return set(kept["exposure"])


def mean_beta_boxplot_retained_exposures() -> set[str]:
    """Variables plotted in the mean |standardized beta| row of the supplement boxplot."""
    if not THREE_METRIC_VALUES.exists():
        return expanded_kept_exposures() - {"market_log_population_total", "storm_event_types"}
    values = pd.read_csv(THREE_METRIC_VALUES)
    paired = values.dropna(subset=["mean_abs_beta_2000_2014", "mean_abs_beta_2015_2024"]).copy()
    if THREE_METRIC_OUTLIERS.exists():
        outliers = pd.read_csv(THREE_METRIC_OUTLIERS)
        beta_outliers = set(outliers.loc[outliers["metric"].eq("mean_abs_beta_std"), "exposure"].dropna())
    else:
        beta_outliers = {"market_log_population_total", "storm_event_types"}
    return set(paired["exposure"]) - beta_outliers


def zscore(x: pd.Series) -> pd.Series:
    arr = pd.to_numeric(x, errors="coerce").astype(float)
    sd = arr.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index)
    return (arr - arr.mean(skipna=True)) / sd


def exposure_intensity_trends(exposures: set[str]) -> pd.DataFrame:
    panel = L.load_panel()
    panel_exposures = [exposure for exposure in sorted(exposures) if exposure in panel.columns]
    selection = pd.read_csv(SELECTION)
    domain_lookup = (
        selection.loc[selection["exposure"].isin(panel_exposures), ["exposure", "class_label"]]
        .drop_duplicates("exposure")
        .set_index("exposure")["class_label"]
        .to_dict()
    )
    label_lookup = {}
    if SCATTER_BASE.exists():
        base_labels = pd.read_csv(SCATTER_BASE, usecols=lambda c: c in {"exposure", "exposure_label"})
        label_lookup = (
            base_labels.dropna(subset=["exposure"])
            .drop_duplicates("exposure")
            .set_index("exposure")["exposure_label"]
            .to_dict()
        )

    rows = []
    for exposure in panel_exposures:
        x = zscore(panel[exposure])
        annual = (
            pd.DataFrame({"year": panel["year"], "exposure_z": x})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .groupby("year", as_index=False)
            .agg(
                annual_mean_exposure_z=("exposure_z", "mean"),
                n_state_months=("exposure_z", "size"),
            )
        )
        a = annual[annual["year"].between(2015, 2024)].sort_values("year")
        fit = linregress(a["year"], a["annual_mean_exposure_z"]) if len(a) >= 3 else None
        rows.append(
            {
                "class_label": domain_lookup.get(exposure),
                "exposure": exposure,
                "exposure_label": label_lookup.get(exposure, exposure),
                "intensity_slope_z_per_year_2015_2024": fit.slope if fit is not None else np.nan,
                "intensity_endpoint_change_z_2015_2024": (
                    float(a["annual_mean_exposure_z"].iloc[-1] - a["annual_mean_exposure_z"].iloc[0])
                    if len(a) >= 2
                    else np.nan
                ),
                "n_intensity_years": int(len(a)),
                "mean_n_state_months_per_year": float(a["n_state_months"].mean()) if len(a) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_summary(long: pd.DataFrame, exposures: set[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = long[
        long["outcome"].eq("per_cow")
        & long["status"].eq("ok")
        & long["year"].between(2000, 2024)
        & long["class_label"].isin(DOMAIN_ORDER)
    ].copy()
    if exposures is not None:
        d = d[d["exposure"].isin(exposures)].copy()
    d["abs_beta_std"] = d["beta_std_per_1sd_exposure"].abs()
    summary = (
        d.groupby(["year", "class_label"], as_index=False)
        .agg(median_abs_beta_std=("abs_beta_std", "median"))
        .pivot(index="year", columns="class_label", values="median_abs_beta_std")
        .reindex(index=range(2000, 2025), columns=DOMAIN_ORDER)
        .sort_index()
    )
    n_exp = (
        d.groupby(["year", "class_label"], as_index=False)
        .agg(n_exposures=("exposure", "nunique"))
        .pivot(index="year", columns="class_label", values="n_exposures")
        .reindex(index=range(2000, 2025), columns=DOMAIN_ORDER)
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    return summary, n_exp


def save_stack_plot(summary: pd.DataFrame, stem: str, show_legend: bool) -> None:
    height = 3.9 if show_legend else 3.45
    fig, ax = plt.subplots(figsize=(8.25, height), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    plot_values = summary.fillna(0)
    years = plot_values.index.to_numpy()
    base = pd.Series(0.0, index=plot_values.index)
    for domain in reversed(DOMAIN_ORDER):
        positive = plot_values[domain].clip(lower=0)
        ax.fill_between(
            years,
            base.to_numpy(),
            (base + positive).to_numpy(),
            label=domain,
            color=COLORS[domain],
            linewidth=0,
        )
        base += positive
    ax.set_xlim(2000, 2025)
    ax.set_ylim(0, float(base.max()) * 1.08)
    ax.axhline(0, color="#444444", linewidth=0.55)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Stacked median |standardized β|", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        handle_by_label = dict(zip(labels, handles))
        ax.legend(
            [handle_by_label[domain] for domain in DOMAIN_ORDER],
            DOMAIN_ORDER,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            frameon=False,
            fontsize=9,
            columnspacing=1.4,
            handlelength=1.5,
        )
    suffix = "" if show_legend else "_wo_legend"
    svg = FIG / f"{stem}{suffix}.svg"
    fig.savefig(svg, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    normalize_svg_text_style(svg)


def save_two_stage(summary: pd.DataFrame, table_name: str, stem: str) -> None:
    rows = []
    fitted = {}
    for domain in SLOPE_DOMAIN_ORDER:
        for stage, start, end in [("Early (2000-2014)", 2000, 2014), ("Late (2015-2024)", 2015, 2024)]:
            d = summary.loc[summary.index.to_series().between(start, end), [domain]].dropna()
            years = d.index.to_series().astype(float)
            fit = linregress(years, d[domain])
            fitted[(domain, start)] = (years.reset_index(drop=True), pd.Series(fit.intercept + fit.slope * years))
            rows.append(
                {
                    "method": "milk per cow beta_std",
                    "domain": domain,
                    "stage": stage,
                    "start_year": start,
                    "end_year": end,
                    "slope_abs_beta_std_per_year": fit.slope,
                    "linear_p": fit.pvalue,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / table_name, index=False)

    fig, ax = plt.subplots(figsize=(2.6, 1.15), constrained_layout=True)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for domain in SLOPE_DOMAIN_ORDER:
        early_years, early_fit = fitted[(domain, 2000)]
        late_years, late_fit = fitted[(domain, 2015)]
        ax.plot(early_years, early_fit, color=COLORS[domain], linewidth=2.7)
        ax.plot(late_years, late_fit, color=COLORS[domain], linewidth=2.7)
    ax.axvline(2015, color="#777777", linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    ax.set_xlim(1999.5, 2025.5)
    ax.set_xticks([2000, 2015, 2025])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Median |standardized β|", fontsize=9, pad=5)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    svg = FIG / f"{stem}.svg"
    fig.savefig(svg, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    normalize_svg_text_style(svg)


def make_beta_summary_figures(long: pd.DataFrame) -> None:
    retained_exposures = mean_beta_boxplot_retained_exposures()
    specs = [
        (
            "main_point2_milk_per_cow_expanded_nonredundant_yearly_beta_std_point2_style",
            "point2_milk_per_cow_expanded_nonredundant_yearly_beta_std_point2_style_by_year.csv",
            "point2_milk_per_cow_expanded_nonredundant_yearly_beta_std_point2_style_n_exposures.csv",
            retained_exposures,
            True,
        ),
        (
            "main_point2_per_cow_herd_adjusted_yearly_beta_std_point2_style",
            "point2_per_cow_herd_adjusted_yearly_beta_std_point2_style_by_year.csv",
            "point2_per_cow_herd_adjusted_yearly_beta_std_point2_style_n_exposures.csv",
            retained_exposures,
            False,
        ),
    ]
    for stem, table_name, n_name, exposures, make_slopes in specs:
        summary, n_exp = make_summary(long, exposures)
        summary.reset_index(names="year").to_csv(TAB / table_name, index=False)
        n_exp.reset_index(names="year").to_csv(TAB / n_name, index=False)
        save_stack_plot(summary, stem, show_legend=True)
        save_stack_plot(summary, stem, show_legend=False)
        if make_slopes:
            save_two_stage(
                summary,
                "point2_milk_per_cow_expanded_nonredundant_yearly_beta_std_point2_style_two_stage_class_slopes.csv",
                "main_point2_milk_per_cow_expanded_nonredundant_yearly_beta_std_point2_style_two_stage_class_slopes",
            )


def save_beta_correlation_std() -> None:
    sd = pd.read_csv(OUT_SD)
    retained_exposures = mean_beta_boxplot_retained_exposures()
    long = pd.read_csv(VARIANT_LONG).merge(sd[["year", "log_milk_per_cow_kg_weighted_sd"]], on="year", how="left")
    long["beta_std_per_1sd_exposure"] = (
        long["beta_log_per_1sd_exposure"] / long["log_milk_per_cow_kg_weighted_sd"]
    )
    base = (
        long.loc[
            long["variant"].eq("fixest_fe")
            & long["status"].eq("ok")
            & long["year"].between(2000, 2024)
            & long["class_label"].isin(DOMAIN_ORDER),
            ["class_label", "exposure", "year", "beta_std_per_1sd_exposure"],
        ]
        .rename(columns={"beta_std_per_1sd_exposure": "main_beta_std"})
        .dropna()
    )
    base = base[base["exposure"].isin(retained_exposures)].copy()
    panels = []
    for variant, analysis_type, label in PANEL_SPECS:
        if variant == "loso_median":
            loso = pd.read_csv(LOSO_LONG).merge(sd[["year", "log_milk_per_cow_kg_weighted_sd"]], on="year", how="left")
            loso["beta_std_per_1sd_exposure"] = (
                loso["beta_log_per_1sd_exposure"] / loso["log_milk_per_cow_kg_weighted_sd"]
            )
            alt = (
                loso.loc[
                    loso["status"].eq("ok")
                    & loso["year"].between(2000, 2024)
                    & loso["class_label"].isin(DOMAIN_ORDER),
                    ["class_label", "exposure", "year", "beta_std_per_1sd_exposure"],
                ]
                .dropna()
                .groupby(["class_label", "exposure", "year"], as_index=False)
                .agg(model_beta_std=("beta_std_per_1sd_exposure", "median"))
            )
            alt = alt[alt["exposure"].isin(retained_exposures)].copy()
        elif variant == "lme4_random_intercept_slope":
            lme = pd.read_csv(LME_LONG)
            alt = (
                lme.loc[
                    lme["status"].eq("ok")
                    & lme["year"].between(2000, 2024)
                    & lme["class_label"].isin(DOMAIN_ORDER),
                    ["class_label", "exposure", "year", "beta_std_per_1sd_exposure"],
                ]
                .rename(columns={"beta_std_per_1sd_exposure": "model_beta_std"})
                .dropna()
            )
            alt = alt[alt["exposure"].isin(retained_exposures)].copy()
        else:
            alt = (
                long.loc[
                    long["variant"].eq(variant)
                    & long["status"].eq("ok")
                    & long["year"].between(2000, 2024)
                    & long["class_label"].isin(DOMAIN_ORDER),
                    ["class_label", "exposure", "year", "beta_std_per_1sd_exposure"],
                ]
                .rename(columns={"beta_std_per_1sd_exposure": "model_beta_std"})
                .dropna()
            )
            alt = alt[alt["exposure"].isin(retained_exposures)].copy()
        pair = base.merge(alt, on=["class_label", "exposure", "year"], how="inner")
        pair["model"] = variant
        pair["model_label"] = label
        pair["analysis_type"] = analysis_type
        panels.append(pair)
    pairs = pd.concat(panels, ignore_index=True)
    summary_rows = []
    for (model, model_label, analysis_type), d in pairs.groupby(["model", "model_label", "analysis_type"], sort=False):
        r, p = pearsonr(d["main_beta_std"], d["model_beta_std"]) if len(d) >= 3 else (np.nan, np.nan)
        summary_rows.append(
            {
                "model": model,
                "model_label": model_label,
                "analysis_type": analysis_type,
                "n_points": len(d),
                "n_exposures": d["exposure"].nunique(),
                "n_years": d["year"].nunique(),
                "pearson_r": r,
                "pearson_p": p,
            }
        )
    pairs.to_csv(TAB / "point2_beta_std_correlation_sensitivity_robustness_model_pairs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        TAB / "point2_beta_std_correlation_sensitivity_robustness_model_summary.csv", index=False
    )

    def draw(show_legend: bool) -> None:
        height = 2.85 if show_legend else 1.625
        if show_legend:
            fig = plt.figure(figsize=(9.20, height), constrained_layout=True)
            gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.18, 1, 1, 1])
            axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 4, 5, 6]]
        else:
            fig = plt.figure(figsize=(8.10, height), constrained_layout=True)
            gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 0.18, 1, 1, 1])
            axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 4, 5, 6]]
        fig.patch.set_alpha(0)
        for ax, (variant, _, label) in zip(axes, PANEL_SPECS):
            d = pairs[pairs["model"].eq(variant)]
            if not show_legend:
                d = d[(d["main_beta_std"].abs() <= 1) & (d["model_beta_std"].abs() <= 1)]
            title = label
            if not show_legend:
                title = ""
            ax.set_facecolor("none")
            ax.plot([-1, 1], [-1, 1], color="#777777", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
            ax.axhline(0, color="#555555", linewidth=0.55, zorder=0)
            ax.axvline(0, color="#555555", linewidth=0.55, zorder=0)
            for domain in DOMAIN_ORDER:
                dd = d[d["class_label"].eq(domain)]
                ax.scatter(
                    dd["main_beta_std"],
                    dd["model_beta_std"],
                    s=14,
                    color=COLORS[domain],
                    edgecolor="#222222",
                    linewidth=0.18,
                    alpha=0.86,
                    clip_on=show_legend,
                    label=domain if ax is axes[0] else None,
                )
            axis_pad = 0 if show_legend else 0.12
            ax.set_xlim(-1 - axis_pad, 1 + axis_pad)
            ax.set_ylim(-1 - axis_pad, 1 + axis_pad)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.set_title(title, fontsize=9, pad=4)
            ax.tick_params(axis="both", labelsize=9)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Sensitivity/robustness standardized β" if show_legend else "", fontsize=9)
        for ax in axes:
            ax.set_xlabel("Main standardized β", fontsize=9)
        if show_legend:
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False, fontsize=9)
        suffix = "" if show_legend else "_wo_legend"
        svg = FIG / f"main_point2_beta_std_correlation_sensitivity_robustness_models{suffix}.svg"
        fig.savefig(svg, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
        plt.close(fig)
        normalize_svg_text_style(svg)

    draw(show_legend=True)
    draw(show_legend=False)


def save_scatter_std() -> None:
    base = pd.read_csv(SCATTER_BASE)
    if THREE_METRIC_OUTLIERS.exists():
        outliers = pd.read_csv(THREE_METRIC_OUTLIERS)
        scatter_outliers = set(outliers["exposure"].dropna())
    else:
        scatter_outliers = {"market_log_population_total", "storm_event_types"}
    scatter_outliers.add("market_population_total")
    base = base[~base["exposure"].isin(scatter_outliers)].copy()
    sd = pd.read_csv(OUT_SD)
    sens = pd.read_csv(TAB / "point2_2015_2024_percow_nominal_alpha02_yearly_sensitivity.csv")
    sens = sens[sens["exposure"].isin(set(base["exposure"]))].copy()
    sens = sens.merge(sd[["year", "log_milk_per_cow_kg_weighted_sd"]], on="year", how="left")
    sens["abs_beta_std"] = (
        sens["beta_log_per_1sd_exposure"] / sens["log_milk_per_cow_kg_weighted_sd"]
    ).abs()
    rows = []
    for (domain, exposure), g in sens.groupby(["class_label", "exposure"], sort=False):
        g = g[g["year"].between(2015, 2024)].sort_values("year")
        if len(g) < 3:
            continue
        fit = linregress(g["year"], g["abs_beta_std"])
        rows.append(
            {
                "class_label": domain,
                "exposure": exposure,
                "mean_abs_beta_std_2015_2024": float(g["abs_beta_std"].mean()),
                "abs_beta_std_slope_per_year_2015_2024": fit.slope,
                "abs_beta_std_endpoint_change_2015_2024": float(g["abs_beta_std"].iloc[-1] - g["abs_beta_std"].iloc[0]),
            }
        )
    beta_std = pd.DataFrame(rows)
    out = base.drop(
        columns=[c for c in base.columns if c.startswith("mean_abs_beta") or c.startswith("abs_beta_slope") or c.startswith("abs_beta_endpoint")],
        errors="ignore",
    ).merge(beta_std, on=["class_label", "exposure"], how="inner")
    out.to_csv(TAB / "point2_2015_2024_exposure_intensity_vs_abs_beta_std_change.csv", index=False)


def main() -> int:
    long = add_beta_std()
    make_beta_summary_figures(long)
    save_beta_correlation_std()
    save_scatter_std()
    print(f"Wrote {OUT_SD}")
    print(f"Wrote {OUT_LONG}")
    print("Wrote beta_std figure/table derivatives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
