#!/usr/bin/env python3
"""Three-row paired two-stage boxplots for beta_std and R2-family metrics."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import stats

POINT = Path(__file__).resolve().parents[1]
TAB = POINT / "tables"
FIG = POINT / "figures"

BETA_LONG = TAB / "point2_herd_adjusted_yearly_sensitivity_beta_std.csv"
R2_LONG = TAB / "point2_milk_per_cow_expanded_nonredundant_yearly_r2.csv"
SELECTION = TAB / "point2_common_sense_expanded_kept_variables.csv"
OUT = FIG / "supp_point2_beta_std_two_stage_mean_abs_beta_boxplots_by_class.svg"
OUT_WO = FIG / "supp_point2_beta_std_two_stage_mean_abs_beta_boxplots_by_class_wo_legend.svg"
VALUES = TAB / "point2_beta_std_two_stage_three_metric_by_variable.csv"
SUMMARY = TAB / "point2_beta_std_two_stage_three_metric_boxplots_by_class_summary.csv"
OUTLIERS = TAB / "point2_beta_std_two_stage_three_metric_boxplots_by_class_outliers_removed.csv"
STATS_OUT = TAB / "point2_beta_std_two_stage_three_metric_paired_class_tests.csv"
# Backward-compatible beta-only outputs used in nearby notes.
BETA_VALUES = TAB / "point2_beta_std_two_stage_mean_abs_beta_by_variable.csv"
BETA_SUMMARY = TAB / "point2_beta_std_two_stage_mean_abs_beta_boxplots_by_class_summary.csv"
BETA_OUTLIERS = TAB / "point2_beta_std_two_stage_mean_abs_beta_boxplots_by_class_outliers_removed.csv"
BETA_STATS = TAB / "point2_beta_std_two_stage_mean_abs_beta_paired_class_tests.csv"

DOMAIN_ORDER = ["Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand"]
COLORS = {
    "Heat": "#32a4b4",
    "Cold": "#33c5b2",
    "Severe weather": "#d5eada",
    "Forage": "#1E7A8D",
    "Feed market": "#fbc4ab",
    "Dairy market": "#E47666",
    "Market demand": "#f09d51",
}
METRICS = [
    {
        "metric": "mean_abs_beta_std",
        "label": "Mean |standardized β|",
        "early": "mean_abs_beta_2000_2014",
        "late": "mean_abs_beta_2015_2024",
    },
    {
        "metric": "adjusted_incremental_r2",
        "label": "Mean adjusted increased R² (%)",
        "early": "mean_adjusted_incremental_r2_2000_2014",
        "late": "mean_adjusted_incremental_r2_2015_2024",
        "plot_scale": 100.0,
    },
    {
        "metric": "partial_r2",
        "label": "Mean increased partial R² (%)",
        "early": "mean_partial_r2_2000_2014",
        "late": "mean_partial_r2_2015_2024",
        "plot_scale": 100.0,
    },
]

METRICS[0]["plot_scale"] = 1.0

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "text.color": "#171717",
        "axes.labelcolor": "#171717",
        "xtick.color": "#171717",
        "ytick.color": "#171717",
        "svg.fonttype": "none",
    }
)


def normalize_svg_text_style(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"font: ([0-9.]+)px 'Arial'", r"font-size: \1px; font-family: 'Arial'", svg)
    for size in [6, 7, 8, 9, 10, 11, 12, 13]:
        svg = svg.replace(f"font-size: {size}px;", f"font-size: {size:.2f}px;")
    path.write_text(svg, encoding="utf-8")


def kept_exposures() -> pd.DataFrame:
    selected = pd.read_csv(SELECTION)
    if "domain_label" not in selected.columns and "class_label" in selected.columns:
        selected = selected.rename(columns={"class_label": "domain_label"})
    return selected[selected["expanded_selection_status"].str.startswith("kept_expanded", na=False)][
        ["domain_label", "exposure"]
    ].drop_duplicates()


def harmonize_class_column(df: pd.DataFrame) -> pd.DataFrame:
    if "domain_label" not in df.columns and "class_label" in df.columns:
        df = df.rename(columns={"class_label": "domain_label"})
    return df


def stage_means(long: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    d = long.copy()
    early = (
        d[d["year"].between(2000, 2014)]
        .groupby(["domain_label", "exposure"], as_index=False)
        .agg(**{f"mean_{prefix}_2000_2014": (value_col, "mean"), f"n_years_{prefix}_2000_2014": ("year", "nunique")})
    )
    late = (
        d[d["year"].between(2015, 2024)]
        .groupby(["domain_label", "exposure"], as_index=False)
        .agg(**{f"mean_{prefix}_2015_2024": (value_col, "mean"), f"n_years_{prefix}_2015_2024": ("year", "nunique")})
    )
    return early.merge(late, on=["domain_label", "exposure"], how="outer")


def build_values() -> pd.DataFrame:
    kept = kept_exposures()
    beta = harmonize_class_column(pd.read_csv(BETA_LONG)).merge(kept, on=["domain_label", "exposure"], how="inner")
    beta = beta[
        beta["outcome"].eq("per_cow")
        & beta["status"].eq("ok")
        & beta["year"].between(2000, 2024)
        & beta["domain_label"].isin(DOMAIN_ORDER)
    ].copy()
    beta["abs_beta_std"] = beta["beta_std_per_1sd_exposure"].abs()
    vals = stage_means(beta, "abs_beta_std", "abs_beta")

    r2 = harmonize_class_column(pd.read_csv(R2_LONG)).merge(kept, on=["domain_label", "exposure"], how="inner")
    r2 = r2[r2["status"].eq("ok") & r2["year"].between(2000, 2024) & r2["domain_label"].isin(DOMAIN_ORDER)].copy()
    # Negative adjusted incremental/partial values are not visual contribution; clip to 0 for comparability with point2-style R2 plots.
    r2["adjusted_incremental_r2_plot"] = pd.to_numeric(r2["adjusted_incremental_r2"], errors="coerce").clip(lower=0)
    r2["partial_r2_plot"] = pd.to_numeric(r2["partial_r2"], errors="coerce").clip(lower=0)
    vals = vals.merge(stage_means(r2, "adjusted_incremental_r2_plot", "adjusted_incremental_r2"), on=["domain_label", "exposure"], how="outer")
    vals = vals.merge(stage_means(r2, "partial_r2_plot", "partial_r2"), on=["domain_label", "exposure"], how="outer")
    vals.to_csv(VALUES, index=False)
    vals[["domain_label", "exposure", "mean_abs_beta_2000_2014", "n_years_abs_beta_2000_2014", "mean_abs_beta_2015_2024", "n_years_abs_beta_2015_2024"]].to_csv(BETA_VALUES, index=False)
    return vals


def remove_metric_outliers(d: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    early_col, late_col = spec["early"], spec["late"]
    paired = d.dropna(subset=[early_col, late_col]).copy()
    kept_rows, out_rows = [], []
    for domain, g in paired.groupby("domain_label", sort=False):
        vals = pd.concat([g[early_col], g[late_col]], ignore_index=True).dropna()
        if len(vals) < 4:
            upper = np.inf
        else:
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            upper = q3 + 1.5 * (q3 - q1)
        is_out = (g[early_col] > upper) | (g[late_col] > upper)
        kept_rows.append(g.loc[~is_out].copy())
        if is_out.any():
            tmp = g.loc[is_out].copy()
            tmp["metric"] = spec["metric"]
            tmp["outlier_upper_1p5iqr"] = upper
            out_rows.append(tmp)
    kept_df = pd.concat(kept_rows, ignore_index=True) if kept_rows else paired.iloc[0:0].copy()
    out_df = pd.concat(out_rows, ignore_index=True) if out_rows else paired.iloc[0:0].copy()
    return kept_df, out_df


def bh_adjust(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    vals = p[ok].to_numpy(float)
    m = len(vals)
    if m == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    restored = np.empty(m, dtype=float)
    restored[order] = np.clip(adj, 0, 1)
    out.loc[ok] = restored
    return out


def p_to_label(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_tests(d_plot: pd.DataFrame, spec: dict) -> pd.DataFrame:
    rows = []
    e_col, l_col = spec["early"], spec["late"]
    for domain in DOMAIN_ORDER:
        g = d_plot[d_plot["domain_label"].eq(domain)]
        early = g[e_col].to_numpy(float)
        late = g[l_col].to_numpy(float)
        diff = late - early
        n = int(len(g))
        if n >= 2 and np.nanstd(diff) > 0:
            t_stat, t_p = stats.ttest_rel(late, early, nan_policy="omit")
        else:
            t_stat, t_p = np.nan, np.nan
        try:
            if n >= 2 and np.any(np.abs(diff) > 0):
                w_stat, w_p = stats.wilcoxon(late, early, zero_method="wilcox", alternative="two-sided", mode="auto")
            else:
                w_stat, w_p = np.nan, np.nan
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        rows.append(
            {
                "metric": spec["metric"],
                "domain_label": domain,
                "n_pairs": n,
                "median_early": float(np.nanmedian(early)) if n else np.nan,
                "median_late": float(np.nanmedian(late)) if n else np.nan,
                "median_late_minus_early": float(np.nanmedian(diff)) if n else np.nan,
                "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
                "paired_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["wilcoxon_p_bh"] = bh_adjust(out["wilcoxon_p"])
    out["paired_t_p_bh"] = bh_adjust(out["paired_t_p"])
    out["wilcoxon_sig"] = out["wilcoxon_p"].map(p_to_label)
    out["wilcoxon_bh_sig"] = out["wilcoxon_p_bh"].map(p_to_label)
    return out


def title_for_domain(domain: str) -> str:
    return (
        domain.replace("Severe weather", "Severe\nweather")
        .replace("Feed market", "Feed\nmarket")
        .replace("Dairy market", "Dairy\nmarket")
        .replace("Market demand", "Market\ndemand")
    )


def draw(show_legend: bool) -> None:
    values = build_values().replace([np.inf, -np.inf], np.nan)
    plot_by_metric = {}
    outlier_rows = []
    test_rows = []
    summary_rows = []
    for spec in METRICS:
        d_plot, outliers = remove_metric_outliers(values, spec)
        plot_by_metric[spec["metric"]] = d_plot
        if len(outliers):
            outlier_rows.append(outliers)
        tests = paired_tests(d_plot, spec)
        test_rows.append(tests)
        for domain in DOMAIN_ORDER:
            g_all = values[values["domain_label"].eq(domain)].dropna(subset=[spec["early"], spec["late"]])
            g = d_plot[d_plot["domain_label"].eq(domain)]
            summary_rows.append(
                {
                    "metric": spec["metric"],
                    "domain_label": domain,
                    "n_variables_total_paired": int(len(g_all)),
                    "n_variables_plotted": int(len(g)),
                    "n_early_greater": int((g[spec["early"]] > g[spec["late"]]).sum()),
                    "n_late_greater": int((g[spec["late"]] > g[spec["early"]]).sum()),
                    "median_early": float(g[spec["early"]].median()) if len(g) else np.nan,
                    "median_late": float(g[spec["late"]].median()) if len(g) else np.nan,
                }
            )
    pd.concat(outlier_rows, ignore_index=True).to_csv(OUTLIERS, index=False) if outlier_rows else pd.DataFrame().to_csv(OUTLIERS, index=False)
    tests_all = pd.concat(test_rows, ignore_index=True)
    tests_all.to_csv(STATS_OUT, index=False)
    summary_all = pd.DataFrame(summary_rows)
    summary_all.to_csv(SUMMARY, index=False)
    # Backward-compatible beta-only outputs.
    summary_all[summary_all["metric"].eq("mean_abs_beta_std")].to_csv(BETA_SUMMARY, index=False)
    tests_all[tests_all["metric"].eq("mean_abs_beta_std")].to_csv(BETA_STATS, index=False)
    out_beta = pd.read_csv(OUTLIERS) if OUTLIERS.exists() and OUTLIERS.stat().st_size else pd.DataFrame()
    out_beta = out_beta[out_beta.get("metric", pd.Series(dtype=str)).eq("mean_abs_beta_std")] if len(out_beta) else out_beta
    out_beta.to_csv(BETA_OUTLIERS, index=False)

    fig_h = 7.7 if show_legend else 8.15
    fig, axes = plt.subplots(len(METRICS), len(DOMAIN_ORDER), figsize=(10.8, fig_h), sharex=True, constrained_layout=True)
    fig.patch.set_alpha(0)

    for row_i, spec in enumerate(METRICS):
        d_plot = plot_by_metric[spec["metric"]]
        scale = float(spec.get("plot_scale", 1.0))
        ymax = float(d_plot[[spec["early"], spec["late"]]].to_numpy().max()) * scale * 1.12 if len(d_plot) else 1.0
        tests = tests_all[tests_all["metric"].eq(spec["metric"])]
        for col_i, domain in enumerate(DOMAIN_ORDER):
            ax = axes[row_i, col_i]
            g = d_plot[d_plot["domain_label"].eq(domain)].sort_values("exposure")
            ax.set_facecolor("none")
            vals = [g[spec["early"]].to_numpy(float) * scale, g[spec["late"]].to_numpy(float) * scale]
            bp = ax.boxplot(
                vals,
                positions=[0, 1],
                widths=0.48,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111111", "linewidth": 1.1},
                whiskerprops={"color": "#333333", "linewidth": 0.75},
                capprops={"color": "#333333", "linewidth": 0.75},
                boxprops={"edgecolor": "#222222", "linewidth": 0.75},
            )
            for patch, alpha in zip(bp["boxes"], [0.62, 1.0]):
                patch.set_facecolor(COLORS[domain])
                patch.set_alpha(alpha)
            jitter_rng = np.random.default_rng(20240826 + row_i * 100 + col_i)
            for _, r in g.iterrows():
                x0 = 0 + jitter_rng.normal(0, 0.018)
                x1 = 1 + jitter_rng.normal(0, 0.018)
                y0 = float(r[spec["early"]]) * scale
                y1 = float(r[spec["late"]]) * scale
                ax.plot([x0, x1], [y0, y1], color="#555555", linewidth=0.42, alpha=0.42, zorder=1)
                ax.scatter([x0, x1], [y0, y1], s=7, color=COLORS[domain], edgecolor="#222222", linewidth=0.16, alpha=0.95, zorder=2)
            n = int(len(g))
            early_greater = int((g[spec["early"]] > g[spec["late"]]).sum()) if n else 0
            p_row = tests[tests["domain_label"].eq(domain)]
            p_label = p_row["wilcoxon_sig"].iloc[0] if len(p_row) else "n/a"
            title_prefix = f"{title_for_domain(domain)}\n" if row_i == 0 else ""
            ax.set_title(f"{title_prefix}{p_label}", fontsize=11, pad=4)
            ax.set_xlim(-0.45, 1.45)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["2000-2014", "2015-2024"], rotation=45, ha="right", fontsize=11)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(axis="y", labelsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_linewidth(0.75)
            ax.spines["bottom"].set_linewidth(0.75)
            if col_i == 0:
                ax.set_ylabel(spec["label"], fontsize=11)

    if show_legend:
        fig.text(0.5, 1.01, "Two-stage paired exposure-milk per cow association metrics", ha="center", va="bottom", fontsize=12, fontweight="bold")
        out = OUT
    else:
        out = OUT_WO
    fig.savefig(out, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    normalize_svg_text_style(out)
    print(f"Wrote {out}")
    print(f"Wrote {VALUES}")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {OUTLIERS}")
    print(f"Wrote {STATS_OUT}")


def main() -> int:
    draw(show_legend=True)
    draw(show_legend=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
