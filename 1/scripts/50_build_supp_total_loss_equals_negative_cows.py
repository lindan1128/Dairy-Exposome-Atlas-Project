#!/usr/bin/env python3
"""Supplemental Point 1 proof: total-vs-per-cow loss equals the cow component."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
STAT = HERE.parent
POINT3 = STAT / "3"
TAB = HERE / "tables"
FIG = HERE / "figures"

INPUT = POINT3 / "tables" / "point3_total_manhattan_focused_decomposition.csv"
OUT = TAB / "supp_point1_total_loss_equals_negative_cows_proof.csv"
OUT_SUMMARY = TAB / "supp_point1_total_loss_equals_negative_cows_summary.csv"
OUT_SVG = FIG / "supp_point1_total_loss_equals_negative_cows.svg"
OUT_PNG = FIG / "supp_point1_total_loss_equals_negative_cows.png"


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

DOMAIN_LEVELS = [
    "Heat",
    "Cold",
    "Severe weather",
    "Forage condition",
    "Agricultural pesticides",
    "Feed market",
    "Dairy market",
    "Market demand",
    "COVID",
    "HPAI",
]
DOMAIN_LABELS = {
    "Heat": "Heat",
    "Cold": "Cold",
    "Severe weather": "Severe weather",
    "Forage condition": "Forage",
    "Agricultural pesticides": "Pesticides",
    "Feed market": "Feed market",
    "Dairy market": "Dairy market",
    "Market demand": "Market demand",
    "COVID": "COVID-19",
    "HPAI": "HPAI",
}
DOMAIN_COLORS = {
    "Heat": "#32a4b4",
    "Cold": "#33c5b2",
    "Severe weather": "#d5eada",
    "Forage condition": "#1E7A8D",
    "Agricultural pesticides": "#c79fa8",
    "Feed market": "#fbc4ab",
    "Dairy market": "#E47666",
    "Market demand": "#f09d51",
    "COVID": "#d2b48c",
    "HPAI": "#9d6b53",
}
DOMAIN_DODGE = {
    domain: (i - (len(DOMAIN_LEVELS) - 1) / 2) * 0.0032
    for i, domain in enumerate(DOMAIN_LEVELS)
}


def clean_svg(path: Path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(
        r"font: ([0-9.]+)px 'Arial'",
        r"font-size: \1px; font-family: Arial",
        svg,
    )
    svg = svg.replace("font-size: 9px;", "font-size: 9.00px;")
    svg = svg.replace("stroke: #FFFFFF; fill: #FFFFFF;", "stroke: none; fill: none;")
    svg = svg.replace("fill: #FFFFFF;", "fill: none;")
    path.write_text(svg, encoding="utf-8")


def build() -> pd.DataFrame:
    source = INPUT if INPUT.exists() else OUT
    d = pd.read_csv(source).copy()
    if "domain_plot" not in d.columns and "class_plot" in d.columns:
        d = d.rename(columns={"class_plot": "domain_plot"})
    if "mechanistic_domain_en" not in d.columns and "mechanistic_subclass_en" in d.columns:
        d = d.rename(columns={"mechanistic_subclass_en": "mechanistic_domain_en"})
    d = d[~d["domain_plot"].isin(["Dairy scale", "Herd structure / scale"])].copy()
    d["plot_domain"] = np.where(
        d["domain_plot"].eq("Pandemic shock") & d["mechanistic_domain_en"].eq("COVID"),
        "COVID",
        np.where(
            d["domain_plot"].eq("Pandemic shock") & d["mechanistic_domain_en"].eq("HPAI"),
            "HPAI",
            d["domain_plot"],
        ),
    )
    d["plot_domain"] = pd.Categorical(d["plot_domain"], categories=DOMAIN_LEVELS, ordered=True)
    d["plot_domain_label"] = d["plot_domain"].astype(str).map(DOMAIN_LABELS)
    d["signed_loss_percow_minus_total"] = d["per_cow_beta_log"] - d["total_beta_log"]
    d["negative_cow_component"] = -d["cows_beta_log"]
    d["proof_error"] = d["signed_loss_percow_minus_total"] - d["negative_cow_component"]
    d["absolute_total_attenuation"] = d["per_cow_beta_log"].abs() - d["total_beta_log"].abs()
    d["attenuated_total_vs_percow"] = d["absolute_total_attenuation"] > 0
    d["cow_component_reduces_abs_percow_signal"] = (
        np.sign(d["per_cow_beta_log"]) == -np.sign(d["cows_beta_log"])
    ) & (d["cows_beta_log"].abs() > 0)
    d["proof_group"] = np.select(
        [
            (~d["point1_total_signal"]) & d["point1_percow_signal"],
            d["point1_total_signal"],
            ~d["point1_total_signal"],
        ],
        [
            "Total no-signal, per-cow signal",
            "Total signal",
            "Total no-signal, no per-cow signal",
        ],
        default="Other",
    )
    return d


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, s in d.groupby("proof_group", dropna=False):
        rows.append(
            {
                "proof_group": group,
                "n_exposures": len(s),
                "n_attenuated_total_vs_percow": int(s["attenuated_total_vs_percow"].sum()),
                "n_cow_component_reduces_abs_percow_signal": int(
                    s["cow_component_reduces_abs_percow_signal"].sum()
                ),
                "median_signed_loss_percow_minus_total": float(s["signed_loss_percow_minus_total"].median()),
                "median_negative_cow_component": float(s["negative_cow_component"].median()),
                "max_abs_proof_error": float(s["proof_error"].abs().max()),
                "median_abs_proof_error": float(s["proof_error"].abs().median()),
            }
        )
    rows.append(
        {
            "proof_group": "all",
            "n_exposures": len(d),
            "n_attenuated_total_vs_percow": int(d["attenuated_total_vs_percow"].sum()),
            "n_cow_component_reduces_abs_percow_signal": int(
                d["cow_component_reduces_abs_percow_signal"].sum()
            ),
            "median_signed_loss_percow_minus_total": float(d["signed_loss_percow_minus_total"].median()),
            "median_negative_cow_component": float(d["negative_cow_component"].median()),
            "max_abs_proof_error": float(d["proof_error"].abs().max()),
            "median_abs_proof_error": float(d["proof_error"].abs().median()),
        }
    )
    return pd.DataFrame(rows)


def plot(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 6.1))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    lim = np.nanmax(
        np.abs(d[["negative_cow_component", "signed_loss_percow_minus_total"]].to_numpy())
    ) * 1.04
    lim = max(lim, 0.03)
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color="#222222",
        linewidth=0.6,
        linestyle="--",
        zorder=1,
    )

    def scatter_domains(target_ax, point_size: float, legend: bool = False) -> None:
        for domain in DOMAIN_LEVELS:
            s = d[d["plot_domain"].astype(str).eq(domain)]
            if s.empty:
                continue
            dodge = DOMAIN_DODGE[domain]
            target_ax.scatter(
                s["negative_cow_component"] + dodge,
                s["signed_loss_percow_minus_total"] - dodge,
                s=point_size,
                alpha=0.86,
                color=DOMAIN_COLORS[domain],
                edgecolors="white",
                linewidths=0.28,
                label=DOMAIN_LABELS[domain] if legend else None,
                zorder=3,
            )

    scatter_domains(ax, point_size=72, legend=True)
    zoom_lim = 0.08
    axins = ax.inset_axes([0.11, 0.58, 0.34, 0.32])
    axins.set_facecolor("white")
    axins.plot(
        [-zoom_lim, zoom_lim],
        [-zoom_lim, zoom_lim],
        color="#222222",
        linewidth=0.55,
        linestyle="--",
        zorder=1,
    )
    scatter_domains(axins, point_size=46, legend=False)
    axins.axhline(0, color="#888888", linewidth=0.25, zorder=0)
    axins.axvline(0, color="#888888", linewidth=0.25, zorder=0)
    axins.set_xlim(-zoom_lim, zoom_lim)
    axins.set_ylim(-zoom_lim, zoom_lim)
    axins.set_xticks([-0.05, 0, 0.05])
    axins.set_yticks([-0.05, 0, 0.05])
    axins.tick_params(axis="both", labelsize=7, length=2, width=0.25)
    axins.grid(True, color="#eeeeee", linewidth=0.16)
    axins.set_title("Zoom around zero", fontsize=7, pad=2)
    for spine in ["top", "right", "left", "bottom"]:
        axins.spines[spine].set_color("#555555")
        axins.spines[spine].set_linewidth(0.3)

    ax.axhline(0, color="#666666", linewidth=0.28, zorder=0)
    ax.axvline(0, color="#666666", linewidth=0.28, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("-β for Milk cows", fontsize=9)
    ax.set_ylabel("β for Milk per cow - β for Total production", fontsize=9)
    ax.set_title("Coefficient loss from Milk per cow to Total production equals the Milk cows component", fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.35,
        markerscale=1.0,
    )
    ax.grid(True, color="#eeeeee", linewidth=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#111111")
        ax.spines[spine].set_linewidth(0.28)
    ax.text(
        0.56,
        0.02,
        "log(total) = log(per cow) + log(cows)\n"
        "therefore β_percow - β_total = -β_cows\n"
        "points lightly dodged by class",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#222222",
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#dddddd", "alpha": 0.9},
    )
    fig.savefig(OUT_SVG, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True, facecolor="none", edgecolor="none")
    plt.close(fig)
    clean_svg(OUT_SVG)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    d = build()
    d.to_csv(OUT, index=False)
    summary = summarize(d)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot(d)
    print(f"Wrote {OUT}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_SVG}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
