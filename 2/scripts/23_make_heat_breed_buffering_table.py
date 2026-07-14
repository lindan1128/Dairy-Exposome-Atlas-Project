#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parents[5] / "analysis" / "statistics" / "2"
TABLE_IN = ROOT / "tables" / "point2_strict_pair_heat_breed_modified_exwas_interactions.csv"
STRICT_AUDIT = ROOT / "tables" / "point2_heat_clean_full_paired_variable_audit.csv"
TABLE_OUT = ROOT / "tables" / "point2_heat_breed_buffering_matched_metrics_table.csv"
FIG_OUT = ROOT / "figures" / "supp_point2_heat_breed_buffering_matched_metrics_table.svg"


GROUP_ORDER = [
    "joint hot-days",
    "monthly intensity",
    "monthly extreme",
    "threshold days: mild",
    "threshold days: moderate",
    "threshold days: severe",
    "threshold heatload",
]

GROUP_LABEL = {
    "joint hot-days": "Joint hot-days",
    "monthly intensity": "Monthly intensity",
    "monthly extreme": "Monthly extreme",
    "threshold days: mild": "Mild threshold days",
    "threshold days: moderate": "Moderate threshold days",
    "threshold days: severe": "Severe threshold days",
    "threshold heatload": "Threshold heatload",
}


def fmt_p(value: str) -> str:
    p = float(value)
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def text_lines(text: str, width: int) -> list[str]:
    return wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


FONT_SIZE = 11.0


def add_text(parts: list[str], x: float, y: float, text: str, size: float = FONT_SIZE, anchor: str = "start",
             weight: str = "normal", fill: str = "#222222") -> None:
    parts.append(
        f"<text x='{x:.1f}' y='{y:.1f}' text-anchor='{anchor}' "
        f"style='font-size: {size:.2f}px; font-family: Arial; font-weight: {weight}; fill: {fill};'>"
        f"{html.escape(text)}</text>"
    )


def add_wrapped_center(parts: list[str], x: float, y_mid: float, text: str, width: int, line_h: float = 11.2) -> None:
    lines = text_lines(text, width)
    y_start = y_mid - (len(lines) - 1) * line_h / 2 + FONT_SIZE * 0.35
    for i, line in enumerate(lines):
        add_text(parts, x, y_start + i * line_h, line, anchor="middle")


def main() -> None:
    strict_clean: set[str] = set()
    with STRICT_AUDIT.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("in_strict_paired_heat") == "True":
                strict_clean.add(row["variables_en"])

    rows: list[dict[str, str]] = []
    with TABLE_IN.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["breed_context"] != "breed_heat_background_z" or row["status"] != "ok":
                continue
            if row["exposure"] not in strict_clean:
                continue
            rows.append(row)

    by_group: dict[str, dict[str, dict[str, str]]] = {group: {} for group in GROUP_ORDER}
    for row in rows:
        by_group[row["strict_pair_group"]][row["strict_pair_form"]] = row

    table_rows = []
    for group in GROUP_ORDER:
        if "Humid paired heat" not in by_group[group] or "Dry paired heat" not in by_group[group]:
            continue
        humid = by_group[group]["Humid paired heat"]
        dry = by_group[group]["Dry paired heat"]
        table_rows.append({
            "matched_metric": GROUP_LABEL[group],
            "humid_heat_variable": humid["strict_pair_label"],
            "humid_raw_p": fmt_p(humid["p_interaction"]),
            "humid_fdr_p": fmt_p(humid["q_bh_context"]),
            "dry_heat_variable": dry["strict_pair_label"],
            "dry_raw_p": fmt_p(dry["p_interaction"]),
            "dry_fdr_p": fmt_p(dry["q_bh_context"]),
            "humid_expected_buffering": humid["expected_buffering"],
            "dry_expected_buffering": dry["expected_buffering"],
            "humid_fdr_q05": humid["fdr_q05"],
            "dry_fdr_q05": dry["fdr_q05"],
        })

    with TABLE_OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    width = 980
    height = 360
    left = 24
    top = 28
    header_h = 42
    row_h = 34
    col_x = [left, 158, 420, 482, 552, 814, 876]
    col_w = [126, 252, 54, 62, 252, 54, 62]
    col_mid = [x + w / 2 for x, w in zip(col_x, col_w)]
    headers = [
        "Matched metric",
        "Humid heat variable",
        "Raw P",
        "FDR P",
        "Dry heat variable",
        "Raw P",
        "FDR P",
    ]

    parts: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='none'/>",
    ]
    add_text(parts, left, 18, "Supplementary Table 1. Breed-background buffering interactions for matched heat metrics")

    table_top = top
    table_bottom = table_top + header_h + row_h * len(table_rows)
    parts.append(
        f"<rect x='{left}' y='{table_top}' width='{width - 2 * left}' height='{header_h}' "
        "fill='#F2F2F2' stroke='none'/>"
    )
    parts.append(
        f"<line x1='{left}' y1='{table_top}' x2='{width - left}' y2='{table_top}' "
        "stroke='#111111' stroke-width='0.65'/>"
    )
    parts.append(
        f"<line x1='{left}' y1='{table_top + header_h}' x2='{width - left}' y2='{table_top + header_h}' "
        "stroke='#111111' stroke-width='0.65'/>"
    )
    parts.append(
        f"<line x1='{left}' y1='{table_bottom}' x2='{width - left}' y2='{table_bottom}' "
        "stroke='#111111' stroke-width='0.65'/>"
    )

    for x, header in zip(col_mid, headers):
        add_text(parts, x, table_top + 23, header, weight="normal", anchor="middle")

    for idx, row in enumerate(table_rows):
        y0 = table_top + header_h + idx * row_h
        if row["humid_fdr_q05"] == "True":
            parts.append(
                f"<rect x='{col_x[3] - 8}' y='{y0}' width='{col_w[3] + 16}' height='{row_h}' "
                "fill='#DCEFF2' stroke='none'/>"
            )
        y_mid = y0 + row_h / 2
        add_wrapped_center(parts, col_mid[0], y_mid, row["matched_metric"], 20)
        add_wrapped_center(parts, col_mid[1], y_mid, row["humid_heat_variable"], 36)
        add_text(parts, col_mid[2], y_mid + FONT_SIZE * 0.35, row["humid_raw_p"], anchor="middle")
        add_text(parts, col_mid[3], y_mid + FONT_SIZE * 0.35, row["humid_fdr_p"], anchor="middle")
        add_wrapped_center(parts, col_mid[4], y_mid, row["dry_heat_variable"], 36)
        add_text(parts, col_mid[5], y_mid + FONT_SIZE * 0.35, row["dry_raw_p"], anchor="middle")
        add_text(parts, col_mid[6], y_mid + FONT_SIZE * 0.35, row["dry_fdr_p"], anchor="middle")

    note = (
        "Raw P values are interaction-test P values from the breed-background modified ExWAS model; "
        "FDR P values are Benjamini-Hochberg adjusted within this matched heat-metric family. "
        "Blue shading marks FDR P < 0.05."
    )
    for i, line in enumerate(text_lines(note, 140)):
        add_text(parts, left, table_bottom + 19 + i * 11.2, line)
    parts.append("</svg>")
    FIG_OUT.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
