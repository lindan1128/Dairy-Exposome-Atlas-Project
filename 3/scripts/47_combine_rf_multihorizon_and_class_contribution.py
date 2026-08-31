#!/usr/bin/env python3
"""Combine RF multihorizon model and regional class contribution SVG panels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import shutil

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
FIGURES = HERE / "figures"
SCRIPTS = HERE / "scripts"
TMP_FIGURES = Path("/private/tmp/point3_rf_combined_components")


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_components() -> Path:
    if TMP_FIGURES.exists():
        shutil.rmtree(TMP_FIGURES)
    TMP_FIGURES.mkdir(parents=True)

    rf_model = load_script(SCRIPTS / "41_run_nested_point4aligned_multihorizon_rf.py", "point3_rf_model")
    rf_model.FIG = TMP_FIGURES
    performance = pd.read_csv(rf_model.TAB / "supp_point3_rf_multihorizon_performance_by_year_scope.csv")
    rf_model.make_figure(performance, show_legend=True)
    rf_model.make_figure(performance, show_legend=False)

    rf_domain = load_script(SCRIPTS / "44_plot_rf_regional_class_contribution.py", "point3_rf_domain")
    rf_domain.FIGURES = TMP_FIGURES
    _, phase_summary = rf_domain.summarize()
    rf_domain.plot(phase_summary, legend=True)
    rf_domain.plot(phase_summary, legend=False)
    return TMP_FIGURES


def read_svg(path: Path) -> tuple[float, float, str]:
    svg = path.read_text(encoding="utf-8")
    match = re.search(
        r"<svg[^>]*width=['\"]([0-9.]+)pt['\"][^>]*height=['\"]([0-9.]+)pt['\"][^>]*viewBox=['\"]([^'\"]+)['\"]",
        svg,
    )
    if not match:
        raise RuntimeError(f"Could not parse SVG size from {path}")
    width, height = float(match.group(1)), float(match.group(2))
    content = re.sub(r"^.*?<svg[^>]*>", "", svg, count=1, flags=re.S)
    content = re.sub(r"</svg>\s*$", "", content, count=1, flags=re.S)
    return width, height, content


def prefix_svg_ids(content: str, prefix: str) -> str:
    ids = re.findall(r'id="([^"]+)"', content)
    for old in sorted(set(ids), key=len, reverse=True):
        new = f"{prefix}{old}"
        content = content.replace(f'id="{old}"', f'id="{new}"')
        content = content.replace(f'href="#{old}"', f'href="#{new}"')
        content = content.replace(f'xlink:href="#{old}"', f'xlink:href="#{new}"')
        content = content.replace(f'url(#{old})', f'url(#{new})')
    return content


def combine(source_dir: Path, top_name: str, bottom_name: str, output_name: str) -> None:
    top_width, top_height, top_content = read_svg(source_dir / top_name)
    bottom_width, bottom_height, bottom_content = read_svg(source_dir / bottom_name)
    top_content = prefix_svg_ids(top_content, "a_")
    bottom_content = prefix_svg_ids(bottom_content, "b_")
    panel_gap = 22.0
    label_size = 13
    width = max(top_width, bottom_width)
    top_scale = width / top_width
    top_display_height = top_height * top_scale
    height = top_display_height + panel_gap + bottom_height
    top_x = 0.0
    bottom_x = (width - bottom_width) / 2
    bottom_y = top_display_height + panel_gap

    svg = f"""<?xml version="1.0" encoding="utf-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width:.6g}pt" height="{height:.6g}pt" viewBox="0 0 {width:.6g} {height:.6g}" version="1.1">
  <g id="panel-a" transform="translate({top_x:.6g}, 0) scale({top_scale:.8g})">
{top_content}
  </g>
  <g id="panel-b" transform="translate({bottom_x:.6g}, {bottom_y:.6g})">
{bottom_content}
  </g>
  <text x="0" y="{label_size}" style="fill: #222222; font-size: {label_size}px; font-family: 'Arial'">a</text>
  <text x="0" y="{bottom_y + label_size:.6g}" style="fill: #222222; font-size: {label_size}px; font-family: 'Arial'">b</text>
</svg>
"""
    (FIGURES / output_name).write_text(svg, encoding="utf-8")


def main() -> None:
    source_dir = render_components()
    combine(
        source_dir,
        "supp_point3_rf_multihorizon_model_comparison.svg",
        "supp_point3_rf_regional_class_contribution_by_horizon.svg",
        "supp_point3_rf_multihorizon_model_and_regional_class_contribution.svg",
    )
    combine(
        source_dir,
        "supp_point3_rf_multihorizon_model_comparison_wo_legend.svg",
        "supp_point3_rf_regional_class_contribution_by_horizon_wo_legend.svg",
        "supp_point3_rf_multihorizon_model_and_regional_class_contribution_wo_legend.svg",
    )
    shutil.rmtree(source_dir)


if __name__ == "__main__":
    main()
