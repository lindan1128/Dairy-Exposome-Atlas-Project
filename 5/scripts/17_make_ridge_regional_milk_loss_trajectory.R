#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else normalizePath(".")
root <- normalizePath(file.path(dirname(script_path), "..", "..", "..", "..", ".."), mustWork = TRUE)
tab5 <- file.path(root, "analysis/statistics/5/tables")
fig5 <- file.path(root, "analysis/statistics/5/figures")

region_order <- c("South", "West", "Midwest", "Northeast")
region_letters <- c("South" = "a", "West" = "b", "Midwest" = "c", "Northeast" = "d")

annual <- read_csv(file.path(tab5, "point5_exposome_milk_loss_risk_region_annual_trajectory.csv"), show_col_types = FALSE)
perf <- read_csv(file.path(tab5, "point5_exposome_milk_loss_risk_region_annual_trajectory_performance.csv"), show_col_types = FALSE)
ci_df <- read_csv(file.path(tab5, "point5_exposome_milk_loss_risk_region_annual_prediction_ci.csv"), show_col_types = FALSE) %>%
  mutate(
    region = factor(region, levels = region_order),
    series = recode(series,
      "Baseline" = "Prediction based on phenotype history",
      "Prediction based on phenotype history and exposome" = "Prediction based on phenotype history and exposome"
    ),
    series = factor(series, levels = c("Prediction based on phenotype history", "Prediction based on phenotype history and exposome"))
  )

plot_df <- annual %>%
  pivot_longer(
    cols = c(observed_loss_pct, baseline_predicted_loss_pct, exposome_predicted_loss_pct),
    names_to = "series", values_to = "loss_pct"
  ) %>%
  mutate(
    region = factor(region, levels = region_order),
    series = factor(
      series,
      levels = c("observed_loss_pct", "baseline_predicted_loss_pct", "exposome_predicted_loss_pct"),
      labels = c("Observed", "Prediction based on phenotype history", "Prediction based on phenotype history and exposome")
    )
  )

panel_lab <- perf %>%
  mutate(
    region = factor(region, levels = region_order),
    panel = region_letters[as.character(region)],
    delta_label = sprintf("Delta RMSE = %.2f", annual_delta_rmse_baseline_minus_exposome)
  )

metric_df <- panel_lab %>%
  transmute(region, x = 2015.1, y = min(plot_df$loss_pct, na.rm = TRUE) - 0.04, delta_label)

series_cols <- c("Observed" = "#8a8a8a", "Prediction based on phenotype history" = "#1487CA", "Prediction based on phenotype history and exposome" = "#D77291")
series_lty <- c("Observed" = "solid", "Prediction based on phenotype history" = "solid", "Prediction based on phenotype history and exposome" = "solid")
series_lwd <- c("Observed" = 0.75, "Prediction based on phenotype history" = 0.75, "Prediction based on phenotype history and exposome" = 1.25)
ribbon_cols <- c("Prediction based on phenotype history" = "#1487CA", "Prediction based on phenotype history and exposome" = "#D77291")

theme_point5 <- theme_classic(base_size = 9, base_family = "Helvetica") +
  theme(
    text = element_text(size = 9, color = "black"),
    axis.text = element_text(size = 9, color = "black"),
    axis.title = element_text(size = 9, color = "black"),
    strip.text = element_blank(),
    strip.background = element_blank(),
    legend.title = element_blank(),
    legend.text = element_text(size = 9, color = "black"),
    legend.key.width = unit(16, "pt"),
    legend.key.height = unit(7, "pt"),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.spacing = unit(18, "pt"),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    axis.line = element_line(linewidth = 0.28, color = "#111111"),
    axis.ticks = element_line(linewidth = 0.22, color = "#111111"),
    axis.ticks.length = unit(4.2, "pt"),
    axis.text.x = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5, color = "#222222"),
    axis.text.y = element_text(size = 9, color = "#222222"),
    axis.title.x = element_text(size = 9, margin = margin(t = 6), color = "black"),
    plot.margin = margin(5, 6, 6, 6)
  )

make_plot <- function(show_legend = TRUE, show_zero = TRUE, show_x_title = TRUE, show_delta = TRUE) {
  p <- ggplot(plot_df, aes(year, loss_pct, color = series, linetype = series, linewidth = series)) +
    geom_ribbon(
      data = ci_df,
      aes(x = year, ymin = ci_low, ymax = ci_high, fill = series, group = series),
      inherit.aes = FALSE,
      alpha = 0.82,
      color = NA
    ) +
    geom_line(lineend = "butt") +
    facet_wrap(~ region, nrow = 1, axes = "all", axis.labels = "all") +
    scale_color_manual(values = series_cols, drop = FALSE) +
    scale_fill_manual(values = ribbon_cols, guide = "none") +
    scale_linetype_manual(values = series_lty, drop = FALSE) +
    scale_linewidth_manual(values = series_lwd, drop = FALSE) +
    scale_x_continuous(
      breaks = 2015:2025,
      labels = function(x) ifelse(x %in% c(2015, 2020, 2025), as.character(x), ""),
      expand = expansion(mult = c(0.03, 0.03))
    ) +
    scale_y_continuous(breaks = c(-1.0, 0.0, 1.0), labels = sprintf("%.1f", c(-1.0, 0.0, 1.0)), limits = c(-1.65, 1.55), expand = expansion(mult = c(0.00, 0.00))) +
    labs(x = if (show_x_title) "Year" else NULL, y = "Next-month milk-loss risk (%)") +
    guides(
      color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = c(0.75, 0.75, 1.25))),
      linetype = "none",
      linewidth = "none"
    ) +
    theme_point5
  if (show_delta) {
    p <- p + geom_text(
      data = metric_df,
      aes(x = x, y = y, label = delta_label),
      inherit.aes = FALSE,
      hjust = 0, vjust = 0,
      size = 8 / .pt,
      color = "black"
    )
  }
  if (show_zero) {
    p <- p + geom_hline(yintercept = 0, color = "#9a9a9a", linewidth = 0.55, linetype = "dashed")
  }
  if (show_legend) {
    p <- p + theme(legend.position = "bottom", legend.margin = margin(t = 0), legend.box.margin = margin(t = -4))
  } else {
    p <- p + theme(legend.position = "none")
  }
  p
}

p <- make_plot(TRUE)
p_wo <- make_plot(FALSE, show_zero = TRUE, show_x_title = FALSE, show_delta = FALSE)

out_base <- file.path(fig5, "main_point5_exposome_milk_loss_risk_region_trajectory")
out_wo <- file.path(fig5, "main_point5_exposome_milk_loss_risk_region_trajectory_wo_legend")
ggsave(paste0(out_base, ".svg"), p, width = 7.9, height = 2.35)
ggsave(paste0(out_base, ".png"), p, width = 7.9, height = 2.35, dpi = 450, bg = "transparent")
ggsave(paste0(out_wo, ".svg"), p_wo, width = 7.9, height = 1.85)
ggsave(paste0(out_wo, ".png"), p_wo, width = 7.9, height = 1.85, dpi = 450, bg = "transparent")
message("Wrote annual regional trajectory SVG/PNG files")
