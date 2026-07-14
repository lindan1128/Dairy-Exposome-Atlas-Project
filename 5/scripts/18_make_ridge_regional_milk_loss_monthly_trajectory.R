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
region_map <- c(
  "AZ" = "West", "CA" = "West", "CO" = "West", "ID" = "West", "NM" = "West", "OR" = "West", "UT" = "West", "WA" = "West",
  "IA" = "Midwest", "IL" = "Midwest", "IN" = "Midwest", "KS" = "Midwest", "MI" = "Midwest", "MN" = "Midwest", "MO" = "Midwest", "OH" = "Midwest", "SD" = "Midwest", "WI" = "Midwest",
  "NY" = "Northeast", "PA" = "Northeast", "VT" = "Northeast",
  "FL" = "South", "GA" = "South", "KY" = "South", "TX" = "South", "VA" = "South"
)

pred <- read_csv(file.path(tab5, "point5_exposome_milk_loss_risk_predictions.csv"), show_col_types = FALSE) %>%
  mutate(
    region = unname(region_map[state_alpha]),
    region = factor(region, levels = region_order),
    date_x = year + (month - 1) / 12
  ) %>%
  filter(!is.na(region), year >= 2014, year <= 2025)

weighted_mean <- function(x, w) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  if (!any(ok)) return(NA_real_)
  weighted.mean(x[ok], w[ok])
}

monthly <- pred %>%
  group_by(region, year, month, date_x) %>%
  summarise(
    observed_loss_pct = weighted_mean(next_loss_pct, milk_cows_head),
    baseline_predicted_loss_pct = weighted_mean(baseline_predicted_loss_pct, milk_cows_head),
    exposome_predicted_loss_pct = weighted_mean(exposome_predicted_loss_pct, milk_cows_head),
    n_state_months = n(),
    n_states = n_distinct(state_alpha),
    .groups = "drop"
  )

roll12 <- function(x) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) {
    lo <- max(1, i - 11)
    out[i] <- mean(x[lo:i], na.rm = TRUE)
  }
  out
}

monthly <- monthly %>%
  arrange(region, date_x) %>%
  group_by(region) %>%
  mutate(
    observed_loss_pct = roll12(observed_loss_pct),
    baseline_predicted_loss_pct = roll12(baseline_predicted_loss_pct),
    exposome_predicted_loss_pct = roll12(exposome_predicted_loss_pct)
  ) %>%
  ungroup()

# State-resampling bootstrap intervals for the two prediction series at the same monthly resolution.
set.seed(20260709)
boot_n <- 2000
boot_weighted_ci <- function(x, w, n_boot = boot_n) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  x <- x[ok]
  w <- w[ok]
  n <- length(x)
  if (n == 0) return(c(NA_real_, NA_real_))
  if (n == 1) return(c(x[1], x[1]))
  idx <- matrix(sample.int(n, size = n * n_boot, replace = TRUE), nrow = n)
  vals <- colSums(matrix(x[idx], nrow = n) * matrix(w[idx], nrow = n)) / colSums(matrix(w[idx], nrow = n))
  as.numeric(quantile(vals, c(0.025, 0.975), na.rm = TRUE))
}

ci_df <- pred %>%
  group_by(region, year, month, date_x) %>%
  summarise(
    baseline_low = boot_weighted_ci(baseline_predicted_loss_pct, milk_cows_head)[1],
    baseline_high = boot_weighted_ci(baseline_predicted_loss_pct, milk_cows_head)[2],
    exposome_low = boot_weighted_ci(exposome_predicted_loss_pct, milk_cows_head)[1],
    exposome_high = boot_weighted_ci(exposome_predicted_loss_pct, milk_cows_head)[2],
    n_boot = boot_n,
    n_states = n_distinct(state_alpha),
    .groups = "drop"
  ) %>%
  arrange(region, date_x) %>%
  group_by(region) %>%
  mutate(
    baseline_low = roll12(baseline_low),
    baseline_high = roll12(baseline_high),
    exposome_low = roll12(exposome_low),
    exposome_high = roll12(exposome_high)
  ) %>%
  ungroup() %>%
  pivot_longer(
    cols = c(baseline_low, baseline_high, exposome_low, exposome_high),
    names_to = c("model", ".value"),
    names_pattern = "(baseline|ridge)_(low|high)"
  ) %>%
  transmute(
    region, year, month, date_x,
    series = if_else(model == "baseline", "Prediction based on phenotype history", "Prediction based on phenotype history and exposome"),
    ci_low = low,
    ci_high = high,
    n_boot,
    n_states
  ) %>%
  mutate(
    ci_low = ci_low,
    ci_high = ci_high,
    series = factor(series, levels = c("Prediction based on phenotype history", "Prediction based on phenotype history and exposome"))
  )

plot_df <- monthly %>%
  filter(year >= 2015, year <= 2025) %>%
  pivot_longer(
    cols = c(observed_loss_pct, baseline_predicted_loss_pct, exposome_predicted_loss_pct),
    names_to = "series", values_to = "loss_pct"
  ) %>%
  mutate(
    series = factor(
      series,
      levels = c("observed_loss_pct", "baseline_predicted_loss_pct", "exposome_predicted_loss_pct"),
      labels = c("Observed", "Prediction based on phenotype history", "Prediction based on phenotype history and exposome")
    )
  )

soft_scale <- function(x) {
  tanh(x)
}

plot_df <- plot_df %>%
  mutate(loss_pct = soft_scale(loss_pct))

ci_df <- ci_df %>%
  filter(year >= 2015, year <= 2025) %>%
  mutate(
    ci_low = soft_scale(ci_low),
    ci_high = soft_scale(ci_high)
  )

series_cols <- c(
  "Observed" = "#8a8a8a",
  "Prediction based on phenotype history" = "#1487CA",
  "Prediction based on phenotype history and exposome" = "#D77291"
)
series_lwd <- c(
  "Observed" = 0.50,
  "Prediction based on phenotype history" = 0.50,
  "Prediction based on phenotype history and exposome" = 0.80
)
ribbon_cols <- c(
  "Prediction based on phenotype history" = "#1487CA",
  "Prediction based on phenotype history and exposome" = "#D77291"
)

theme_point5 <- theme_bw(base_size = 9, base_family = "Helvetica") +
  theme(
    text = element_text(size = 9, color = "black"),
    axis.text = element_text(size = 9, color = "black"),
    axis.title = element_text(size = 9, color = "black"),
    strip.text = element_text(size = 9, color = "black", margin = margin(t = 4, b = 4)),
    strip.background = element_rect(fill = "#f1f1f1", color = "#54656c", linewidth = 0.28),
    legend.title = element_blank(),
    legend.text = element_text(size = 9, color = "black"),
    legend.key.width = unit(18, "pt"),
    legend.key.height = unit(7, "pt"),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.spacing = unit(14, "pt"),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    panel.border = element_rect(fill = NA, color = "#111111", linewidth = 0.28),
    axis.line = element_blank(),
    axis.ticks = element_line(linewidth = 0.22, color = "#111111"),
    axis.ticks.length = unit(4.2, "pt"),
    axis.text.x = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5, color = "#222222"),
    axis.text.y = element_text(size = 9, color = "#222222"),
    axis.title.x = element_text(size = 9, margin = margin(t = 6), color = "black"),
    plot.margin = margin(5, 6, 10, 6)
  )

make_plot <- function(show_legend = TRUE, show_x_title = TRUE) {
  p <- ggplot(plot_df, aes(date_x, loss_pct, color = series, linewidth = series)) +
    geom_ribbon(
      data = ci_df,
      aes(x = date_x, ymin = ci_low, ymax = ci_high, fill = series, group = series),
      inherit.aes = FALSE,
      alpha = 0.72,
      color = NA
    ) +
    geom_line(lineend = "butt") +
    facet_wrap(~ region, nrow = 1, axes = "all", axis.labels = "all") +
    scale_color_manual(values = series_cols, drop = FALSE) +
    scale_fill_manual(values = ribbon_cols, guide = "none") +
    scale_linewidth_manual(values = series_lwd, drop = FALSE) +
    scale_x_continuous(
      breaks = 2015:2025,
      labels = function(x) ifelse(x %in% c(2015, 2020, 2025), as.character(x), ""),
      limits = c(2015, 2025 + 11 / 12),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    scale_y_continuous(breaks = c(-1.0, 0.0, 1.0), labels = sprintf("%.1f", c(-1.0, 0.0, 1.0)), limits = c(-1, 1), expand = expansion(mult = c(0.00, 0.00))) +
    labs(x = if (show_x_title) "Year" else NULL, y = "Next-month milk-loss risk (%)") +
    guides(
      color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = c(0.60, 0.60, 0.95))),
      linewidth = "none"
    ) +
    theme_point5
  if (show_legend) {
    p <- p + theme(legend.position = "bottom", legend.margin = margin(t = 0), legend.box.margin = margin(t = -4))
  } else {
    p <- p + theme(legend.position = "none")
  }
  p
}

p <- make_plot(TRUE)
p_wo <- make_plot(FALSE, show_x_title = FALSE)

out_base <- file.path(fig5, "main_point5_exposome_milk_loss_risk_region_monthly_trajectory")
out_wo <- file.path(fig5, "main_point5_exposome_milk_loss_risk_region_monthly_trajectory_wo_legend")
ggsave(paste0(out_base, ".svg"), p, width = 7.9, height = 2.50)
ggsave(paste0(out_base, ".png"), p, width = 7.9, height = 2.50, dpi = 450)
ggsave(paste0(out_wo, ".svg"), p_wo, width = 7.9, height = 2.25)
ggsave(paste0(out_wo, ".png"), p_wo, width = 7.9, height = 2.25, dpi = 450)
message("Wrote monthly regional trajectory SVG/PNG files")
