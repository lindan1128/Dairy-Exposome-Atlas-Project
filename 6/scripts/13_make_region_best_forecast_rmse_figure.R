#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grid)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else normalizePath(".")
root <- normalizePath(file.path(dirname(script_path), "..", "..", "..", "..", ".."), mustWork = TRUE)
tab <- file.path(root, "analysis/statistics/6/tables")
fig <- file.path(root, "analysis/statistics/6/figures")
dir.create(fig, recursive = TRUE, showWarnings = FALSE)

region_order <- c("South", "West", "Midwest", "Northeast")
region_strip_cols <- c(
  "South" = grDevices::adjustcolor("#D49BAC", alpha.f = 0.8),
  "West" = grDevices::adjustcolor("#B8A9C4", alpha.f = 0.8),
  "Midwest" = grDevices::adjustcolor("#EBC28A", alpha.f = 0.8),
  "Northeast" = grDevices::adjustcolor("#E79B79", alpha.f = 0.8)
)

perf <- read_csv(file.path(tab, "point6_region_best_forecast_by_state_performance.csv"), show_col_types = FALSE) %>%
  mutate(
    region = factor(region, levels = region_order),
    state_alpha = factor(state_alpha, levels = state_alpha[order(region, desc(improvement_pct))])
  )

plot_df <- perf %>%
  select(state_alpha, region, baseline_rmse, region_best_exposome_rmse) %>%
  pivot_longer(
    cols = c(baseline_rmse, region_best_exposome_rmse),
    names_to = "model",
    values_to = "rmse"
  ) %>%
  mutate(
    model = recode(
      model,
      baseline_rmse = "Prediction based on phenotype history",
      region_best_exposome_rmse = "Prediction based on phenotype history and region-level exposome"
    ),
    model = factor(
      model,
      levels = c(
        "Prediction based on phenotype history",
        "Prediction based on phenotype history and region-level exposome"
      )
    )
  )

p <- ggplot(perf, aes(x = state_alpha)) +
  geom_segment(
    aes(xend = state_alpha, y = baseline_rmse, yend = region_best_exposome_rmse),
    colour = "black",
    linewidth = 0.45,
    alpha = 0.80
  ) +
  geom_point(
    data = plot_df,
    aes(y = rmse, fill = model, shape = model),
    colour = "black",
    stroke = 0.28,
    size = ifelse(plot_df$model == "Prediction based on phenotype history and region-level exposome", 3.2, 2.2)
  ) +
  facet_grid(. ~ region, scales = "free_x", space = "free_x") +
  scale_fill_manual(
    values = c(
      "Prediction based on phenotype history" = "#1487CA",
      "Prediction based on phenotype history and region-level exposome" = "#D77291"
    ),
    name = NULL
  ) +
  scale_shape_manual(
    values = c(
      "Prediction based on phenotype history" = 24,
      "Prediction based on phenotype history and region-level exposome" = 21
    ),
    name = NULL
  ) +
  labs(x = NULL, y = "RMSE (%)") +
  guides(
    fill = guide_legend(nrow = 1, byrow = TRUE),
    shape = guide_legend(nrow = 1, byrow = TRUE)
  ) +
  theme_bw(base_size = 9, base_family = "Helvetica") +
  theme(
    text = element_text(size = 9, color = "black"),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.28),
    strip.background = element_rect(fill = "grey92", colour = "black", linewidth = 0.28),
    strip.text = element_text(size = 9, colour = "black"),
    axis.text.x = element_text(size = 9, angle = 90, vjust = 0.5, hjust = 1, colour = "#222222"),
    axis.text.y = element_text(size = 9, colour = "#222222"),
    axis.title.y = element_text(size = 9, colour = "black"),
    axis.ticks = element_line(linewidth = 0.22, color = "#111111"),
    axis.ticks.length = unit(4.2, "pt"),
    legend.position = "bottom",
    legend.text = element_text(size = 9, color = "black"),
    legend.key.width = unit(16, "pt"),
    legend.key.height = unit(7, "pt"),
    legend.background = element_blank(),
    legend.key = element_blank(),
    panel.spacing = unit(18, "pt"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(5, 6, 6, 6)
  )

base <- file.path(fig, "point6_region_best_forecast_by_state_rmse")
ggsave(paste0(base, ".svg"), p, width = 7.9, height = 2.35, bg = "transparent")
ggsave(paste0(base, ".png"), p, width = 7.9, height = 2.35, dpi = 450, bg = "transparent")

p_wo <- p + theme(legend.position = "none")
ggsave(paste0(base, "_wo_legend.svg"), p_wo, width = 7.9, height = 1.85, bg = "transparent")
ggsave(paste0(base, "_wo_legend.png"), p_wo, width = 7.9, height = 1.85, dpi = 450, bg = "transparent")

panel_widths <- perf %>%
  count(region, name = "n_states") %>%
  mutate(width = 7.9 * n_states / sum(n_states))

make_region_plot <- function(region_name) {
  region_perf <- perf %>%
    filter(region == region_name) %>%
    mutate(state_alpha = factor(as.character(state_alpha), levels = levels(state_alpha)[levels(state_alpha) %in% as.character(state_alpha)]))

  region_plot_df <- plot_df %>%
    filter(region == region_name) %>%
    mutate(state_alpha = factor(as.character(state_alpha), levels = levels(region_perf$state_alpha)))

  ggplot(region_perf, aes(x = state_alpha)) +
    geom_segment(
      aes(xend = state_alpha, y = baseline_rmse, yend = region_best_exposome_rmse),
      colour = "black",
      linewidth = 0.45,
      alpha = 0.80
    ) +
    geom_point(
      data = region_plot_df,
      aes(y = rmse, fill = model, shape = model),
      colour = "black",
      stroke = 0.28,
      size = ifelse(region_plot_df$model == "Prediction based on phenotype history and region-level exposome", 3.2, 2.2)
    ) +
    facet_grid(. ~ region, scales = "free_x", space = "free_x") +
    scale_fill_manual(
      values = c(
        "Prediction based on phenotype history" = "#1487CA",
        "Prediction based on phenotype history and region-level exposome" = "#D77291"
      ),
      name = NULL
    ) +
    scale_shape_manual(
      values = c(
        "Prediction based on phenotype history" = 24,
        "Prediction based on phenotype history and region-level exposome" = 21
      ),
      name = NULL
    ) +
    labs(x = NULL, y = "RMSE (%)") +
    theme_bw(base_size = 9, base_family = "Helvetica") +
    theme(
      text = element_text(size = 9, color = "black"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.28),
      strip.background = element_rect(fill = region_strip_cols[[region_name]], colour = "black", linewidth = 0.28),
      strip.text = element_text(size = 9, colour = "black"),
      axis.text.x = element_text(size = 9, angle = 90, vjust = 0.5, hjust = 1, colour = "#222222"),
      axis.text.y = element_text(size = 9, colour = "#222222"),
      axis.title.y = element_text(size = 9, colour = "black"),
      axis.ticks = element_line(linewidth = 0.22, color = "#111111"),
      axis.ticks.length = unit(4.2, "pt"),
      legend.position = "none",
      panel.spacing = unit(18, "pt"),
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      plot.margin = margin(5, 6, 6, 6)
    )
}

for (region_name in region_order) {
  width <- panel_widths %>% filter(region == region_name) %>% pull(width) + 1.6
  rp <- make_region_plot(region_name)
  safe_region <- tolower(region_name)
  out <- paste0(base, "_", safe_region, "_wo_legend")
  ggsave(paste0(out, ".svg"), rp, width = width, height = 1.85, bg = "transparent")
  ggsave(paste0(out, ".png"), rp, width = width, height = 1.85, dpi = 450, bg = "transparent")
}

cat("wrote point6_region_best_forecast_by_state_rmse\n")
