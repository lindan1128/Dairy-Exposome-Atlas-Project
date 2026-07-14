#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
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

domain_order <- c(
  "Heat", "Cold", "Severe weather",
  "Forage", "Feed market", "Market demand", "Dairy market"
)

imp <- read_csv(
  file.path(tab, "point6_point5consistent_state_raw_feature_permutation_rawp30_domain_summary.csv"),
  show_col_types = FALSE
) %>%
  filter(domain_label %in% domain_order, min_p_value < 0.30, mean_importance_rmse > 0)

zmax <- max(imp$mean_importance_rmse, na.rm = TRUE)
if (!is.finite(zmax) || zmax <= 0) zmax <- 1

height <- 4.6
radius <- 0.20
theta <- seq(0, 2 * pi, length.out = 25)

cone <- bind_rows(lapply(seq_len(length(theta) - 1), function(i) {
  tibble(
    group = i,
    x = c(radius * cos(theta[i]), radius * cos(theta[i + 1]), 0),
    y = c(0, 0, height)
  )
}))

p <- ggplot() +
  geom_polygon(
    data = cone,
    aes(x = x, y = y, group = group),
    fill = "#7f7f7f",
    color = NA,
    alpha = 1
  ) +
  annotate(
    "text",
    x = 0.46,
    y = height * 0.68,
    label = sprintf("%.2f", zmax),
    hjust = 0,
    vjust = 0.5,
    size = 8.8 / .pt,
    color = "white",
    family = "Helvetica"
  ) +
  annotate(
    "text",
    x = 0.46,
    y = height * 0.68,
    label = sprintf("%.2f", zmax),
    hjust = 0,
    vjust = 0.5,
    size = 8 / .pt,
    color = "black",
    family = "Helvetica"
  ) +
  annotate(
    "text",
    x = 0.33,
    y = 0,
    label = "0",
    hjust = 0,
    vjust = 0.5,
    size = 8.8 / .pt,
    color = "white",
    family = "Helvetica"
  ) +
  annotate(
    "text",
    x = 0.33,
    y = 0,
    label = "0",
    hjust = 0,
    vjust = 0.5,
    size = 8 / .pt,
    color = "black",
    family = "Helvetica"
  ) +
  coord_equal(xlim = c(-0.28, 1.22), ylim = c(-0.08, height + 0.08), expand = FALSE, clip = "off") +
  theme_void(base_family = "Helvetica") +
  theme(
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(0, 0, 0, 0)
  )

out <- file.path(fig, "point6_point5consistent_cone_scale_legend")
ggsave(paste0(out, ".svg"), p, width = 0.9, height = 2.2, bg = "transparent")
ggsave(paste0(out, ".png"), p, width = 0.9, height = 2.2, dpi = 600, bg = "transparent")

cat("wrote point6_point5consistent_cone_scale_legend\n")
