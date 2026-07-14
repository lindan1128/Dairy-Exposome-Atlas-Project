#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
  library(grid)
})

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)][1])))
here <- normalizePath(file.path(script_dir, ".."))
tab <- file.path(here, "tables")
fig <- file.path(here, "figures")

clean_svg <- function(path) {
  x <- readLines(path, warn = FALSE)
  x <- gsub("stroke: #FFFFFF; fill: #FFFFFF;", "stroke: none; fill: none;", x, fixed = TRUE)
  x <- gsub("fill: #FFFFFF;", "fill: none;", x, fixed = TRUE)
  writeLines(x, path)
}

d <- read_csv(
  file.path(tab, "point4_annual_sparse_exposome_topk_sensitivity.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    top_k = factor(top_k, levels = 2:10, labels = paste0("K = ", 2:10)),
    is_primary = top_k == "K = 5"
  )

k_cols <- c(
  "K = 2" = "#440154",
  "K = 3" = "#472D7B",
  "K = 4" = "#3B528B",
  "K = 5" = "#CF625D",
  "K = 6" = "#21918C",
  "K = 7" = "#28AE80",
  "K = 8" = "#5EC962",
  "K = 9" = "#ADDC30",
  "K = 10" = "#D9C94E"
)

p <- ggplot(d, aes(x = year, y = incremental_r2_pct, color = top_k, group = top_k)) +
  geom_line(aes(linewidth = is_primary, alpha = is_primary), lineend = "butt") +
  scale_color_manual(values = k_cols, name = NULL) +
  scale_linewidth_manual(values = c("FALSE" = 0.38, "TRUE" = 0.90), guide = "none") +
  scale_alpha_manual(values = c("FALSE" = 0.78, "TRUE" = 1.00), guide = "none") +
  scale_x_continuous(breaks = seq(2000, 2025, 5), expand = expansion(mult = c(0.01, 0.02))) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = seq(0, 80, 10),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Top-K sensitivity of annual aggregate exposome R²",
    x = "Year",
    y = "Aggregate incremental R² (%)"
  ) +
  guides(
    color = guide_legend(
      nrow = 1,
      byrow = TRUE,
      override.aes = list(linewidth = c(rep(0.45, 3), 0.90, rep(0.45, 5)), alpha = 1)
    )
  ) +
  theme_bw(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(size = 9, family = "Arial", face = "plain", color = "#222222"),
    plot.title = element_text(size = 9, hjust = 0.5, margin = margin(b = 5)),
    axis.title = element_text(size = 9),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.text = element_text(size = 9, color = "#222222"),
    panel.grid.major = element_line(color = "#eeeeee", linewidth = 0.18),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(color = "#111111", linewidth = 0.25),
    axis.ticks = element_line(color = "#111111", linewidth = 0.22),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.justification = "center",
    legend.text = element_text(size = 9),
    legend.key.width = unit(18, "pt"),
    legend.key.height = unit(8, "pt"),
    legend.spacing.x = unit(8, "pt"),
    legend.margin = margin(t = 3, r = 0, b = 0, l = 0),
    plot.margin = margin(5, 8, 5, 8),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA)
  )

out <- file.path(fig, "supp_point4_annual_aggregate_r2_topk_sensitivity.svg")
ggsave(out, p, width = 8.5, height = 4.0, units = "in", bg = "transparent")
clean_svg(out)
