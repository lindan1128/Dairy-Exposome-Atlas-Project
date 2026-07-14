#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
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

d <- read_csv(file.path(tab, "point4_yearly_network_density_summary.csv"), show_col_types = FALSE)

plot_df <- d %>%
  transmute(
    year,
    value = n_edges_bh_fdr05
  )

p <- ggplot(plot_df, aes(x = year, y = value)) +
  geom_line(color = "#3F6C8F", linewidth = 0.48) +
  geom_point(color = "#3F6C8F", fill = "#FFFFFF", shape = 21, size = 1.65, stroke = 0.32) +
  scale_x_continuous(breaks = seq(2000, 2025, 5), expand = expansion(mult = c(0.01, 0.02))) +
  scale_y_continuous(
    labels = function(x) {
      if (max(abs(x), na.rm = TRUE) > 10) {
        label_number(accuracy = 1, big.mark = ",")(x)
      } else {
        label_number(accuracy = 0.01)(x)
      }
    }
  ) +
  labs(
    title = "BH-FDR significant exposure-exposure edges",
    x = "Year",
    y = NULL
  ) +
  theme_bw(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(size = 9, family = "Arial", face = "plain", color = "#222222"),
    plot.title = element_text(size = 9, hjust = 0.5, margin = margin(b = 5)),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 9, color = "#222222"),
    strip.text = element_text(size = 9, color = "#222222", margin = margin(t = 3, b = 3)),
    strip.background = element_rect(fill = "#F2F2F2", color = "#A8A8A8", linewidth = 0.28),
    panel.grid.major = element_line(color = "#eeeeee", linewidth = 0.18),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.25),
    axis.ticks = element_line(color = "#111111", linewidth = 0.22),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(5, 6, 5, 6),
    legend.position = "none"
  )

out <- file.path(fig, "supp_point4_yearly_network_density_summary.svg")
ggsave(out, p, width = 6.3, height = 2.3, units = "in", bg = "transparent")
clean_svg(out)
