#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)][1])))
here <- normalizePath(file.path(script_dir, ".."))
stat <- normalizePath(file.path(here, ".."))
data_tab <- normalizePath(file.path(here, "..", "..", "..", "data", "us_milk", "tables"))
fig <- file.path(here, "figures")

clean_svg <- function(path) {
  x <- readLines(path, warn = FALSE)
  x <- gsub("stroke: #FFFFFF; fill: #FFFFFF;", "stroke: none; fill: none;", x, fixed = TRUE)
  x <- gsub("fill: #FFFFFF;", "fill: none;", x, fixed = TRUE)
  writeLines(x, path)
}

annual <- read_csv(
  file.path(data_tab, "analysis_1_2_4_state_year_breed_genetic_context.csv"),
  show_col_types = FALSE
) %>%
  filter(is.finite(cdcb_dhi_breed_heat_background_z), is.finite(cdcb_k2_reported_breed_records)) %>%
  group_by(year) %>%
  summarize(
    component_year_raw = weighted.mean(
      cdcb_dhi_breed_heat_background_z,
      cdcb_k2_reported_breed_records,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  mutate(
    component_year_z = as.numeric(scale(component_year_raw)),
    component_label = factor("Breed heat background", levels = c("Breed heat background")),
    component_short = "Heat-tolerant breed composition index"
  )

line_pal <- c(
  "Breed heat background" = "#ffcb69"
)

line_labels <- annual %>%
  group_by(component_label, component_short) %>%
  filter(year == max(year)) %>%
  ungroup() %>%
  mutate(label_year = 2020)

p_base <- ggplot(annual, aes(year, component_year_z, color = component_label)) +
  geom_hline(yintercept = 0, color = "#9a9a9a", linewidth = 0.22) +
  geom_line(linewidth = 2.65, alpha = 0.95) +
  scale_color_manual(values = line_pal) +
  scale_x_continuous(
    breaks = c(2000, 2005, 2010, 2015, 2020, 2025),
    limits = c(2000, 2025),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  scale_y_continuous(breaks = c(-1, 0, 1)) +
  labs(x = NULL, y = NULL) +
  coord_cartesian(clip = "off") +
  theme_bw(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(size = 9, family = "Arial", face = "plain", color = "#222222"),
    axis.title = element_blank(),
    axis.text = element_text(size = 9, family = "Arial", color = "#222222"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#eeeeee", linewidth = 0.18),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.24),
    axis.ticks = element_line(color = "#111111", linewidth = 0.2),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(2, 8, 2, 3),
    legend.position = "none"
  )

p <- p_base +
  geom_text_repel(
    data = line_labels,
    aes(x = label_year, label = component_short),
    family = "Arial",
    size = 9 / .pt,
    nudge_x = 0,
    direction = "y",
    hjust = 0,
    segment.color = NA,
    max.overlaps = Inf,
    box.padding = 0.04,
    point.padding = 0.02
  )

out <- file.path(fig, "main_point2_breed_record_holstein_heat_trait_mini.svg")
ggsave(out, p, width = 3.25, height = 1.025, units = "in", bg = "transparent")
clean_svg(out)

out2 <- file.path(fig, "main_point2_breed_record_holstein_heat_trait_mini_wo_legend.svg")
ggsave(out2, p_base, width = 3.55, height = 1.025, units = "in", bg = "transparent")
clean_svg(out2)
