#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)][1])))
here <- normalizePath(file.path(script_dir, ".."))
stat <- normalizePath(file.path(here, ".."))
tab2 <- file.path(here, "tables")
fig <- file.path(here, "figures")

clean_svg <- function(path) {
  x <- readLines(path, warn = FALSE)
  x <- gsub("stroke: #FFFFFF; fill: #FFFFFF;", "stroke: none; fill: none;", x, fixed = TRUE)
  x <- gsub("fill: #FFFFFF;", "fill: none;", x, fixed = TRUE)
  writeLines(x, path)
}

strict_audit <- read_csv(
  file.path(tab2, "point2_heat_clean_full_paired_variable_audit.csv"),
  show_col_types = FALSE
) %>%
  filter(in_strict_paired_heat) %>%
  distinct(variables_en)

counts <- read_csv(
  file.path(tab2, "point2_strict_pair_heat_breed_modified_exwas_interactions.csv"),
  show_col_types = FALSE
) %>%
  filter(
    breed_context == "breed_heat_background_z",
    status == "ok",
    exposure %in% strict_audit$variables_en
  ) %>%
  group_by(strict_pair_form) %>%
  summarize(
    full = n_distinct(exposure),
    direction = sum(expected_buffering, na.rm = TRUE),
    fdr = sum(fdr_q05, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    heat_label = recode(
      strict_pair_form,
      "Humid paired heat" = "Humid heat",
      "Dry paired heat" = "Dry heat"
    ),
    heat_label = factor(heat_label, levels = c("Dry heat", "Humid heat"))
  ) %>%
  arrange(heat_label)

write_csv(counts, file.path(tab2, "point2_strict_pair_heat_breed_buffering_counts.csv"))

max_full <- max(counts$full, na.rm = TRUE)

plot_df <- bind_rows(
  counts %>% transmute(heat_label, layer = "Full", value = full, alpha = 0.3),
  counts %>% transmute(heat_label, layer = "Direction", value = direction, alpha = 0.6),
  counts %>% transmute(heat_label, layer = "FDR", value = fdr, alpha = 1.0)
) %>%
  mutate(
    layer = factor(layer, levels = c("Full", "Direction", "FDR")),
    heat_label = factor(heat_label, levels = c("Dry heat", "Humid heat"))
  )

p <- ggplot(plot_df, aes(y = heat_label, x = value, alpha = layer)) +
  geom_col(
    fill = "#6AB9C6",
    color = "#111111",
    linewidth = 0.22,
    width = 0.72,
    position = "identity"
  ) +
  scale_alpha_manual(values = c("Full" = 0.3, "Direction" = 0.6, "FDR" = 1.0), guide = "none") +
  scale_x_continuous(
    breaks = c(0, 3, max_full),
    limits = c(0, max_full + 0.25),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(size = 9, family = "Arial", face = "plain", color = "#222222"),
    axis.title = element_blank(),
    axis.text = element_text(size = 9, family = "Arial", color = "#222222"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.major.x = element_line(color = "#eeeeee", linewidth = 0.18),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.24),
    axis.ticks = element_line(color = "#111111", linewidth = 0.2),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(2, 3, 2, 3),
    legend.position = "none"
  )

out <- file.path(fig, "main_point2_strict_pair_heat_breed_buffering_mini.svg")
ggsave(out, p, width = 3.40, height = 0.875, units = "in", bg = "transparent")
clean_svg(out)
