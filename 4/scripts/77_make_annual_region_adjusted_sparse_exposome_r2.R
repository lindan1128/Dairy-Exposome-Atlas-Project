#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else normalizePath(".")
root <- normalizePath(file.path(dirname(script_path), "..", "..", "..", "..", ".."), mustWork = TRUE)
point_dir <- file.path(root, "analysis", "statistics", "4")
tab_dir <- file.path(point_dir, "tables")
fig_dir <- file.path(point_dir, "figures")

summary <- read_csv(
  file.path(tab_dir, "point4_annual_region_adjusted_sparse_exposome_incremental_r2.csv"),
  show_col_types = FALSE
)
contrib <- read_csv(
  file.path(tab_dir, "point4_annual_region_adjusted_sparse_exposome_selected_variable_contribution.csv"),
  show_col_types = FALSE
)

domain_order <- c(
  "Heat", "Cold", "Severe weather", "Forage", "Pesticides",
  "Feed market", "Dairy market", "Milk price", "Market demand", "Dairy scale"
)
domain_cols <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage" = "#1D7B8D",
  "Pesticides" = "#c79fa8",
  "Feed market" = "#fbc4ab",
  "Dairy market" = "#E47666",
  "Milk price" = "#E47666",
  "Market demand" = "#f09d51",
  "Dairy scale" = "#fec89a",
  "Herd scale" = "#f6a04d"
)

summary_long <- summary %>%
  mutate(
    year = as.integer(year),
    baseline = factor(
      baseline,
      levels = c(
        "Region",
        "Region + breed context",
        "Region + herd scale",
        "Region + breed context + herd scale",
        "No region",
        "No region + breed context",
        "No region + herd scale",
        "No region + breed context + herd scale"
      ),
      labels = c(
        "Beyond region",
        "Beyond region + breed context",
        "Beyond region + herd scale",
        "Beyond region + breed context + herd scale",
        "Beyond intercept",
        "Beyond breed context",
        "Beyond herd scale",
        "Beyond breed context + herd scale"
      )
    ),
    baseline_family = if_else(include_region_baseline, "Region-adjusted", "No-region"),
    incremental_r2_pct = 100 * combined_incremental_r2,
    base_r2_pct = 100 * base_model_r2,
    full_r2_pct = 100 * full_model_r2
  ) %>%
  filter(!is.na(baseline))

domain_contrib <- contrib %>%
  mutate(
    year = as.integer(year),
    baseline = factor(
      baseline,
      levels = c(
        "Region",
        "Region + breed context",
        "Region + herd scale",
        "Region + breed context + herd scale",
        "No region",
        "No region + breed context",
        "No region + herd scale",
        "No region + breed context + herd scale"
      ),
      labels = c(
        "Beyond region",
        "Beyond region + breed context",
        "Beyond region + herd scale",
        "Beyond region + breed context + herd scale",
        "Beyond intercept",
        "Beyond breed context",
        "Beyond herd scale",
        "Beyond breed context + herd scale"
      )
    ),
    baseline_family = if_else(include_region_baseline, "Region-adjusted", "No-region"),
    domain_label = factor(domain_label, levels = domain_order),
    incremental_r2_pct = 100 * pmax(selected_variable_incremental_r2, 0)
  ) %>%
  group_by(baseline, year, domain_label) %>%
  summarise(incremental_r2_pct = sum(incremental_r2_pct, na.rm = TRUE), .groups = "drop") %>%
  tidyr::complete(
    baseline,
    year = seq(min(year), max(year)),
    domain_label,
    fill = list(incremental_r2_pct = 0)
  ) %>%
  filter(!is.na(baseline), !is.na(domain_label))

selected <- contrib %>%
  filter(baseline == "Region") %>%
  mutate(
    year = as.integer(year),
    exposure_short = gsub("^daymet_dairy_weighted_", "", exposure),
    exposure_short = if_else(nchar(exposure_short) > 34, paste0(substr(exposure_short, 1, 33), "..."), exposure_short),
    label = paste0(selection_rank, ". ", exposure_short),
    incremental_r2_pct = 100 * pmax(selected_variable_incremental_r2, 0)
  ) %>%
  arrange(year, selection_rank) %>%
  filter(!is.na(year), !is.na(domain_label))

write_csv(
  domain_contrib,
  file.path(tab_dir, "point4_annual_region_adjusted_sparse_exposome_domain_contribution_plot.csv")
)

base_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(size = 9, color = "black"),
    axis.text = element_text(size = 9, color = "black"),
    axis.title = element_text(size = 9, color = "black"),
    strip.text = element_text(size = 9, color = "black"),
    strip.background = element_rect(fill = "white", color = "black", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey88", linewidth = 0.25),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3),
    legend.title = element_text(size = 9, color = "black"),
    legend.text = element_text(size = 9, color = "black"),
    legend.key.height = unit(0.16, "in"),
    legend.key.width = unit(0.24, "in"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(2, 2, 2, 2)
  )

p_r2 <- ggplot(summary_long, aes(year, incremental_r2_pct, color = baseline, group = baseline)) +
  geom_line(linewidth = 0.72) +
  geom_point(size = 1.65, shape = 21, fill = "white", stroke = 0.35) +
  scale_x_continuous(breaks = seq(2000, 2025, 5), limits = c(2000, 2025)) +
  scale_y_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.10))) +
  scale_color_manual(
    values = c(
      "Beyond region" = "#c44e52",
      "Beyond region + breed context" = "#4C93AD",
      "Beyond region + herd scale" = "#E09F3E",
      "Beyond region + breed context + herd scale" = "#2F6B4F",
      "Beyond intercept" = "#8D6E63",
      "Beyond breed context" = "#7E57C2",
      "Beyond herd scale" = "#C77DFF",
      "Beyond breed context + herd scale" = "#6D597A"
    ),
    name = "Baseline"
  ) +
  labs(x = NULL, y = "Annual incremental R² (%)") +
  base_theme +
  theme(
    axis.text.x = element_blank(),
    legend.position = c(0.02, 0.98),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = alpha("white", 0.72), color = NA)
  )

p_r2_only <- p_r2 +
  labs(x = "Year", y = "Incremental R² (%)") +
  facet_wrap(~baseline_family, ncol = 1) +
  theme(
    axis.text.x = element_text(size = 9, color = "black"),
    legend.position = "right",
    legend.justification = "center",
    legend.background = element_blank()
  )

dynamic_summary <- summary %>%
  filter(baseline == "No region + herd scale + available breed context") %>%
  mutate(
    year = as.integer(year),
    incremental_r2_pct = 100 * combined_incremental_r2,
    is_significant = combined_incremental_r2_p < 0.05,
    adjustment = case_when(
      grepl("cdcb_breed_heat_background_state_z", baseline_covariates) ~ "Dairy scale + breed context",
      TRUE ~ "Dairy scale only"
    ),
    adjustment = factor(adjustment, levels = c("Dairy scale only", "Dairy scale + breed context"))
  )

p_dynamic <- ggplot(dynamic_summary, aes(year, incremental_r2_pct)) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    color = "#7e9da2",
    fill = "#7e9da2",
    alpha = 0.22,
    linewidth = 2.72
  ) +
  geom_point(aes(fill = is_significant), shape = 21, size = 2.65, stroke = 0.35, color = "black") +
  scale_fill_manual(values = c("TRUE" = "#7e9da2", "FALSE" = "grey70"), guide = "none") +
  scale_x_continuous(breaks = seq(2000, 2025, 5), limits = c(2000, 2025)) +
  scale_y_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.08, 0.16))) +
  labs(x = "Year", y = "Incremental R² (%)") +
  theme_classic(base_size = 9) +
  theme(
    text = element_text(size = 9, color = "black"),
    axis.text = element_text(size = 9, color = "black"),
    axis.title = element_text(size = 9, color = "black"),
    legend.title = element_text(size = 9, color = "black"),
    legend.text = element_text(size = 9, color = "black"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(2, 2, 2, 2),
    axis.text.x = element_text(size = 9, color = "black"),
    legend.position = "none",
    legend.background = element_blank()
  )

p_area <- ggplot(domain_contrib, aes(year, incremental_r2_pct, fill = domain_label)) +
  geom_area(alpha = 0.88, color = "white", linewidth = 0.10) +
  geom_line(
    data = summary_long %>%
      transmute(baseline_family, baseline, year = as.integer(year), incr_pct = incremental_r2_pct),
    aes(year, incr_pct),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 0.42
  ) +
  facet_wrap(baseline_family ~ baseline, ncol = 2) +
  scale_x_continuous(breaks = seq(2000, 2025, 5), limits = c(2000, 2025)) +
  scale_y_continuous(labels = label_number(accuracy = 1), expand = expansion(mult = c(0.02, 0.08))) +
  scale_fill_manual(values = domain_cols, name = "Domain") +
  labs(x = "Year", y = "Selected-variable incremental R² (%)") +
  base_theme +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

p_selected <- selected %>%
  filter(year %in% c(2000, 2005, 2010, 2015, 2020, 2025)) %>%
  mutate(year = factor(year, levels = c(2000, 2005, 2010, 2015, 2020, 2025))) %>%
  ggplot(aes(selection_rank, incremental_r2_pct, fill = domain_label)) +
  geom_col(width = 0.72, color = "black", linewidth = 0.12) +
  facet_wrap(~year, nrow = 1) +
  scale_x_continuous(breaks = 1:5) +
  scale_y_continuous(labels = label_number(accuracy = 0.5), expand = expansion(mult = c(0, 0.08))) +
  scale_fill_manual(values = domain_cols, guide = "none") +
  labs(x = "Selected exposure rank", y = "Single-variable ΔR² (%)") +
  base_theme

ggsave(
  file.path(fig_dir, "main_point4_annual_incremental_r2_herd_breed_adjusted.svg"),
  p_dynamic,
  width = 3.9,
  height = 2.7,
  units = "in",
  bg = "transparent"
)
ggsave(
  file.path(fig_dir, "main_point4_annual_incremental_r2_herd_breed_adjusted_wo_legend.svg"),
  p_dynamic & theme(legend.position = "none"),
  width = 3.9,
  height = 2.7,
  units = "in",
  bg = "transparent"
)

message("Wrote herd/breed adjusted annual incremental R2 figures.")
