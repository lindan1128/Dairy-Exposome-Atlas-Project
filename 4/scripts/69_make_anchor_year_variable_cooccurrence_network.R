#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
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

trailing <- commandArgs(trailingOnly = TRUE)
variant_arg <- grep("^--variant=", trailing, value = TRUE)
variant <- if (length(variant_arg)) sub("^--variant=", "", variant_arg[1]) else "yearly_pruned_union"
suffix <- if (variant == "pooled_pruned") "" else paste0("_", variant)

nodes <- read_csv(file.path(tab_dir, paste0("point4_anchor_year_variable_network", suffix, "_nodes.csv")), show_col_types = FALSE)
edges <- read_csv(file.path(tab_dir, paste0("point4_anchor_year_variable_network", suffix, "_edges_plot_backbone.csv")), show_col_types = FALSE)
summary <- read_csv(file.path(tab_dir, paste0("point4_anchor_year_variable_network", suffix, "_summary.csv")), show_col_types = FALSE)

anchor_years <- c(2000, 2010, 2025)
domain_order <- c(
  "Heat", "Cold", "Severe weather", "Forage", "Pesticides",
  "Feed market", "Dairy market", "Market demand", "Dairy scale"
)
domain_cols <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage" = "#1D7B8D",
  "Pesticides" = "#c79fa8",
  "Feed market" = "#fbc4ab",
  "Dairy market" = "#E47666",
  "Market demand" = "#f09d51",
  "Dairy scale" = "#fec89a",
  "Herd scale" = "#f6a04d"
)
class_cols <- c(
  "Nature and climate" = "#60BFA4",
  "Forage and pasture condition" = "#1D7B8D",
  "Chemical and pollution exposome" = "#c79fa8",
  "Market and production-system" = "#F06F26"
)

nodes2 <- nodes %>%
  filter(year %in% anchor_years) %>%
  filter(plot_backbone %in% TRUE) %>%
  mutate(
    year = factor(year, levels = anchor_years),
    domain_label = factor(domain_label, levels = domain_order),
    assoc_direction = case_when(
      exwas_direction == "Negative milk association" ~ "Negative milk association",
      exwas_direction == "Positive milk association" ~ "Positive milk association",
      TRUE ~ "No/unstable milk association"
    ),
    assoc_direction = factor(
      assoc_direction,
      levels = c("Negative milk association", "Positive milk association", "No/unstable milk association")
    ),
    single_variable_r2_pct = if_else(is.finite(single_variable_r2_pct), pmax(single_variable_r2_pct, 0), 0),
    source_class_plot = case_when(
      domain_label %in% c("Heat", "Cold", "Severe weather") ~ "Nature and climate",
      domain_label == "Forage" ~ "Forage and pasture condition",
      domain_label == "Pesticides" ~ "Chemical and pollution exposome",
      domain_label %in% c("Feed market", "Dairy market", "Market demand", "Dairy scale", "Herd scale") ~ "Market and production-system",
      TRUE ~ as.character(source_class)
    )
  )

# Place exposure variables on the outer ring, ordered by domain and contribution.
# The milk phenotype sits at the center. This keeps exposure-milk associations
# visually distinct from exposure-exposure co-occurrence edges.
nodes2 <- nodes2 %>%
  arrange(year, domain_label, subdomain_label, desc(single_variable_r2_pct), exposure) %>%
  group_by(year) %>%
  mutate(
    ring_i = row_number(),
    ring_n = n(),
    theta_step = 2 * pi / ring_n,
    theta = pi / 2 - 2 * pi * (ring_i - 1) / ring_n,
    theta_left = theta + theta_step / 2,
    theta_right = theta - theta_step / 2,
    x = 1.08 * cos(theta),
    y = 1.08 * sin(theta)
  ) %>%
  ungroup()

node_pos <- nodes2 %>% select(year, exposure, x, y)
edges2 <- edges %>%
  filter(year %in% anchor_years) %>%
  mutate(
    year = factor(year, levels = anchor_years),
    edge_direction = factor(edge_direction, levels = c("Positive co-occurrence", "Negative co-occurrence")),
    abs_r = abs(spearman_r)
  ) %>%
  left_join(node_pos %>% rename(exposure_a = exposure, x_a = x, y_a = y), by = c("year", "exposure_a")) %>%
  left_join(node_pos %>% rename(exposure_b = exposure, x_b = x, y_b = y), by = c("year", "exposure_b")) %>%
  filter(is.finite(x_a), is.finite(x_b))

milk_edges <- nodes2 %>%
  filter(is.finite(beta), is.finite(p), p < 0.05) %>%
  mutate(
    milk_edge_direction = factor(
      if_else(beta < 0, "Negative milk association", "Positive milk association"),
      levels = c("Negative milk association", "Positive milk association")
    ),
    milk_edge_width = pmin(exwas_neglogp, 12)
  )

milk_nodes <- tibble::tibble(
  year = factor(anchor_years, levels = anchor_years),
  x = 0,
  y = 0,
  label = "Milk\nproduction"
)

outer_class_ring_base <- nodes2 %>%
  filter(!is.na(source_class_plot)) %>%
  mutate(
    source_class_plot = factor(
      source_class_plot,
      levels = c(
        "Nature and climate",
        "Forage and pasture condition",
        "Chemical and pollution exposome",
        "Market and production-system"
      )
    ),
    source_class_plot = as.character(source_class_plot)
  ) %>%
  filter(!is.na(source_class_plot)) %>%
  group_by(year, source_class_plot) %>%
  summarise(
    i_start = min(ring_i, na.rm = TRUE),
    i_end = max(ring_i, na.rm = TRUE),
    ring_n = first(ring_n),
    .groups = "drop"
  ) %>%
  mutate(
    theta_start = pi / 2 - 2 * pi * (i_start - 1.5) / ring_n,
    theta_end = pi / 2 - 2 * pi * (i_end - 0.5) / ring_n
  )

outer_class_ring <- outer_class_ring_base %>%
  group_by(year, source_class_plot) %>%
  group_modify(~{
    theta_outer <- seq(.x$theta_start, .x$theta_end, length.out = 80)
    theta_inner <- rev(theta_outer)
    r_outer <- 1.46
    r_inner <- 1.34
    tibble::tibble(
      source_class_plot = .x$source_class_plot,
      x = c(r_outer * cos(theta_outer), r_inner * cos(theta_inner)),
      y = c(r_outer * sin(theta_outer), r_inner * sin(theta_inner)),
      point_order = seq_len(160)
    )
  }) %>%
  ungroup()

labels <- summary %>%
  filter(year %in% anchor_years) %>%
  mutate(
    year = factor(year, levels = anchor_years),
    label = paste0("n=", n_states, "; BH-FDR edges=", n_edges_bh_fdr05)
  )

theme_net <- theme_void(base_size = 9) +
  theme(
    text = element_text(size = 9, color = "black"),
    strip.text = element_blank(),
    strip.background = element_blank(),
    legend.title = element_text(size = 8, color = "black"),
    legend.text = element_text(size = 8, color = "black"),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    panel.spacing = unit(0.05, "in"),
    plot.margin = margin(1, 1, 1, 1)
  )

node_size_limits <- c(0, max(nodes2$single_variable_r2_pct, na.rm = TRUE))
line_width_limits <- c(0, max(c(edges2$abs_r, milk_edges$milk_edge_width), na.rm = TRUE))

make_network_plot <- function(plot_year, show_center_label = TRUE) {
  year_factor <- factor(plot_year, levels = anchor_years)
  p <- ggplot() +
    geom_polygon(
      data = outer_class_ring %>% filter(year == year_factor),
      aes(x = x, y = y, group = source_class_plot, fill = source_class_plot),
      inherit.aes = FALSE,
      alpha = 1,
      color = "white",
      linewidth = 0.12,
      show.legend = FALSE
    ) +
    geom_curve(
      data = edges2 %>% filter(year == year_factor),
      aes(
        x = x_a, y = y_a, xend = x_b, yend = y_b,
        color = edge_direction,
        linewidth = abs_r
      ),
      curvature = 0.08,
      alpha = 0.42,
      lineend = "round"
    ) +
    geom_segment(
      data = milk_edges %>% filter(year == year_factor),
      aes(
        x = 0, y = 0, xend = x, yend = y,
        color = milk_edge_direction,
        linewidth = milk_edge_width
      ),
      alpha = 0.62,
      lineend = "round"
    ) +
    geom_point(
      data = milk_nodes %>% filter(year == year_factor),
      aes(x, y),
      shape = 21,
      fill = "white",
      color = "black",
      stroke = 0.35,
      size = 7.6,
      inherit.aes = FALSE
    )
  if (isTRUE(show_center_label)) {
    p <- p +
      geom_text(
      data = milk_nodes %>% filter(year == year_factor),
      aes(x, y, label = label),
      size = 3.3,
      lineheight = 0.82,
      inherit.aes = FALSE
      )
  }
  p +
    geom_point(
      data = nodes2 %>% filter(year == year_factor),
      aes(x, y, size = single_variable_r2_pct, fill = domain_label, color = assoc_direction),
      shape = 21,
      stroke = 0.34,
      alpha = 0.92
    ) +
    coord_equal(xlim = c(-1.55, 1.55), ylim = c(-1.53, 1.53), clip = "off") +
    scale_fill_manual(values = c(domain_cols, class_cols), name = "Domain") +
    scale_color_manual(
      values = c(
        "Positive co-occurrence" = "#B1C99C",
        "Negative co-occurrence" = "#FFCC8D",
        "Negative milk association" = "#FFCC8D",
        "Positive milk association" = "#B1C99C",
        "No/unstable milk association" = "grey55"
      ),
      breaks = c("Positive co-occurrence", "Negative co-occurrence", "Negative milk association", "Positive milk association"),
      name = "Association"
    ) +
    scale_size_area(
      max_size = 7.2,
      limits = node_size_limits,
      breaks = c(5, 10, 25, 50),
      name = "Adjusted single-exposure\nincremental R² (%)"
    ) +
    scale_linewidth(range = c(0.18, 1.10), limits = line_width_limits, guide = "none") +
    guides(
      fill = guide_legend(override.aes = list(size = 4, color = "black"), order = 1),
      size = guide_legend(order = 2),
      color = guide_legend(order = 3, override.aes = list(fill = "white", linewidth = 0.8, size = 3))
    ) +
    theme_net
}

for (plot_year in anchor_years) {
  p_year <- make_network_plot(plot_year, show_center_label = TRUE)
  p_year_wo_legend <- make_network_plot(plot_year, show_center_label = FALSE) +
    theme(legend.position = "none")
  ggsave(
    file.path(fig_dir, paste0("main_point4_anchor_year_variable_cooccurrence_network", suffix, "_", plot_year, ".svg")),
    p_year,
    width = 3.9,
    height = 2.55,
    units = "in",
    bg = "transparent"
  )
  ggsave(
    file.path(fig_dir, paste0("main_point4_anchor_year_variable_cooccurrence_network", suffix, "_", plot_year, "_wo_legend.svg")),
    p_year_wo_legend,
    width = 3.9,
    height = 2.55,
    units = "in",
    bg = "transparent"
  )
}

legend_subdomains <- nodes2 %>%
  distinct(domain_label, subdomain_label) %>%
  arrange(domain_label, subdomain_label) %>%
  mutate(
    legend_i = row_number(),
    legend_col = ((legend_i - 1) %/% 9) + 1,
    legend_row = 9 - ((legend_i - 1) %% 9),
    x = (legend_col - 1) * 2.15,
    y = legend_row,
    domain_label = as.character(domain_label)
  )

legend_size <- tibble::tibble(
  x = 7.25,
  y = c(7.2, 6.0, 4.8, 3.6),
  contribution = c(5, 10, 25, 50),
  label = c("5", "10", "25", "50")
)

p_legend <- ggplot() +
  geom_text(
    data = tibble::tibble(x = 0, y = 9.25, label = "Exposure subdomain"),
    aes(x = x, y = y, label = label),
    hjust = 0,
    family = "Arial",
    size = 3.0,
    color = "black"
  ) +
  geom_point(
    data = legend_subdomains,
    aes(x = x, y = y, fill = domain_label),
    shape = 21,
    size = 3.2,
    color = "black",
    stroke = 0.25,
    show.legend = FALSE
  ) +
  geom_text(
    data = legend_subdomains,
    aes(x = x + 0.16, y = y, label = subdomain_label),
    hjust = 0,
    family = "Arial",
    size = 3.0,
    color = "black"
  ) +
  geom_text(
    data = tibble::tibble(
      x = 7.25,
      y = 9.25,
      label = "Adjusted single-exposure\nincremental R² (%)"
    ),
    aes(x = x, y = y, label = label),
    hjust = 0,
    lineheight = 0.9,
    family = "Arial",
    size = 3.0,
    color = "black"
  ) +
  geom_point(
    data = legend_size,
    aes(x = x + 0.18, y = y, size = contribution),
    shape = 21,
    fill = "grey85",
    color = "black",
    stroke = 0.28,
    show.legend = FALSE
  ) +
  geom_text(
    data = legend_size,
    aes(x = x + 0.55, y = y, label = label),
    hjust = 0,
    family = "Arial",
    size = 3.0,
    color = "black"
  ) +
  scale_fill_manual(values = domain_cols) +
  scale_size_area(max_size = 7.2, limits = node_size_limits) +
  coord_cartesian(xlim = c(-0.1, 8.9), ylim = c(0.45, 9.55), clip = "off") +
  theme_void(base_size = 9) +
  theme(
    text = element_text(family = "Arial", size = 9, color = "black"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(2, 2, 2, 2)
  )

ggsave(
  file.path(fig_dir, paste0("main_point4_anchor_year_variable_cooccurrence_network", suffix, "_legend.svg")),
  p_legend,
  width = 6.8,
  height = 2.7,
  units = "in",
  bg = "transparent"
)

message("Wrote anchor-year variable co-occurrence network figures: ", variant)
