#!/usr/bin/env Rscript
# Radial fan heatmap for true permutation-test raw-P10-filtered regional ridge importance.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(grid)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else normalizePath(".")
root <- normalizePath(file.path(dirname(script_path), "..", "..", "..", "..", ".."), mustWork = TRUE)
tab5 <- file.path(root, "analysis/statistics/5/tables")
fig5 <- file.path(root, "analysis/statistics/5/figures")

region_order <- c("South", "West", "Midwest", "Northeast")
domain_order <- c("Heat", "Cold", "Severe weather", "Forage", "Pesticides", "Feed market", "Dairy market", "Market demand")
domain_cols <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage" = "#1E7A8D",
  "Pesticides" = "#c79fa8",
  "Feed market" = "#fbc4ab",
  "Dairy market" = "#E47666",
  "Market demand" = "#f09d51"
)

long <- read_csv(file.path(tab5, "point5_exposome_milk_loss_risk_region_variable_permutation_rawp10_importance_long.csv"), show_col_types = FALSE)
union <- read_csv(file.path(tab5, "point5_exposome_milk_loss_risk_region_variable_permutation_rawp10_union.csv"), show_col_types = FALSE)

class_order <- c("Nature and climate", "Forage and pasture condition", "Chemical and pollution exposome", "Market and production-system")
class_cols <- c(
  "Nature and climate" = "#60BFA4",
  "Market and production-system" = "#F06F26",
  "Forage and pasture condition" = "#1E7A8D",
  "Chemical and pollution exposome" = "#c79fa8"
)

label_node <- function(x) {
  x <- gsub("^daymet_dairy_weighted_", "", x)
  x <- gsub("^daymet_", "", x)
  x <- gsub("^chem_pesticide_", "", x)
  x <- gsub("^storm_", "", x)
  x <- gsub("^market_", "", x)
  x <- gsub("^forage_", "", x)
  x <- gsub("_", " ", x)
  x <- gsub("thi", "THI", x, ignore.case = TRUE)
  x <- gsub("tmax", "Tmax", x, ignore.case = TRUE)
  x <- gsub("tmin", "Tmin", x, ignore.case = TRUE)
  x
}

union <- union %>%
  mutate(
    domain_label = factor(domain_label, levels = domain_order),
    source_class = factor(source_class, levels = class_order)
  ) %>%
  arrange(source_class, domain_label, desc(max_rawp10_importance), exposure) %>%
  mutate(node_id = row_number(), node_label = label_node(exposure))

plot_df <- long %>%
  select(region, test_year, exposure, rawp10_importance, p_value, feature, model_context, added_domain) %>%
  inner_join(
    union %>% select(exposure, domain_label, subdomain_label, source_class, mechanistic_domain_en, node_id),
    by = "exposure"
  ) %>%
  mutate(region = factor(region, levels = region_order), year = as.integer(test_year))

n_nodes <- nrow(union)
years <- sort(unique(plot_df$year))
year_id <- setNames(seq_along(years), years)
plot_df <- plot_df %>% mutate(year_id = year_id[as.character(year)])

vmax <- as.numeric(quantile(plot_df$rawp10_importance[plot_df$rawp10_importance > 0], 0.85, na.rm = TRUE))
if (!is.finite(vmax) || vmax <= 0) vmax <- max(plot_df$rawp10_importance, na.rm = TRUE)
plot_df$plot_support <- pmin(plot_df$rawp10_importance, vmax)

# Fan geometry. Angles span a 170-degree fan, leaving a handle below.
theta_min <- 205 * pi / 180
theta_max <- -25 * pi / 180
node_width <- (theta_max - theta_min) / n_nodes
r_inner <- 0.50
ring_width <- 0.075
r_outer <- r_inner + length(years) * ring_width

make_tile <- function(theta1, theta2, r1, r2, n = 10) {
  th_top <- seq(theta1, theta2, length.out = n)
  th_bot <- seq(theta2, theta1, length.out = n)
  data.frame(
    x = c(r2 * cos(th_top), r1 * cos(th_bot)),
    y = c(r2 * sin(th_top), r1 * sin(th_bot))
  )
}

tile_poly <- plot_df %>%
  rowwise() %>%
  do({
    theta1 <- theta_min + (.$node_id - 1) * node_width
    theta2 <- theta_min + .$node_id * node_width
    r1 <- r_inner + (.$year_id - 1) * ring_width
    r2 <- r_inner + .$year_id * ring_width
    poly <- make_tile(theta1, theta2, r1, r2)
    poly$region <- .$region
    poly$exposure <- .$exposure
    poly$node_id <- .$node_id
    poly$year <- .$year
    poly$annual_improvement_support <- .$rawp10_importance
    poly$plot_support <- pmin(.$rawp10_importance, vmax)
    poly$domain_label <- as.character(.$domain_label)
    poly$group_id <- paste(.$region, .$exposure, .$year, sep = "__")
    poly
  }) %>% ungroup()

# Ring grid and year labels.
grid_r <- r_inner + seq_along(years) * ring_width
grid_df <- do.call(rbind, lapply(seq_along(grid_r), function(i) {
  th <- seq(theta_min, theta_max, length.out = 300)
  data.frame(x = grid_r[i] * cos(th), y = grid_r[i] * sin(th), year = years[i], ring = i)
}))
year_lab <- data.frame(
  year = years,
  x = (grid_r - ring_width * 0.5) * cos(theta_min - 0.035),
  y = (grid_r - ring_width * 0.5) * sin(theta_min - 0.035),
  angle = (theta_min - 0.035) * 180 / pi - 90
)

# Domain arcs and labels.
domain_blocks <- union %>%
  group_by(domain_label) %>%
  summarise(start = min(node_id), end = max(node_id), mid = mean(c(min(node_id), max(node_id))), .groups = "drop") %>%
  mutate(
    theta1 = theta_min + (start - 1) * node_width,
    theta2 = theta_min + end * node_width,
    theta_mid = theta_min + (mid - 0.5) * node_width,
    label_x = (r_outer + 0.23) * cos(theta_mid),
    label_y = (r_outer + 0.23) * sin(theta_mid),
    label_angle = theta_mid * 180 / pi,
    label_angle = ifelse(label_angle < -90, label_angle + 180, label_angle),
    label_angle = ifelse(label_angle > 90, label_angle - 180, label_angle)
  )
arc_df <- do.call(rbind, lapply(seq_len(nrow(domain_blocks)), function(i) {
  th <- seq(domain_blocks$theta1[i], domain_blocks$theta2[i], length.out = 80)
  data.frame(
    x = (r_outer + 0.08) * cos(th), y = (r_outer + 0.08) * sin(th),
    domain_label = as.character(domain_blocks$domain_label[i])
  )
}))

# Outermost class ring for the text-free version. Source classes are contiguous
# because nodes are ordered by class and then domain.
class_blocks <- union %>%
  group_by(source_class) %>%
  summarise(start = min(node_id), end = max(node_id), .groups = "drop") %>%
  mutate(
    theta1 = theta_min + (start - 1) * node_width,
    theta2 = theta_min + end * node_width
  )
class_arc_df <- do.call(rbind, lapply(seq_len(nrow(class_blocks)), function(i) {
  th <- seq(class_blocks$theta1[i], class_blocks$theta2[i], length.out = 120)
  data.frame(
    x = (r_outer + 0.15) * cos(th), y = (r_outer + 0.15) * sin(th),
    source_class = as.character(class_blocks$source_class[i])
  )
}))

# Outer domain box glyphs. Each glyph summarizes all selected nodes x years within the domain.
dist_df <- plot_df %>%
  group_by(region, domain_label) %>%
  summarise(
    q25 = quantile(rawp10_importance, 0.25, na.rm = TRUE),
    med = median(rawp10_importance, na.rm = TRUE),
    q75 = quantile(rawp10_importance, 0.75, na.rm = TRUE),
    lo = quantile(rawp10_importance, 0.10, na.rm = TRUE),
    hi = quantile(rawp10_importance, 0.90, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(domain_blocks %>% select(domain_label, theta_mid), by = "domain_label") %>%
  mutate(
    scale = 0.28 / max(hi, na.rm = TRUE),
    r0 = r_outer + 0.16,
    r_lo = r0 + lo * scale,
    r_q25 = r0 + q25 * scale,
    r_med = r0 + med * scale,
    r_q75 = r0 + q75 * scale,
    r_hi = r0 + hi * scale,
    w = abs(node_width) * 1.8
  )

make_box <- function(theta, r1, r2, w) {
  make_tile(theta - w / 2, theta + w / 2, r1, r2, n = 2)
}
box_poly <- dist_df %>%
  rowwise() %>%
  do({
    poly <- make_box(.$theta_mid, .$r_q25, .$r_q75, .$w)
    poly$region <- .$region
    poly$domain_label <- as.character(.$domain_label)
    poly$group_id <- paste(.$region, .$domain_label, sep = "__")
    poly
  }) %>% ungroup()
whisker_df <- dist_df %>%
  transmute(
    region, domain_label,
    x = r_lo * cos(theta_mid), y = r_lo * sin(theta_mid),
    xend = r_hi * cos(theta_mid), yend = r_hi * sin(theta_mid),
    xmed1 = r_med * cos(theta_mid - w / 2), ymed1 = r_med * sin(theta_mid - w / 2),
    xmed2 = r_med * cos(theta_mid + w / 2), ymed2 = r_med * sin(theta_mid + w / 2)
  )

# Handle colour legend inside the fan.
leg_vals <- seq(0, vmax, length.out = 60)
legend_df <- data.frame(
  x = -0.028, xmax = 0.028,
  y = seq(-0.66, -0.04, length.out = 60),
  ymax = seq(-0.66, -0.04, length.out = 60) + 0.62 / 60,
  value = leg_vals
)

p <- ggplot() +
  geom_polygon(data = tile_poly, aes(x = x, y = y, group = group_id, fill = plot_support), color = "#eeeeee", linewidth = 0.10) +
  geom_path(data = grid_df, aes(x = x, y = y, group = ring), color = "#d7d7d7", linewidth = 0.20) +
  geom_path(data = arc_df, aes(x = x, y = y, group = domain_label, color = domain_label), linewidth = 1.4, lineend = "butt", inherit.aes = FALSE) +
  geom_polygon(data = box_poly, aes(x = x, y = y, group = group_id, color = domain_label), fill = "white", alpha = 0.88, linewidth = 0.35) +
  geom_segment(data = whisker_df, aes(x = x, y = y, xend = xend, yend = yend, color = domain_label), linewidth = 0.35) +
  geom_segment(data = whisker_df, aes(x = xmed1, y = ymed1, xend = xmed2, yend = ymed2, color = domain_label), linewidth = 0.65) +
  geom_rect(data = legend_df, aes(xmin = x, xmax = xmax, ymin = y, ymax = ymax, fill = value), color = NA, inherit.aes = FALSE) +
  annotate("text", x = 0, y = -0.015, label = sprintf("%.2f", vmax), size = 2.2, color = "#6d6d6d") +
  annotate("text", x = 0, y = -0.705, label = "0", size = 2.2, color = "#6d6d6d") +
  geom_text(data = year_lab, aes(x = x, y = y, label = year, angle = angle), size = 2.05, color = "#8a8a8a") +
  geom_text(data = domain_blocks, aes(x = label_x, y = label_y, label = domain_label, angle = label_angle), size = 2.35, color = "#777777") +
  facet_wrap(~ region, ncol = 2) +
  scale_fill_gradient(low = "white", high = "#D77291", limits = c(0, vmax), oob = scales::squish, name = "Permutation importance") +
  scale_color_manual(values = domain_cols, guide = "none") +
  coord_equal(clip = "off") +
  theme_void(base_family = "Helvetica") +
  theme(
    strip.text = element_text(size = 9, face = "plain", margin = margin(b = 2)),
    legend.position = "none",
    plot.margin = margin(8, 18, 8, 18)
  )

out_base <- file.path(fig5, "main_point5_exposome_milk_loss_risk_region_variable_permutation_rawp10_fan")
ggsave(paste0(out_base, ".svg"), p, width = 8.6, height = 7.2)
ggsave(paste0(out_base, ".png"), p, width = 8.6, height = 7.2, dpi = 450)

message("Wrote ", out_base, ".svg/png")


# Text-free version for manuscript assembly. The handle is retained as the colour legend;
# all other text and the outer distribution glyphs are removed.
p_wo <- ggplot() +
  geom_polygon(data = tile_poly, aes(x = x, y = y, group = group_id, fill = plot_support), color = "#eeeeee", linewidth = 0.10) +
  geom_path(data = grid_df, aes(x = x, y = y, group = ring), color = "#d7d7d7", linewidth = 0.20) +
  geom_path(data = arc_df, aes(x = x, y = y, group = domain_label, color = domain_label), linewidth = 2.4, lineend = "butt", inherit.aes = FALSE) +
  geom_path(data = class_arc_df, aes(x = x, y = y, group = source_class, color = source_class), linewidth = 2.4, lineend = "butt", inherit.aes = FALSE) +
  geom_rect(data = legend_df, aes(xmin = x, xmax = xmax, ymin = y, ymax = ymax, fill = value), color = NA, inherit.aes = FALSE) +
  facet_wrap(~ region, ncol = 2) +
  scale_fill_gradient(low = "white", high = "#D77291", limits = c(0, vmax), oob = scales::squish, name = "Permutation importance") +
  scale_color_manual(values = c(domain_cols, class_cols), guide = "none") +
  coord_equal(clip = "off") +
  theme_void(base_family = "Helvetica") +
  theme(
    strip.text = element_blank(),
    legend.position = "none",
    plot.margin = margin(8, 18, 8, 18)
  )

out_wo <- file.path(fig5, "main_point5_exposome_milk_loss_risk_region_variable_permutation_rawp10_fan_wo_legend")
ggsave(paste0(out_wo, ".svg"), p_wo, width = 8.6, height = 7.2)
ggsave(paste0(out_wo, ".png"), p_wo, width = 8.6, height = 7.2, dpi = 450)

message("Wrote ", out_wo, ".svg/png")
