#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grid)
  library(sf)
})
sf::sf_use_s2(FALSE)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else normalizePath(".")
root <- normalizePath(file.path(dirname(script_path), "..", "..", "..", "..", ".."), mustWork = TRUE)
tab <- file.path(root, "analysis/statistics/6/tables")
fig <- file.path(root, "analysis/statistics/6/figures")
dir.create(fig, recursive = TRUE, showWarnings = FALSE)

out_base <- file.path(fig, "point6_point5consistent_state_raw_feature_domain_importance_3d_map_rawp30_imputed26")

domain_order <- c(
  "Heat", "Cold", "Severe weather",
  "Forage", "Feed market", "Market demand", "Dairy market"
)
domain_cols <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage" = "#1E7A8D",
  "Feed market" = "#fbc4ab",
  "Market demand" = "#f09d51",
  "Dairy market" = "#E47666"
)
region_cols <- c(
  "map_South" = grDevices::adjustcolor("#D49BAC", alpha.f = 0.6),
  "map_West" = grDevices::adjustcolor("#B8A9C4", alpha.f = 0.6),
  "map_Midwest" = grDevices::adjustcolor("#EBC28A", alpha.f = 0.6),
  "map_Northeast" = grDevices::adjustcolor("#E79B79", alpha.f = 0.6),
  "map_other" = "#eeeeee"
)

poly <- read_csv(file.path(tab, "point6_us_state_map_polygons.csv"), show_col_types = FALSE)
imp_raw <- read_csv(
  file.path(tab, "point6_point5consistent_state_raw_feature_permutation_rawp30_domain_summary.csv"),
  show_col_types = FALSE
)
analysis_states <- read_csv(
  file.path(root, "analysis/statistics/0/tables/point0_26_state_percow_state_list.csv"),
  show_col_types = FALSE
) %>%
  pull(state_alpha) %>%
  sort()
state_region <- read_csv(
  file.path(root, "analysis/statistics/5/tables/point5_forecast_state_month_feature_panel.csv"),
  show_col_types = FALSE
) %>%
  distinct(state_alpha, region)

make_state_anchor <- function(df) {
  polys <- df %>%
    group_by(state_alpha, segment_id) %>%
    arrange(point_id, .by_group = TRUE) %>%
    summarise(coords = list(as.matrix(data.frame(x = x, y = y))), .groups = "drop") %>%
    mutate(coords = lapply(coords, function(m) {
      if (!all(m[1, ] == m[nrow(m), ])) {
        m <- rbind(m, m[1, ])
      }
      m
    }))

  seg_geom <- st_as_sf(
    polys %>% select(state_alpha, segment_id),
    geometry = st_make_valid(st_sfc(lapply(polys$coords, function(m) st_polygon(list(m))), crs = 4326))
  )
  state_geom <- seg_geom %>%
    group_by(state_alpha) %>%
    summarise(geometry = st_union(geometry), .groups = "drop") %>%
    st_make_valid()

  pts <- suppressWarnings(st_point_on_surface(state_geom))
  xy <- st_coordinates(pts)
  tibble(state_alpha = state_geom$state_alpha, x = xy[, 1], y = xy[, 2])
}

cent <- make_state_anchor(poly) %>%
  filter(state_alpha %in% analysis_states) %>%
  left_join(state_region, by = "state_alpha")

imp_all <- cent %>%
  select(state_alpha, region, x, y) %>%
  tidyr::crossing(domain = factor(domain_order, levels = domain_order)) %>%
  left_join(
    imp_raw %>%
      filter(domain_label %in% domain_order) %>%
      transmute(
        state_alpha,
        domain = domain_label,
        mean_importance_rmse,
        min_p_value,
        n_rawp30_exposures
      ),
    by = c("state_alpha", "domain")
  ) %>%
  mutate(
    mean_importance_rmse = coalesce(mean_importance_rmse, 0),
    min_p_value = coalesce(min_p_value, 1),
    n_rawp30_exposures = coalesce(n_rawp30_exposures, 0L)
  )

missing_states <- imp_all %>%
  group_by(state_alpha, region) %>%
  summarise(total_importance = sum(mean_importance_rmse, na.rm = TRUE), .groups = "drop") %>%
  filter(total_importance <= 0)

region_domain_mean <- imp_all %>%
  anti_join(missing_states %>% select(state_alpha), by = "state_alpha") %>%
  group_by(region, domain) %>%
  summarise(
    mean_importance_rmse_imp = mean(mean_importance_rmse, na.rm = TRUE),
    min_p_value_imp = min(min_p_value, na.rm = TRUE),
    n_rawp30_exposures_imp = round(mean(n_rawp30_exposures, na.rm = TRUE)),
    .groups = "drop"
  )

imp_all <- imp_all %>%
  left_join(region_domain_mean, by = c("region", "domain")) %>%
  mutate(
    imputed = state_alpha %in% missing_states$state_alpha,
    mean_importance_rmse = if_else(imputed, mean_importance_rmse_imp, mean_importance_rmse),
    min_p_value = if_else(imputed, min_p_value_imp, min_p_value),
    n_rawp30_exposures = if_else(imputed, as.integer(n_rawp30_exposures_imp), n_rawp30_exposures)
  ) %>%
  select(state_alpha, region, x, y, domain, mean_importance_rmse, min_p_value, n_rawp30_exposures, imputed)

write_csv(imp_all, file.path(tab, "point6_point5consistent_state_raw_feature_permutation_rawp30_domain_summary_imputed26_for_map.csv"))

imp <- imp_all %>%
  filter(min_p_value < 0.30, mean_importance_rmse > 0)

zmax <- max(imp$mean_importance_rmse, na.rm = TRUE)
if (!is.finite(zmax) || zmax <= 0) zmax <- 1

project3d <- function(x, y, z) {
  x0 <- x + 96
  y0 <- y - 37.5
  tibble(
    X = x0 + 0.19 * y0,
    Y = 0.64 * y0 + 0.86 * z,
    depth = 0.38 * y0 - 0.18 * z
  )
}

top_poly <- poly %>%
  bind_cols(project3d(poly$x, poly$y, 0)) %>%
  left_join(state_region, by = "state_alpha") %>%
  mutate(
    group_id = interaction(state_alpha, segment_id, drop = TRUE),
    map_fill = if_else(state_alpha %in% analysis_states, paste0("map_", region), "map_other"),
    map_fill = coalesce(map_fill, "map_other")
  )

bottom_poly <- poly %>%
  bind_cols(project3d(poly$x, poly$y, -1.65)) %>%
  mutate(group_id = interaction(state_alpha, segment_id, drop = TRUE))

side_poly <- poly %>%
  group_by(state_alpha, segment_id) %>%
  arrange(point_id, .by_group = TRUE) %>%
  mutate(
    x_next = lead(x, default = first(x)),
    y_next = lead(y, default = first(y)),
    edge_id = row_number()
  ) %>%
  ungroup() %>%
  rowwise() %>%
  do({
    r <- .
    p <- project3d(
      c(r$x, r$x_next, r$x_next, r$x),
      c(r$y, r$y_next, r$y_next, r$y),
      c(0, 0, -1.65, -1.65)
    )
    tibble(
      state_alpha = r$state_alpha,
      segment_id = r$segment_id,
      group_id = paste(r$state_alpha, r$segment_id, r$edge_id, sep = "__"),
      X = p$X,
      Y = p$Y,
      depth = mean(p$depth)
    )
  }) %>%
  ungroup()

domain_offsets <- tibble(
  domain = factor(domain_order, levels = domain_order),
  dx = c(-0.78, -0.26, 0.26, 0.78, -0.52, 0.00, 0.52),
  dy = c( 0.32,  0.32, 0.32, 0.32, -0.32, -0.32, -0.32)
)

cone_base <- imp %>%
  left_join(domain_offsets, by = "domain") %>%
  mutate(
    layout_scale = if_else(state_alpha %in% c("VT"), 0.46, 1.00),
    radius_scale = if_else(state_alpha %in% c("VT"), 0.62, 1.00),
    cx = x + dx * layout_scale,
    cy = y + dy * layout_scale,
    radius = 0.228 * radius_scale,
    height = 0.12 + 10.6 * sqrt(pmax(mean_importance_rmse, 0) / zmax)
  )

make_cone_triangles <- function(row, n = 18) {
  theta <- seq(0, 2 * pi, length.out = n + 1)
  cone_depth <- project3d(row$cx, row$cy, 0)$depth
  out <- lapply(seq_len(n), function(i) {
    base <- project3d(
      c(
        row$cx + row$radius * cos(theta[i]),
        row$cx + row$radius * cos(theta[i + 1]),
        row$cx
      ),
      c(
        row$cy + row$radius * sin(theta[i]),
        row$cy + row$radius * sin(theta[i + 1]),
        row$cy
      ),
      c(0.04, 0.04, row$height)
    )
    tibble(
      state_alpha = row$state_alpha,
      domain = as.character(row$domain),
      group_id = paste(row$state_alpha, row$domain, i, sep = "__"),
      state_y = row$y,
      cone_depth = cone_depth,
      cone_y = row$cy,
      X = base$X,
      Y = base$Y,
      depth = mean(base$depth)
    )
  })
  bind_rows(out)
}

cone_tri <- cone_base %>%
  split(seq_len(nrow(.))) %>%
  lapply(make_cone_triangles) %>%
  bind_rows() %>%
  mutate(domain = factor(domain, levels = domain_order)) %>%
  arrange(desc(state_y), desc(cone_y), desc(cone_depth), group_id) %>%
  mutate(group_id = factor(group_id, levels = unique(group_id)))

scale_df <- tibble(
  z = c(0, zmax / 2, zmax),
  label = sprintf("%.2f", c(0, zmax / 2, zmax)),
  h = 10.6 * sqrt(z / zmax)
) %>%
  bind_cols(project3d(-124.5, 49.0, .$h))

map_xlim <- range(top_poly$X, cone_tri$X, na.rm = TRUE)
map_ylim <- range(bottom_poly$Y, top_poly$Y, cone_tri$Y, na.rm = TRUE)
xpad <- diff(map_xlim) * 0.025
ypad <- diff(map_ylim) * 0.06

make_plot <- function(show_legend = TRUE) {
  p <- ggplot() +
    geom_polygon(
      data = side_poly,
      aes(x = X, y = Y, group = group_id),
      fill = "#d7d7d7",
      color = NA,
      alpha = 1
    ) +
    geom_polygon(
      data = top_poly,
      aes(x = X, y = Y, group = group_id, fill = map_fill),
      color = "#c7c7c7",
      linewidth = 0.12,
      alpha = 1
    ) +
    geom_polygon(
      data = cone_tri,
      aes(x = X, y = Y, group = group_id, fill = domain),
      color = NA,
      alpha = 1
    ) +
    scale_fill_manual(values = c(domain_cols, region_cols), breaks = domain_order, name = NULL) +
    coord_equal(
      xlim = c(map_xlim[1] - xpad, map_xlim[2] + xpad),
      ylim = c(map_ylim[1] - ypad, map_ylim[2] + ypad),
      expand = FALSE,
      clip = "off"
    ) +
    theme_void(base_size = 9, base_family = "Helvetica") +
    theme(
      text = element_text(size = 9, color = "black"),
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.position = if (show_legend) "bottom" else "none",
      legend.text = element_text(size = 9, color = "black"),
      legend.key.width = unit(14, "pt"),
      legend.key.height = unit(7, "pt"),
      plot.margin = margin(0, 0, 0, 0)
    ) +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE))

  if (show_legend) {
    p <- p +
      annotate(
        "segment",
        x = scale_df$X[1],
        xend = scale_df$X[3],
        y = scale_df$Y[1],
        yend = scale_df$Y[3],
        linewidth = 0.35,
        color = "#666666"
      ) +
      geom_text(
        data = scale_df,
        aes(x = X - 0.35, y = Y, label = label),
        inherit.aes = FALSE,
        hjust = 1,
        size = 7 / .pt,
        color = "#444444"
      ) +
      labs(title = "Point 5-consistent state-level forecast domain importance") +
      theme(
        plot.title = element_text(size = 11, face = "bold", hjust = 0.02, margin = margin(b = 2)),
        plot.margin = margin(3, 0, 0, 0)
      )
  }
  p
}

p <- make_plot(TRUE)
p_wo <- make_plot(FALSE)

ggsave(paste0(out_base, ".svg"), p, width = 7.9, height = 4.8, bg = "transparent")
ggsave(paste0(out_base, ".png"), p, width = 7.9, height = 4.8, dpi = 600, bg = "transparent")
ggsave(paste0(out_base, "_wo_legend.svg"), p_wo, width = 7.9, height = 4.0, bg = "transparent")
ggsave(paste0(out_base, "_wo_legend.png"), p_wo, width = 7.9, height = 4.0, dpi = 600, bg = "transparent")

cat("wrote point6_point5consistent_state_raw_feature_domain_importance_3d_map_rawp30\n")
