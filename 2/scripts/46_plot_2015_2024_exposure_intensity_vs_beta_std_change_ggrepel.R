#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggrepel)
  library(svglite)
})

cmd <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", cmd[grep("--file=", cmd)][1])
point <- normalizePath(file.path(dirname(script_path), ".."))
tab <- file.path(point, "tables")
fig <- file.path(point, "figures")

in_table <- file.path(tab, "point2_2015_2024_exposure_intensity_vs_abs_beta_std_change.csv")
out_stem <- "main_point2_2015_2024_exposure_intensity_vs_abs_beta_std_change_scatter"

domain_order <- c("Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand")
colors <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage" = "#1E7A8D",
  "Feed market" = "#fbc4ab",
  "Dairy market" = "#E47666",
  "Market demand" = "#f09d51"
)

select_mechanism_labels <- function(pdat) {
  keep <- data.table(
    exposure = c(
      "daymet_dairy_weighted_dry_heat_days_t32_rh40",
      "daymet_dairy_weighted_wetbulb_days_ge_26c",
      "daymet_dairy_weighted_consec_ice_days_maxrun",
      "daymet_dairy_weighted_wet_cold_load_lt45",
      "nass_pastureland_poor_or_very_poor_pct",
      "nass_forage_poor_or_very_poor_pct"
    ),
    repel_label = c(
      "Dry heat days\n(Tmax≥32°C, RH≤40%)",
      "Severe wet-bulb heat\n(Wet-bulb≥26°C)",
      "Consecutive ice days",
      "Wet-cold load\n(THI<45)",
      "Poor pasture condition",
      "Poor forage condition"
    )
  )
  pdat[keep, on = "exposure", nomatch = 0]
}

plot_one <- function(d, zoom = FALSE, wo_legend = FALSE) {
  pdat <- copy(d)
  pdat <- pdat[
    is.finite(intensity_slope_z_per_year_2015_2024) &
      is.finite(abs_beta_std_slope_per_year_2015_2024)
  ]
  if (zoom) {
    pdat <- pdat[
      intensity_slope_z_per_year_2015_2024 >= -0.05 &
        intensity_slope_z_per_year_2015_2024 <= 0.07
    ]
    y_bound <- quantile(abs(pdat$abs_beta_std_slope_per_year_2015_2024), 0.985, na.rm = TRUE)
    pdat <- pdat[abs(abs_beta_std_slope_per_year_2015_2024) <= y_bound * 1.15]
  }
  if (zoom && wo_legend) {
    left_upper_feed <- c(
      "feed_hay_to_corn_price_ratio_state_month_anomaly",
      "feed_hay_to_corn_price_ratio",
      "feed_hay_to_corn_price_ratio_state_month_robust_z",
      "feed_alfalfa_hay_price_ratio_state_month_anomaly"
    )
    upper_cold <- c(
      "daymet_dairy_weighted_dry_cold_thi_mean",
      "daymet_dairy_weighted_dry_cold_thi_min"
    )
    pdat <- pdat[!(exposure %in% left_upper_feed)]
    pdat <- pdat[!(exposure %in% upper_cold)]
  }
  pdat[, y_plot := abs_beta_std_slope_per_year_2015_2024]
  pdat[, class_label := factor(class_label, levels = domain_order)]
  if (!("mean_incremental_r2_2015_2024" %in% names(pdat))) {
    pdat[, mean_incremental_r2_2015_2024 := 0]
  }
  pdat[!is.finite(mean_incremental_r2_2015_2024) | mean_incremental_r2_2015_2024 < 0, mean_incremental_r2_2015_2024 := 0]

  labels <- select_mechanism_labels(pdat)
  pdat[, repel_label := ""]
  pdat[labels, repel_label := i.repel_label, on = "exposure"]
  manual_label_exposures <- c("daymet_dairy_weighted_wet_cold_load_lt45")
  manual_labels <- copy(pdat[exposure %in% manual_label_exposures])
  if (nrow(manual_labels)) {
    manual_labels[, `:=`(
      label_x = -0.030,
      label_y = -0.0042,
      segment_x = -0.0242,
      segment_y = -0.0031,
      point_edge_x = intensity_slope_z_per_year_2015_2024 - 0.0011,
      point_edge_y = y_plot + 0.00025,
      repel_label = "Wet-cold load\n(THI<45)"
    )]
  }
  repel_dat <- pdat[repel_label != "" & !(exposure %in% manual_label_exposures)]

  x_abs <- max(abs(pdat$intensity_slope_z_per_year_2015_2024), na.rm = TRUE)
  y_abs <- max(abs(pdat$y_plot), na.rm = TRUE)
  x_pad <- max(x_abs * 0.20, 0.012)
  y_pad <- max(y_abs * 0.18, 0.004)
  x_lim <- range(pdat$intensity_slope_z_per_year_2015_2024, na.rm = TRUE) + c(-x_pad, x_pad)
  y_lim <- range(pdat$y_plot, na.rm = TRUE) + c(-y_pad, y_pad)
  if (zoom && wo_legend) {
    y_lim <- c(-0.02, 0.02)
  }
  quad_bg <- data.table(
    xmin = c(0, 0, x_lim[1], x_lim[1]),
    xmax = c(x_lim[2], x_lim[2], 0, 0),
    ymin = c(0, y_lim[1], 0, y_lim[1]),
    ymax = c(y_lim[2], 0, y_lim[2], 0),
    fill = c("#d4a373", "#faedcd", "#e9edc9", "#ccd5ae")
  )

  p <- ggplot(pdat, aes(x = intensity_slope_z_per_year_2015_2024, y = y_plot)) +
    geom_rect(
      data = quad_bg,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      fill = quad_bg$fill,
      alpha = 0.34,
      color = NA
    ) +
    geom_hline(yintercept = 0, color = "#777777", linewidth = 0.35, linetype = "22") +
    geom_vline(xintercept = 0, color = "#777777", linewidth = 0.35, linetype = "22") +
    geom_point(
      aes(fill = class_label, size = mean_incremental_r2_2015_2024),
      shape = 21,
      stroke = 0.35,
      color = "#222222",
      alpha = 0.92
    ) +
    geom_text_repel(
      data = repel_dat,
      aes(label = repel_label),
      family = "Arial",
      size = 9 / ggplot2::.pt,
      color = "#222222",
      alpha = 0.92,
      seed = 42,
      max.overlaps = Inf,
      box.padding = 0.95,
      point.padding = 0.58,
      min.segment.length = 0.02,
      force = 80,
      force_pull = 0.035,
      max.time = 12,
      max.iter = 30000,
      segment.color = "#000000",
      segment.size = 0.34,
      segment.alpha = 0.95,
      arrow = grid::arrow(length = grid::unit(0.055, "inches"), angle = 20, type = "closed")
    ) +
    geom_segment(
      data = manual_labels,
      aes(x = segment_x, y = segment_y, xend = point_edge_x, yend = point_edge_y),
      inherit.aes = FALSE,
      color = "#000000",
      linewidth = 0.34,
      alpha = 0.95,
      arrow = grid::arrow(length = grid::unit(0.055, "inches"), angle = 20, type = "closed")
    ) +
    geom_text(
      data = manual_labels,
      aes(x = label_x, y = label_y, label = repel_label),
      inherit.aes = FALSE,
      family = "Arial",
      size = 9 / ggplot2::.pt,
      color = "#222222",
      alpha = 0.92,
      hjust = 0.5,
      vjust = 0.5
    ) +
    scale_fill_manual(values = colors, breaks = domain_order, drop = FALSE) +
    scale_size_continuous(
      range = c(2.0, 6.0),
      breaks = scales::breaks_pretty(n = 3),
      labels = function(x) sprintf("%.2f%%", 100 * x),
      name = "Mean incremental R²"
    ) +
    coord_cartesian(xlim = x_lim, ylim = y_lim, clip = "off") +
    labs(
      x = "2015-2024 trend in exposure intensity (SD/year)",
      y = "2015-2024 trend in exposure-milk association strength (standardized β/year)",
      fill = NULL
    ) +
    theme_classic(base_family = "Arial", base_size = 9) +
    theme(
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.text = element_text(size = 9),
      axis.text = element_text(size = 9),
      axis.title = element_text(size = 9),
      plot.margin = margin(5.5, 5.5, 5.5, 5.5)
    ) +
    guides(
      fill = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(size = 3)),
      size = guide_legend(order = 2)
    )

  if (wo_legend) {
    p <- p + theme(legend.position = "none")
  }

  suffix <- if (zoom) "_zoom" else ""
  suffix <- paste0(suffix, if (wo_legend) "_wo_legend" else "")
  out <- file.path(fig, paste0(out_stem, suffix, ".svg"))
  out_height <- if (zoom && wo_legend) 4.61333 else 5.61333
  svglite(out, width = 8.36666, height = out_height, bg = "transparent", system_fonts = list(Arial = "Arial"))
  print(p)
  dev.off()
  cat("Wrote", out, "\n")
}

d <- fread(in_table)
d[, exposure_label := gsub(">=", "≥", exposure_label, fixed = TRUE)]
d[, exposure_label := gsub("<=", "≤", exposure_label, fixed = TRUE)]
plot_one(d, zoom = FALSE)
plot_one(d, zoom = TRUE)
plot_one(d, zoom = TRUE, wo_legend = TRUE)
