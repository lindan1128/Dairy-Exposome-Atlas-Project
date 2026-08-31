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
out_svg <- file.path(fig, "supp_point2_2015_2024_heat_dry_vs_humid_exposure_intensity_vs_abs_beta_std_change_zoom_wo_legend.svg")

d <- fread(in_table)
d <- d[
  is.finite(intensity_slope_z_per_year_2015_2024) &
    is.finite(abs_beta_std_slope_per_year_2015_2024)
]

# Match the main zoom plotting window, then retain only strictly defined dry-heat and humid/wet-heat metrics.
zoom_pool <- d[
  intensity_slope_z_per_year_2015_2024 >= -0.05 &
    intensity_slope_z_per_year_2015_2024 <= 0.07
]
y_bound <- quantile(abs(zoom_pool$abs_beta_std_slope_per_year_2015_2024), 0.985, na.rm = TRUE)
zoom_pool <- zoom_pool[abs(abs_beta_std_slope_per_year_2015_2024) <= y_bound * 1.15]
pdat <- zoom_pool[class_label == "Heat"]

pdat[, heat_type := fifelse(
  grepl("dry|vpd", exposure, ignore.case = TRUE),
  "Dry heat",
  fifelse(
    grepl("humid|wetbulb", exposure, ignore.case = TRUE),
    "Humid heat",
    NA_character_
  )
)]
pdat <- pdat[!is.na(heat_type)]
pdat[, y_plot := abs_beta_std_slope_per_year_2015_2024]
pdat[, heat_type := factor(heat_type, levels = c("Dry heat", "Humid heat"))]

label_map <- data.table(
  exposure = c(
    "daymet_dairy_weighted_dry_heat_days_t32_rh40",
    "daymet_dairy_weighted_vpd_days_ge_3kpa",
    "daymet_dairy_weighted_humid_hot_days_t72wb24",
    "daymet_dairy_weighted_wetbulb_days_ge_26c"
  ),
  label = c(
    "Dry heat days\n(Tmax≥32°C, RH≤40%)",
    "VPD days\n(≥3 kPa)",
    "Humid hot days\n(THI≥72, wet-bulb≥24°C)",
    "Severe wet-bulb heat\n(Wet-bulb≥26°C)"
  )
)
pdat[, label := ""]
pdat[label_map, label := i.label, on = "exposure"]

x_lim <- c(-0.006, 0.027)
y_pad <- max(abs(pdat$y_plot), na.rm = TRUE) * 0.22
y_lim <- range(pdat$y_plot, na.rm = TRUE) + c(-y_pad, y_pad)
quad_bg <- data.table(
  xmin = c(0, 0, x_lim[1], x_lim[1]),
  xmax = c(x_lim[2], x_lim[2], 0, 0),
  ymin = c(0, y_lim[1], 0, y_lim[1]),
  ymax = c(y_lim[2], 0, y_lim[2], 0),
  fill = c("#d4a373", "#faedcd", "#e9edc9", "#ccd5ae")
)

heat_color <- "#32a4b4"

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
    aes(shape = heat_type),
    size = 3.1,
    stroke = 0.45,
    color = "#222222",
    fill = heat_color,
    alpha = 0.95
  ) +
  geom_text_repel(
    data = pdat[label != ""],
    aes(label = label),
    family = "Arial",
    size = 8.7 / ggplot2::.pt,
    color = "#222222",
    alpha = 0.92,
    seed = 43,
    max.overlaps = Inf,
    box.padding = 0.85,
    point.padding = 0.52,
    min.segment.length = 0.02,
    force = 70,
    force_pull = 0.035,
    max.time = 12,
    max.iter = 30000,
    segment.color = "#000000",
    segment.size = 0.34,
    segment.alpha = 0.95,
    arrow = grid::arrow(length = grid::unit(0.055, "inches"), angle = 20, type = "closed")
  ) +
  scale_shape_manual(values = c("Dry heat" = 21, "Humid heat" = 24), name = "Heat type") +
  guides(shape = guide_legend(override.aes = list(fill = heat_color, color = "#222222", size = 3.1))) +
  coord_cartesian(xlim = x_lim, ylim = y_lim, clip = "off") +
  labs(
    x = "2015-2024 trend in exposure intensity (SD/year)",
    y = "2015-2024 trend in exposure-milk association strength (standardized β/year)"
  ) +
  theme_classic(base_family = "Arial", base_size = 9) +
  theme(
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.position = "bottom",
    axis.text = element_text(size = 9),
    axis.title = element_text(size = 9),
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 9),
    legend.key = element_rect(fill = "transparent", color = NA),
    plot.margin = margin(5.5, 5.5, 5.5, 5.5)
  )

svglite(out_svg, width = 8.36666, height = 4.61333, bg = "transparent", system_fonts = list(Arial = "Arial"))
print(p)
dev.off()
cat("Wrote", out_svg, "\n")
