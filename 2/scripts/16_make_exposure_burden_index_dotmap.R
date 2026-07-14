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

domain_levels <- c(
  "Heat",
  "Cold",
  "Severe weather",
  "Forage condition",
  "Agricultural pesticides",
  "Feed market",
  "Milk price / dairy market",
  "Market demand"
)
domain_labels <- c(
  "Heat" = "Heat",
  "Cold" = "Cold",
  "Severe weather" = "Severe weather",
  "Forage condition" = "Forage",
  "Agricultural pesticides" = "Pesticides",
  "Feed market" = "Feed market",
  "Milk price / dairy market" = "Dairy market",
  "Market demand" = "Market demand",
  "Herd structure / scale" = "Dairy scale",
  "Dairy scale" = "Dairy scale"
)

idx <- read_csv(
  file.path(tab, "point2_exposure_signal_vs_all_clean_translation_index.csv"),
  show_col_types = FALSE
) %>%
  filter(domain %in% domain_levels) %>%
  mutate(
    domain = factor(domain, levels = rev(domain_levels)),
    domain_label = factor(domain_labels[as.character(domain)], levels = domain_labels[rev(domain_levels)]),
    year_f = factor(year, levels = 2000:2025),
    direction = case_when(
      year == 2000 ~ "2000 baseline",
      exposure_burden_index_2000 > 1 ~ "> 2000 level",
      exposure_burden_index_2000 < 1 ~ "< 2000 level",
      TRUE ~ "2000 baseline"
    ),
    direction = factor(direction, levels = c("> 2000 level", "< 2000 level", "2000 baseline")),
    index_deviation = abs(exposure_burden_index_2000 - 1)
  ) %>%
  group_by(exposure_pool, domain) %>%
  mutate(
    index_deviation_scaled = {
      rng <- range(index_deviation, na.rm = TRUE)
      if (is.finite(rng[1]) && is.finite(rng[2]) && rng[2] > rng[1]) {
        rescale(index_deviation, to = c(0.02, 1.00))
      } else {
        rep(0.50, dplyr::n())
      }
    },
    index_deviation_alpha = rescale(index_deviation_scaled, to = c(0.60, 1.00))
  ) %>%
  ungroup()

trend <- idx %>%
  arrange(exposure_pool, domain, year) %>%
  group_by(exposure_pool, domain, domain_label) %>%
  summarize(
    slope_per_year = coef(lm(exposure_burden_index_2000 ~ year))[["year"]],
    year_start = min(year[is.finite(exposure_burden_index_2000)]),
    year_end = max(year[is.finite(exposure_burden_index_2000)]),
    index_start = exposure_burden_index_2000[year == year_start][1],
    index_end = exposure_burden_index_2000[year == year_end][1],
    pct_change_observed_window = 100 * (index_end / index_start - 1),
    .groups = "drop"
  )
write_csv(trend, file.path(tab, "point2_exposure_burden_index_dotmap_trends.csv"))

dir_cols <- c(
  "> 2000 level" = "#B1C99C",
  "< 2000 level" = "#FFCC8D",
  "2000 baseline" = "#BDBDBD"
)

base_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(size = 9, face = "plain", color = "#222222"),
    axis.title = element_blank(),
    axis.text.x = element_text(size = 9, hjust = 0.5, vjust = 0.62, color = "#222222"),
    axis.text.y = element_text(size = 9, color = "#222222"),
    axis.ticks = element_line(linewidth = 0.22, color = "#111111"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.28),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "#FFFFFF", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    legend.title = element_text(size = 9, face = "plain"),
    legend.text = element_text(size = 9, face = "plain"),
    legend.position = "right",
    legend.box = "vertical",
    legend.spacing.y = unit(2, "pt"),
    plot.margin = margin(t = 2, r = 5, b = 2, l = 2)
  )

make_plot <- function(pool_label, stem) {
  d <- idx %>% filter(exposure_pool == pool_label)
  highlight_rows <- data.frame(
    xmin = which(levels(d$year_f) == "2001") - 0.5,
    xmax = length(levels(d$year_f)) + 0.5,
    ymin = match(c("Heat", "Severe weather"), levels(d$domain_label)) - 0.5,
    ymax = match(c("Heat", "Severe weather"), levels(d$domain_label)) + 0.5
  )
  p <- ggplot(d, aes(year_f, domain_label)) +
    annotate(
      "rect",
      xmin = seq(1.5, length(unique(d$year_f)) - 0.5, by = 1),
      xmax = seq(2.5, length(unique(d$year_f)) + 0.5, by = 1),
      ymin = -Inf,
      ymax = Inf,
      fill = alpha("#F1F1F2", 0.25),
      color = NA
    ) +
    geom_rect(
      data = highlight_rows,
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      fill = "#d8a48f",
      alpha = 0.15,
      color = NA
    ) +
    geom_vline(
      xintercept = seq(1.5, length(unique(d$year_f)) - 0.5, by = 1),
      color = "#111111",
      linewidth = 0.28,
      linetype = "dotted"
    ) +
    geom_point(
      aes(size = index_deviation_scaled, fill = direction, alpha = index_deviation_alpha),
      shape = 21,
      color = "#111111",
      stroke = 0.15
    ) +
    scale_fill_manual(values = dir_cols, name = "Exposure level") +
    scale_alpha_identity() +
    scale_size_area(
      max_size = 5.8,
      limits = c(0, 1),
      breaks = c(0.25, 0.50, 0.75),
      labels = c("25", "50", "75"),
      name = "Deviation from 2000"
    ) +
    scale_x_discrete(position = "top", drop = FALSE, guide = guide_axis(angle = 90)) +
    scale_y_discrete(drop = FALSE) +
    guides(
      fill = guide_legend(
        override.aes = list(size = 3.4, alpha = 1),
        order = 1
      ),
      size = guide_legend(
        override.aes = list(fill = "#111111", color = "#111111", alpha = 1),
        order = 2
      )
    ) +
    base_theme

  out <- file.path(fig, paste0(stem, ".svg"))
  ggsave(out, p, width = 8.5, height = 2.0, bg = "transparent")
  clean_svg(out)

  out2 <- file.path(fig, paste0(stem, "_wo_legend.svg"))
  ggsave(out2, p + theme(legend.position = "none"), width = 8.5, height = 2.0, bg = "transparent")
  clean_svg(out2)
}

make_plot(
  "Chord-signal variables",
  "main_point2_exposure_burden_index_dotmap_chord_signal"
)
make_plot(
  "All clean native variables",
  "main_point2_exposure_burden_index_dotmap_all_clean"
)
