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

era_levels <- c("2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2025")

d <- read_csv(file.path(tab, "point2_humid_dry_era_lag_profile_5era.csv"), show_col_types = FALSE) %>%
  mutate(
    pool = factor(pool, levels = c("chord-signal heat", "clean-full heat", "strict paired heat")),
    pool_label = recode(
      as.character(pool),
      "clean-full heat" = "Full exposure set",
      "strict paired heat" = "Strict paired heat",
      "chord-signal heat" = "ExWAS-significant set"
    ),
    pool_label = factor(pool_label, levels = c("ExWAS-significant set", "Full exposure set", "Strict paired heat")),
    heat_form = factor(heat_form, levels = c("Humid heat", "Dry heat")),
    heat_label = recode(as.character(heat_form), "Humid heat" = "Humid heat", "Dry heat" = "Dry heat"),
    heat_label = factor(heat_label, levels = c("Humid heat", "Dry heat")),
    era = factor(era, levels = era_levels)
  )

summary <- d %>%
  group_by(pool, pool_label, heat_form, heat_label, era) %>%
  summarize(
    lag1 = loss[lag == 1][1],
    lag3 = loss[lag == 3][1],
    lag5 = loss[lag == 5][1],
    acute_residual = mean(loss[lag %in% 1:3], na.rm = TRUE),
    recovery_lag5 = lag5,
    .groups = "drop"
  )
write_csv(summary, file.path(tab, "point2_humid_dry_era_lag_profile_5era_summary.csv"))

pal <- c(
  "2000-2004" = "#D3E3E6",
  "2005-2009" = "#B3D7DD",
  "2010-2014" = "#A4C1C5",
  "2015-2019" = "#8DBEC6",
  "2020-2025" = "#72BAC6"
)

base_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(size = 9, face = "plain", color = "#222222"),
    axis.title = element_text(size = 9, face = "plain"),
    axis.text = element_text(size = 9, color = "#222222"),
    strip.text = element_text(size = 9, face = "plain"),
    strip.background = element_rect(fill = "white", color = "#111111", linewidth = 0.28),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(color = "#eeeeee", linewidth = 0.25),
    panel.spacing.x = unit(0.7, "lines"),
    panel.spacing.y = unit(0.8, "lines"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    legend.text = element_text(size = 9, face = "plain", color = "#222222"),
    legend.key.width = unit(13, "pt"),
    legend.key.height = unit(9, "pt"),
    legend.position = "bottom",
    legend.title = element_blank(),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.28)
  )

p <- ggplot(d %>% filter(pool %in% c("chord-signal heat", "clean-full heat")),
            aes(lag, loss, color = era, fill = era)) +
  geom_hline(yintercept = 0, linewidth = 0.35, color = "#777777") +
  geom_line(linewidth = 0.95, na.rm = TRUE) +
  geom_point(shape = 21, color = "#111111", stroke = 0.22, size = 2.0, na.rm = TRUE) +
  facet_grid(pool_label ~ heat_label, scales = "free_y") +
  scale_color_manual(values = pal) +
  scale_fill_manual(values = pal) +
  scale_x_continuous(breaks = 1:5) +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +
  labs(
    x = "Lag after exposure (months)",
    y = "Per-cow milk loss (% per SD exposure; positive = loss)"
  ) +
  base_theme

out <- file.path(fig, "main_point2_humid_dry_era_lag_profile_5era.svg")
ggsave(out, p, width = 7.2, height = 4.6, bg = "transparent")
clean_svg(out)

p2 <- p +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "none")
out2 <- file.path(fig, "main_point2_humid_dry_era_lag_profile_5era_wo_legend.svg")
ggsave(out2, p2, width = 7.7, height = 4.3, bg = "transparent")
clean_svg(out2)

p_strict <- ggplot(d %>% filter(pool == "strict paired heat"),
                   aes(lag, loss, color = era, fill = era)) +
  geom_hline(yintercept = 0, linewidth = 0.35, color = "#777777") +
  geom_line(linewidth = 1.95, na.rm = TRUE) +
  geom_point(shape = 21, color = "#111111", stroke = 0.22, size = 3.0, na.rm = TRUE) +
  facet_wrap(~ heat_label, scales = "fixed", nrow = 1, axes = "all_y", axis.labels = "margins") +
  scale_color_manual(values = pal) +
  scale_fill_manual(values = pal) +
  scale_x_continuous(breaks = 1:5) +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +
  labs(x = NULL, y = NULL) +
  base_theme +
  theme(
    legend.position = "none",
    panel.spacing.x = unit(0.45, "lines")
  )

out3 <- file.path(fig, "main_point2_humid_dry_era_lag_profile_strict_pair_5era_wo_legend.svg")
ggsave(
  out3,
  p_strict + theme(strip.text = element_blank(), strip.background = element_blank()),
  width = 4.55,
  height = 2.05,
  bg = "transparent"
)
clean_svg(out3)

