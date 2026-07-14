#!/usr/bin/env Rscript

# Leave-one-year-out (LOYO) robustness of the Fig 2c yearly trends (milk per cow).
# Each panel: yearly median |beta| points, full-data trend (bold), and the 26
# leave-one-year-out refits (faint). If the faint lines track the bold one, the
# trend does not depend on any single year.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
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

dom_levels <- c("Heat", "Cold", "Severe weather", "Forage condition",
                "Agricultural pesticides", "Feed market",
                "Milk price / dairy market", "Market demand",
                "Dairy scale")
dom_labels <- c("Heat", "Cold", "Severe weather", "Forage", "Pesticides",
                "Feed market", "Dairy market", "Market demand", "Dairy scale")
names(dom_labels) <- dom_levels
dom_pal <- c("Heat" = "#32a4b4", "Cold" = "#33c5b2", "Severe weather" = "#d5eada",
             "Forage condition" = "#1E7A8D", "Agricultural pesticides" = "#c79fa8",
             "Feed market" = "#fbc4ab", "Milk price / dairy market" = "#E47666",
             "Market demand" = "#f09d51", "Dairy scale" = "#fec89a")

pts <- read_csv(file.path(tab, "point1_chord_signal_yearly_domain_summary.csv"),
                show_col_types = FALSE) %>%
  filter(phenotype_scope == "per_cow_26", domain %in% dom_levels) %>%
  mutate(domain = factor(domain, levels = dom_levels),
         dlab = factor(dom_labels[as.character(domain)], levels = dom_labels))

year_ranges <- pts %>%
  filter(is.finite(median_abs_beta)) %>%
  group_by(domain) %>%
  summarise(x_min = min(year), x_max = max(year),
            y_data_min = min(median_abs_beta),
            y_data_max = max(median_abs_beta),
            .groups = "drop")

loyo <- read_csv(file.path(tab, "point1_yearly_trend_loyo_lines.csv"),
                 show_col_types = FALSE) %>%
  filter(phenotype == "Milk per cow", domain %in% dom_levels) %>%
  mutate(dlab = factor(dom_labels[domain], levels = dom_labels)) %>%
  left_join(year_ranges, by = "domain") %>%
  mutate(y_min = intercept + slope * x_min,
         y_max = intercept + slope * x_max)

summ <- read_csv(file.path(tab, "point1_yearly_trend_loyo_summary.csv"),
                 show_col_types = FALSE) %>%
  filter(phenotype == "Milk per cow", domain %in% dom_levels) %>%
  left_join(year_ranges, by = "domain") %>%
  mutate(dlab = factor(dom_labels[domain], levels = dom_labels),
         full_r = sign(full_slope) * sqrt(full_r2),
         p_label = if_else(full_p < 0.001, "P < 0.001",
                           sprintf("P = %.3f", full_p)),
         y_min = full_intercept + full_slope * x_min,
         y_max = full_intercept + full_slope * x_max,
         y_span = pmax(y_data_max - y_data_min, 0.01),
         y_text1 = y_data_max - 0.08 * y_span,
         y_text2 = y_data_max - 0.18 * y_span,
         y_text3 = y_data_max - 0.28 * y_span,
         label_r = sprintf("R² = %.2f, %s", full_r2, p_label),
         label_r2 = sprintf("LOYO~R^2~'%.2f-%.2f'", loyo_r2_min, loyo_r2_max),
         label_sign = sprintf("sign %.0f%% stable", 100 * sign_stable_share))

base_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(size = 9, color = "#222222"),
    plot.title = element_text(size = 9),
    axis.title = element_text(size = 9),
    axis.text = element_text(size = 8, color = "#222222"),
    strip.text = element_text(size = 9),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.18, color = "#eeeeee"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.28)
  )

p <- ggplot() +
  geom_segment(data = loyo, aes(x = x_min, xend = x_max, y = y_min, yend = y_max),
               color = "#999999", linewidth = 0.22, alpha = 0.55) +
  geom_segment(data = summ, aes(x = x_min, xend = x_max, y = y_min, yend = y_max,
                                color = domain), linewidth = 0.9) +
  geom_point(data = pts, aes(year, median_abs_beta, color = domain),
             size = 1.1, alpha = 0.9) +
  geom_text(data = summ, aes(x = 2000.5, y = y_text1, label = label_r),
            inherit.aes = FALSE, hjust = 0, vjust = 0.5,
            size = 3.0, color = "#333333") +
  geom_text(data = summ, aes(x = 2000.5, y = y_text2, label = label_r2),
            inherit.aes = FALSE, parse = TRUE, hjust = 0, vjust = 0.5,
            size = 3.0, color = "#333333") +
  geom_text(data = summ, aes(x = 2000.5, y = y_text3, label = label_sign),
            inherit.aes = FALSE, hjust = 0, vjust = 0.5,
            size = 3.0, color = "#333333") +
  facet_wrap(~ dlab, ncol = 3, scales = "free_y") +
  scale_color_manual(values = dom_pal) +
  scale_x_continuous(breaks = c(2000, 2010, 2020), limits = c(2000, 2025)) +
  labs(title = "Leave-one-year-out robustness of yearly milk-per-cow exposure sensitivity",
       x = "Year", y = "Yearly median exposure sensitivity (|standardized β|)") +
  base_theme

out <- file.path(fig, "supp_point1_yearly_trend_loyo.svg")
ggsave(out, p, width = 9, height = 7.5, units = "in", bg = "transparent")
clean_svg(out)
message("Wrote ", out)
