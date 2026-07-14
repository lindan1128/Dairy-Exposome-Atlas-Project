#!/usr/bin/env Rscript

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

domain_levels <- c(
  "Heat", "Cold", "Severe weather", "Forage condition", "Agricultural pesticides",
  "Feed market", "Milk price / dairy market", "Market demand",
  "Dairy scale"
)
domain_labels <- c(
  "Heat" = "Heat",
  "Cold" = "Cold",
  "Severe weather" = "Severe weather",
  "Agricultural pesticides" = "Pesticides",
  "Forage condition" = "Forage",
  "Feed market" = "Feed market",
  "Milk price / dairy market" = "Dairy market",
  "Market demand" = "Market demand",
  "Dairy scale" = "Dairy scale",
  "Herd structure / scale" = "Dairy scale"
)
domain_pal <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage condition" = "#1E7A8D",
  "Agricultural pesticides" = "#c79fa8",
  "Feed market" = "#fbc4ab",
  "Milk price / dairy market" = "#E47666",
  "Market demand" = "#f09d51",
  "Dairy scale" = "#fec89a",
  "Herd structure / scale" = "#fec89a"
)
endpoint_labels <- c(
  "total_26" = "Total production",
  "per_cow_26" = "Milk per cow"
)
endpoint_pal <- c(
  "Total production" = "#69A7BE",
  "Milk per cow" = "#EF7E79"
)
endpoint_line_pal <- c(
  "Total production" = "#4C93AD",
  "Milk per cow" = "#CF625D"
)
endpoint_lty <- c(
  "Total production" = "solid",
  "Milk per cow" = "22"
)

nice4_breaks <- function(lims) {
  top <- max(lims, na.rm = TRUE)
  if (!is.finite(top) || top <= 0) {
    return(c(0, 0.05, 0.10, 0.15))
  }
  candidates <- c(0.005, 0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25, 0.50, 1.00)
  step <- candidates[which(candidates >= top / 3)[1]]
  if (!is.finite(step)) {
    step <- ceiling(top / 3)
  }
  seq(0, by = step, length.out = 4)
}

d <- read_csv(file.path(tab, "point1_chord_signal_yearly_domain_summary.csv"), show_col_types = FALSE) %>%
  filter(domain %in% domain_levels, phenotype_scope %in% names(endpoint_labels)) %>%
  mutate(
    domain = factor(domain, levels = domain_levels),
    domain_label = factor(domain_labels[as.character(domain)],
                          levels = domain_labels[domain_levels]),
    phenotype_scope = factor(phenotype_scope, levels = c("total_26", "per_cow_26"),
                             labels = endpoint_labels[c("total_26", "per_cow_26")]),
    year_plot = year + ifelse(phenotype_scope == "Total production", -0.10, 0.10),
    endpoint_line = factor(paste0(as.character(phenotype_scope), " trend"),
                           levels = paste0(names(endpoint_line_pal), " trend"))
  )

trend_summary <- d %>%
  group_by(phenotype_scope, domain) %>%
  summarise(
    n_years = n_distinct(year),
    n_signal_variables = max(n_signal_variables, na.rm = TRUE),
    mean_annual_abs_beta = mean(median_abs_beta, na.rm = TRUE),
    slope_abs_beta_per_year = ifelse(
      n() >= 8 && sd(year, na.rm = TRUE) > 0,
      coef(lm(median_abs_beta ~ year))[["year"]],
      NA_real_
    ),
    .groups = "drop"
  ) %>%
  mutate(domain = as.character(domain), domain_label = domain_labels[domain])
write_csv(trend_summary, file.path(tab, "point1_chord_signal_yearly_domain_trend_summary.csv"))

p_stars <- function(p) {
  case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ ""
  )
}

trend_labels <- d %>%
  group_by(domain_label, phenotype_scope, endpoint_line) %>%
  summarise(
    r2 = ifelse(n() >= 3 && sd(year, na.rm = TRUE) > 0,
                summary(lm(median_abs_beta ~ year))$r.squared,
                NA_real_),
    p_trend = ifelse(n() >= 3 && sd(year, na.rm = TRUE) > 0,
                     coef(summary(lm(median_abs_beta ~ year)))["year", "Pr(>|t|)"],
                     NA_real_),
    ymin_panel = min(median_abs_beta, na.rm = TRUE),
    ymax_panel = max(median_abs_beta, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    stars = p_stars(p_trend),
    label = ifelse(
      domain_label == "Dairy scale",
      paste0("R²=", sprintf("%.2f", r2), stars),
      paste0(ifelse(stars == "", "", paste0(stars, " ")), "R²=", sprintf("%.2f", r2))
    ),
    x = ifelse(domain_label == "Dairy scale", 2000.6, 2024.3),
    y = ifelse(domain_label == "Dairy scale", 0, Inf),
    hjust_label = ifelse(domain_label == "Dairy scale", 0, 1),
    vjust_label = case_when(
      domain_label == "Dairy scale" & phenotype_scope == "Total production" ~ -2.35,
      domain_label == "Dairy scale" & phenotype_scope == "Milk per cow" ~ -1.35,
      phenotype_scope == "Total production" ~ 1.35,
      TRUE ~ 2.35
    )
  )

base_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(size = 9, face = "plain", color = "#222222"),
    plot.title = element_text(size = 9, face = "plain"),
    axis.title = element_text(size = 9, face = "plain"),
    axis.text = element_text(size = 9, color = "#222222"),
    strip.text = element_text(size = 9, face = "plain"),
    legend.title = element_text(size = 9, face = "plain"),
    legend.text = element_text(size = 9, face = "plain"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    axis.line = element_line(linewidth = 0.28, color = "#111111"),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.28)
  )

p <- ggplot(d, aes(year_plot, median_abs_beta, color = domain, group = domain)) +
  geom_point(aes(color = phenotype_scope),
             shape = 16, size = 2, stroke = 0, alpha = 0.88) +
  geom_smooth(aes(x = year, y = median_abs_beta, color = endpoint_line,
                  group = phenotype_scope),
              method = "lm", formula = y ~ x, se = FALSE,
              linewidth = 1.62, linetype = "solid", alpha = 1) +
  geom_text(
    data = trend_labels,
    aes(x = x, y = y, label = label, color = endpoint_line,
        vjust = vjust_label, hjust = hjust_label),
    inherit.aes = FALSE,
    size = 2.6,
    fontface = "plain",
    show.legend = FALSE
  ) +
  facet_wrap(~ domain_label, ncol = 3, scales = "free_y", axes = "all_x", axis.labels = "margins") +
  scale_color_manual(
    values = c(endpoint_pal, setNames(endpoint_line_pal, paste0(names(endpoint_line_pal), " trend"))),
    name = NULL,
    drop = FALSE
  ) +
  scale_x_continuous(breaks = seq(2000, 2025, by = 5), limits = c(1999.75, 2025.25)) +
  scale_y_continuous(
    limits = c(0, NA),
    breaks = nice4_breaks,
    labels = function(x) sprintf("%.2f", x)
  ) +
  labs(
    title = "Year-specific association strength by domain and endpoint",
    x = NULL,
    y = "Median |standardized beta|"
  ) +
  base_theme +
  theme(
    legend.position = "bottom",
    panel.spacing.x = unit(0.45, "lines"),
    panel.spacing.y = unit(0.55, "lines"),
    axis.text.x = element_text(size = 9, angle = 90, hjust = 1, vjust = 0.5),
    strip.background = element_blank()
  )

out <- file.path(fig, "main_point1_chord_signal_yearly_association_trends.svg")
ggsave(out, p, width = 4.65, height = 4.5, units = "in", bg = "transparent")
clean_svg(out)

p_wo <- p + labs(title = NULL, x = NULL, y = NULL) + theme(legend.position = "none")
out_wo <- file.path(fig, "main_point1_chord_signal_yearly_association_trends_wo_legend.svg")
ggsave(out_wo, p_wo, width = 4.65, height = 4.5, units = "in", bg = "transparent")
clean_svg(out_wo)

message("Wrote chord-signal yearly association trend figures.")
