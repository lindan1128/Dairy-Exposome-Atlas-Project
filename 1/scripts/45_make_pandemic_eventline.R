#!/usr/bin/env Rscript

# Simple event-line view of COVID and HPAI milk responses.
# Row 1 = COVID monthly anomaly around 2020; Row 2 = HPAI event-time profile.
# Columns = endpoints (milk per cow, total production). CI ribbons + event shading.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
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

endpoint_pal <- c(
  "Milk per cow" = "#CF625D",
  "Total production" = "#4C93AD"
)
endpoint_levels <- c("Milk per cow", "Total production")

recode_endpoint <- function(x) {
  factor(
    recode(
      x,
      "milk_per_cow_kg" = "Milk per cow",
      "milk_production_million_kg" = "Total production",
      "milk_production_from_cows_million_kg" = "Total production"
    ),
    levels = endpoint_levels
  )
}

base_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(size = 9, face = "plain", color = "#222222"),
    plot.title = element_text(size = 9, face = "plain"),
    axis.title = element_text(size = 9, face = "plain"),
    axis.text = element_text(size = 9, color = "#222222"),
    strip.text = element_text(size = 9, face = "plain"),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.18, color = "#eeeeee"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.28)
  )

# ---- COVID monthly anomaly, zoomed to 2017-2023 ----
covid <- read_csv(file.path(tab, "point1_pandemic_covid_calendar_profile.csv"), show_col_types = FALSE) %>%
  filter(outcome %in% c("milk_per_cow_kg", "milk_production_million_kg")) %>%
  mutate(date = as.Date(date), endpoint = recode_endpoint(outcome)) %>%
  filter(date >= as.Date("2018-01-01"), date <= as.Date("2024-12-31"))

p_covid <- ggplot(covid, aes(date, mean_anomaly_pct, color = endpoint, fill = endpoint)) +
  annotate("rect", xmin = as.Date("2020-03-01"), xmax = as.Date("2020-06-30"),
           ymin = -Inf, ymax = Inf, fill = "#f3d5c8", alpha = 0.55) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#666666", linewidth = 0.28) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.13, color = NA) +
  geom_line(linewidth = 0.6) +
  facet_wrap(~ endpoint, nrow = 1, scales = "free_y") +
  scale_color_manual(values = endpoint_pal, drop = FALSE) +
  scale_fill_manual(values = endpoint_pal, drop = FALSE) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y", limits = c(as.Date("2018-01-01"), as.Date("2025-01-01"))) +
  labs(title = "a   COVID-19-era deviation from the 2010-2019 pre-pandemic trend baseline",
       x = "Year", y = "Change relative to baseline (%)") +
  base_theme

# ---- HPAI event-time profile ----
hpai <- read_csv(file.path(tab, "point1_pandemic_hpai_event_profile.csv"), show_col_types = FALSE) %>%
  filter(outcome %in% c("milk_per_cow_kg", "milk_production_million_kg")) %>%
  mutate(endpoint = recode_endpoint(outcome))

p_hpai <- ggplot(hpai, aes(tau, mean_effect_pct, color = endpoint, fill = endpoint)) +
  annotate("rect", xmin = 0, xmax = 2, ymin = -Inf, ymax = Inf, fill = "#f3d5c8", alpha = 0.55) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#666666", linewidth = 0.28) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "#555555", linewidth = 0.3) +
  geom_ribbon(aes(ymin = ci_low, ymax = ci_high), alpha = 0.13, color = NA) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.3) +
  facet_wrap(~ endpoint, nrow = 1, scales = "free_y") +
  scale_color_manual(values = endpoint_pal, drop = FALSE) +
  scale_fill_manual(values = endpoint_pal, drop = FALSE) +
  scale_x_continuous(breaks = seq(-6, 6, by = 2)) +
  labs(title = "b   Milk response around first dairy-HPAI detection",
       x = "Months relative to first dairy-HPAI detection",
       y = "Change relative to baseline (%)") +
  base_theme

combined <- p_covid / p_hpai + plot_layout(heights = c(1, 1))

out <- file.path(fig, "supp_point1_pandemic_eventline.svg")
ggsave(out, combined, width = 9, height = 6.6, units = "in", bg = "transparent")
clean_svg(out)
message("Wrote ", out)
