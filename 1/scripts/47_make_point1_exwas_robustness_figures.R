#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(grid)
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

domain_levels <- c(
  "Heat", "Cold", "Severe weather", "Forage condition", "Agricultural pesticides",
  "Feed market", "Milk price / dairy market", "Market demand",
  "Dairy scale", "COVID", "HPAI"
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
  "Dairy scale" = "Dairy scale",
  "COVID" = "COVID-19",
  "HPAI" = "HPAI"
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
  "COVID" = "#d2b48c",
  "HPAI" = "#9d6b53"
)

theme_p1 <- theme_classic(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(size = 9, family = "Arial", face = "plain", color = "#111111"),
    plot.title = element_text(size = 9, family = "Arial", face = "plain"),
    axis.title = element_text(size = 9, family = "Arial", face = "plain"),
    axis.text = element_text(size = 9, family = "Arial", color = "#111111"),
    legend.title = element_text(size = 9, family = "Arial", face = "plain"),
    legend.text = element_text(size = 9, family = "Arial", face = "plain"),
    strip.text = element_text(size = 9, family = "Arial", face = "plain"),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    axis.line = element_line(linewidth = 0.28, color = "#111111"),
    plot.margin = margin(6, 12, 6, 12)
  )

event_representatives <- c(
  "COVID" = "covid_new_cases",
  "HPAI" = "county_sum_hpai_wild_bird_detections"
)

keep_event_representatives <- function(d) {
  d %>%
    filter(
      !(as.character(domain) %in% names(event_representatives)) |
        exposure == event_representatives[as.character(domain)]
    )
}

normalize_domain_names <- function(d) {
  d %>%
    mutate(
      domain = recode(
        as.character(domain),
        "Dairy market" = "Milk price / dairy market"
      )
    )
}

assoc <- read_csv(file.path(tab, "point1_native_only_endpoint_exwas_associations.csv"), show_col_types = FALSE) %>%
  normalize_domain_names() %>%
  filter(
    window == "native",
    phenotype_scope %in% c("total_26", "per_cow_26"),
    domain %in% domain_levels,
    is.finite(plot_p)
  )
if ("measurement_support_variable" %in% names(assoc)) {
  assoc <- assoc %>% filter(!as.logical(coalesce(measurement_support_variable, FALSE)))
}
assoc <- assoc %>%
  keep_event_representatives() %>%
  mutate(domain = factor(domain, levels = domain_levels))

ord <- assoc %>%
  distinct(exposure, domain, source_class) %>%
  arrange(domain, exposure) %>%
  mutate(x_order = row_number())
n_order <- max(ord$x_order, na.rm = TRUE)
domain_gap <- 10
ord <- ord %>%
  mutate(
    domain_rank = as.integer(factor(domain, levels = domain_levels)),
    y_order = n_order + 1 - x_order + (length(domain_levels) - domain_rank) * domain_gap
  )

bounds <- ord %>%
  group_by(domain) %>%
  summarise(ymin = min(y_order), ymax = max(y_order), .groups = "drop") %>%
  arrange(factor(domain, levels = domain_levels)) %>%
  mutate(
    mid = (ymin + ymax) / 2,
    sep_between_previous = (lag(ymin) + ymax) / 2
  )
continuous_domains <- setdiff(domain_levels, c("COVID", "HPAI"))
bounds_cont <- bounds %>% filter(as.character(domain) %in% continuous_domains)
domain_seps_cont <- bounds_cont$sep_between_previous[-1]
top_axis_y_cont <- max(ord$y_order[as.character(ord$domain) %in% continuous_domains], na.rm = TRUE) + 0.5
bottom_axis_y_cont <- min(ord$y_order[as.character(ord$domain) %in% continuous_domains], na.rm = TRUE) - 0.5

loso <- read_csv(file.path(tab, "point1_exwas_domain_matched_loso_beta_robustness.csv"), show_col_types = FALSE) %>%
  normalize_domain_names() %>%
  keep_event_representatives() %>%
  left_join(ord, by = c("exposure", "domain", "source_class")) %>%
  mutate(domain = factor(domain, levels = domain_levels))
if (!"loso_beta_mean" %in% names(loso)) {
  loso <- loso %>%
    mutate(loso_beta_mean = rowMeans(across(c(loso_beta_lo, loso_beta_median, loso_beta_hi)), na.rm = TRUE))
}

spec_range <- read_csv(file.path(tab, "point1_exwas_model_spec_beta_range.csv"), show_col_types = FALSE) %>%
  normalize_domain_names() %>%
  keep_event_representatives() %>%
  left_join(ord, by = c("exposure", "domain", "source_class")) %>%
  mutate(domain = factor(domain, levels = domain_levels))

spec_long <- read_csv(file.path(tab, "point1_exwas_model_spec_beta_long.csv"), show_col_types = FALSE) %>%
  normalize_domain_names() %>%
  filter(status == "ok", is.finite(beta)) %>%
  keep_event_representatives() %>%
  left_join(ord, by = c("exposure", "domain", "source_class")) %>%
  mutate(
    domain = factor(domain, levels = domain_levels),
    spec_id = factor(
      spec_id,
      levels = c(
        "primary",
        "state_month",
        "state_year",
        "state_month_year",
        "state_month_linear_year",
        "state_month_state_trend"
      )
    ),
    spec_label = factor(
      spec_label,
      levels = c(
        "Primary: state + year-month FE",
        "State + month FE",
        "State + year FE",
        "State + month + year FE",
        "State + month FE + linear year trend",
        "State + month FE + state-specific linear trend"
      )
    )
  )

spec_pal <- c(
  "Primary: state + year-month FE" = "#111111",
  "State + month FE" = "#4E79A7",
  "State + year FE" = "#F28E2B",
  "State + month + year FE" = "#59A14F",
  "State + month FE + linear year trend" = "#B07AA1",
  "State + month FE + state-specific linear trend" = "#E15759"
)
spec_labels_compact <- c(
  "State + month FE" = "State+month",
  "State + year FE" = "State+year",
  "State + month + year FE" = "Month+year",
  "State + month FE + linear year trend" = "Linear year",
  "State + month FE + state-specific linear trend" = "State trend"
)

event_sens <- read_csv(file.path(tab, "point1_exwas_event_study_sensitivity.csv"), show_col_types = FALSE) %>%
  mutate(
    domain = factor(domain, levels = domain_levels),
    sensitivity_label = factor(
      sensitivity_label,
      levels = c(
        "Primary definition",
        "Short event window",
        "Long event window",
        "Short baseline",
        "Long baseline",
        "Trend-adjusted baseline"
      )
    )
  )

event_sens_pal <- c(
  "Primary definition" = "#111111",
  "Short event window" = "#4E79A7",
  "Long event window" = "#F28E2B",
  "Short baseline" = "#59A14F",
  "Long baseline" = "#B07AA1",
  "Trend-adjusted baseline" = "#E15759"
)
event_sens_labels_compact <- c(
  "Short event window" = "Short win.",
  "Long event window" = "Long win.",
  "Short baseline" = "Short base",
  "Long baseline" = "Long base",
  "Trend-adjusted baseline" = "Trend adj."
)

y_guides <- list(
  geom_hline(yintercept = domain_seps_cont, linetype = "solid", color = "#777777", linewidth = 0.28),
  geom_hline(yintercept = c(top_axis_y_cont, bottom_axis_y_cont), color = "#111111", linewidth = 0.28)
)

make_loso_endpoint <- function(scope, label, show_y = TRUE) {
  d_loso <- loso %>%
    filter(phenotype_scope == scope, is.finite(y_order), !(as.character(domain) %in% c("COVID", "HPAI")))
  scale_source <- d_loso %>%
    select(loso_beta_lo, loso_beta_hi, loso_beta_median, loso_beta_mean) %>%
    unlist(use.names = FALSE)
  lim <- max(abs(scale_source), na.rm = TRUE)
  lim <- ceiling(lim * 20) / 20
  if (!is.finite(lim) || lim == 0) lim <- 0.1
  y_scale <- scale_y_continuous(
    breaks = bounds_cont$mid,
    labels = if (show_y) domain_labels[as.character(bounds_cont$domain)] else NULL,
    expand = expansion(mult = 0.01)
  )
  ggplot(d_loso, aes(y = y_order)) +
    y_guides +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_errorbarh(
      aes(xmin = loso_beta_lo, xmax = loso_beta_hi, color = domain),
      height = 0,
      linewidth = 0.36,
      alpha = 0.86
    ) +
    geom_point(aes(x = loso_beta_median, fill = domain), shape = 21, size = 1.2, color = "#111111", stroke = 0.12) +
    geom_point(aes(x = loso_beta_mean, fill = domain), shape = 24, size = 1.15, color = "#111111", stroke = 0.12) +
    scale_color_manual(values = domain_pal, guide = "none") +
    scale_fill_manual(values = domain_pal, labels = domain_labels, name = "Domain", drop = FALSE) +
    y_scale +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 4), expand = expansion(mult = c(0.02, 0.02))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(title = label, x = "Leave-one-state-out standardized beta range", y = NULL) +
    theme_p1 +
    theme(
      legend.position = "bottom",
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_loso_event_endpoint <- function(scope, label, show_y = TRUE) {
  d_event <- loso %>%
    filter(phenotype_scope == scope, as.character(domain) %in% c("COVID", "HPAI")) %>%
    mutate(event_y = if_else(as.character(domain) == "COVID", 2, 1))
  lim <- max(abs(c(d_event$loso_beta_lo, d_event$loso_beta_hi, d_event$loso_beta_median)), na.rm = TRUE)
  lim <- ceiling(lim * 1.15 * 10) / 10
  if (!is.finite(lim) || lim == 0) lim <- 1
  ggplot(d_event, aes(y = event_y)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_errorbarh(
      aes(xmin = loso_beta_lo, xmax = loso_beta_hi, color = domain),
      height = 0,
      linewidth = 0.36,
      alpha = 0.86
    ) +
    geom_point(aes(x = loso_beta_median, fill = domain), shape = 21, size = 1.2, color = "#111111", stroke = 0.12) +
    scale_color_manual(values = domain_pal, guide = "none") +
    scale_fill_manual(values = domain_pal, guide = "none") +
    scale_y_continuous(
      breaks = c(2, 1),
      labels = if (show_y) c("COVID-19", "HPAI") else NULL,
      limits = c(0.4, 2.6),
      expand = expansion(mult = 0)
    ) +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 4), expand = expansion(mult = c(0.02, 0.02))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(title = NULL, x = "Event effect (%)", y = NULL) +
    theme_p1 +
    theme(
      legend.position = "none",
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_modelspec_endpoint <- function(scope, label, show_y = TRUE) {
  d_spec <- spec_long %>%
    filter(phenotype_scope == scope, is.finite(y_order), !(as.character(domain) %in% c("COVID", "HPAI")))
  d_range <- d_spec %>%
    group_by(phenotype_scope, domain, source_class, exposure, y_order) %>%
    summarise(
      beta_lo = quantile(beta, 0.025, na.rm = TRUE),
      beta_hi = quantile(beta, 0.975, na.rm = TRUE),
      .groups = "drop"
    )
  scale_source <- bind_rows(
    d_range %>% transmute(domain, value = beta_lo),
    d_range %>% transmute(domain, value = beta_hi),
    d_spec %>% transmute(domain, value = beta)
  ) %>%
    filter(is.finite(value))
  lim <- max(abs(scale_source$value), na.rm = TRUE)
  lim <- ceiling(lim * 20) / 20
  if (!is.finite(lim) || lim == 0) lim <- 0.1
  y_scale <- scale_y_continuous(
    breaks = bounds_cont$mid,
    labels = if (show_y) domain_labels[as.character(bounds_cont$domain)] else NULL,
    expand = expansion(mult = 0.01)
  )
  ggplot() +
    y_guides +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_errorbarh(
      data = d_range,
      aes(y = y_order, xmin = beta_lo, xmax = beta_hi),
      height = 0,
      linewidth = 0.28,
      color = "#8a8a8a",
      alpha = 0.62
    ) +
    geom_point(
      data = d_spec,
      aes(x = beta, y = y_order, fill = spec_label),
      shape = 21,
      size = 0.82,
      color = "#111111",
      stroke = 0.12,
      alpha = 0.92
    ) +
    scale_fill_manual(values = spec_pal, name = "Model specification", drop = FALSE) +
    y_scale +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 4), expand = expansion(mult = c(0.02, 0.02))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(title = label, x = "Model-specification standardized beta range", y = NULL) +
    theme_p1 +
    theme(
      legend.position = "bottom",
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_modelspec_event_endpoint <- function(scope, label, show_y = TRUE) {
  d_event <- event_sens %>%
    filter(
      phenotype_scope == scope,
      as.character(domain) %in% c("COVID", "HPAI"),
      is.finite(effect_pct)
    ) %>%
    mutate(event_y = if_else(as.character(domain) == "COVID", 2, 1))
  lim <- max(abs(d_event$effect_pct), na.rm = TRUE)
  lim <- ceiling(lim * 1.15 * 10) / 10
  if (!is.finite(lim) || lim == 0) lim <- 1
  ggplot(d_event, aes(y = event_y)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_point(
      aes(x = effect_pct, fill = sensitivity_label),
      shape = 21,
      size = 1.05,
      color = "#111111",
      stroke = 0.12,
      alpha = 0.94
    ) +
    scale_fill_manual(values = event_sens_pal, name = "Event-study sensitivity", drop = FALSE) +
    scale_y_continuous(
      breaks = c(2, 1),
      labels = if (show_y) c("COVID-19", "HPAI") else NULL,
      limits = c(0.4, 2.6),
      expand = expansion(mult = 0)
    ) +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 4), expand = expansion(mult = c(0.02, 0.02))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(title = NULL, x = "Event effect (%)", y = NULL) +
    theme_p1 +
    theme(
      legend.position = "bottom",
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_loso_domain_endpoint <- function(scope, dom, label = NULL, show_y = TRUE, show_x_title = FALSE,
                                      xlim_override = NULL) {
  d_loso <- loso %>%
    filter(
      phenotype_scope == scope,
      as.character(domain) == dom,
      is.finite(y_order)
    ) %>%
    arrange(desc(y_order)) %>%
    mutate(
      y_local = row_number(),
      y_mean = y_local + 0.16
    )
  scale_source <- d_loso %>%
    select(loso_beta_lo, loso_beta_hi, loso_beta_median, loso_beta_mean) %>%
    unlist(use.names = FALSE)
  lim <- if (is.null(xlim_override)) max(abs(scale_source), na.rm = TRUE) else xlim_override
  lim <- ceiling(lim * 20) / 20
  if (!is.finite(lim) || lim == 0) lim <- 0.1
  mid <- (min(d_loso$y_local, na.rm = TRUE) + max(d_loso$y_local, na.rm = TRUE)) / 2
  ggplot(d_loso, aes(y = y_local)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_errorbarh(
      aes(xmin = loso_beta_lo, xmax = loso_beta_hi, color = domain),
      height = 0,
      linewidth = 0.36,
      alpha = 0.86
    ) +
    geom_point(aes(x = loso_beta_median, fill = domain), shape = 21, size = 1.2, color = "#111111", stroke = 0.12) +
    geom_point(aes(x = loso_beta_mean, y = y_mean, fill = domain), shape = 24, size = 1.35, color = "#111111", stroke = 0.14) +
    scale_color_manual(values = domain_pal, guide = "none") +
    scale_fill_manual(values = domain_pal, guide = "none") +
    scale_y_continuous(
      breaks = mid,
      labels = if (show_y) domain_labels[[dom]] else NULL,
      expand = expansion(mult = c(0.18, 0.18))
    ) +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 3), expand = expansion(mult = c(0.03, 0.03))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(
      title = label,
      x = if (show_x_title) "LOSO estimate (standardized β or event effect %)" else NULL,
      y = NULL
    ) +
    theme_p1 +
    theme(
      legend.position = "none",
      plot.margin = margin(2, 12, 2, 12),
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_loso_event_domain_endpoint <- function(scope, dom, label = NULL, show_y = TRUE, show_x_title = FALSE,
                                            xlim_override = NULL) {
  d_event <- loso %>%
    filter(
      phenotype_scope == scope,
      as.character(domain) == dom
    ) %>%
    mutate(
      y_local = 1,
      y_mean = y_local + 0.13
    )
  scale_source <- d_event %>%
    select(loso_beta_lo, loso_beta_hi, loso_beta_median, loso_beta_mean) %>%
    unlist(use.names = FALSE)
  lim <- if (is.null(xlim_override)) max(abs(scale_source), na.rm = TRUE) else xlim_override
  lim <- ceiling(lim * 1.15 * 10) / 10
  if (!is.finite(lim) || lim == 0) lim <- 1
  ggplot(d_event, aes(y = y_local)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_errorbarh(
      aes(xmin = loso_beta_lo, xmax = loso_beta_hi, color = domain),
      height = 0,
      linewidth = 0.36,
      alpha = 0.86
    ) +
    geom_point(aes(x = loso_beta_median, fill = domain), shape = 21, size = 1.2, color = "#111111", stroke = 0.12) +
    geom_point(aes(x = loso_beta_mean, y = y_mean, fill = domain), shape = 24, size = 1.35, color = "#111111", stroke = 0.14) +
    scale_color_manual(values = domain_pal, guide = "none") +
    scale_fill_manual(values = domain_pal, guide = "none") +
    scale_y_continuous(
      breaks = 1,
      labels = if (show_y) domain_labels[[dom]] else NULL,
      limits = c(0.6, 1.4),
      expand = expansion(mult = 0)
    ) +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 3), expand = expansion(mult = c(0.03, 0.03))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(
      title = label,
      x = if (show_x_title) "LOSO estimate (standardized β or event effect %)" else NULL,
      y = NULL
    ) +
    theme_p1 +
    theme(
      legend.position = "none",
      plot.margin = margin(2, 12, 2, 12),
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_loso_domain_row <- function(dom, is_first = FALSE, is_last = FALSE) {
  row_fun <- if (dom %in% c("COVID", "HPAI")) make_loso_event_domain_endpoint else make_loso_domain_endpoint
  d_domain <- loso %>%
    filter(
      phenotype_scope %in% c("per_cow_26", "total_26"),
      as.character(domain) == dom
    )
  xlim_domain <- d_domain %>%
    select(loso_beta_lo, loso_beta_hi, loso_beta_median, loso_beta_mean) %>%
    unlist(use.names = FALSE) %>%
    abs() %>%
    max(na.rm = TRUE)
  row_fun("per_cow_26", dom, if (is_first) "Milk per cow" else NULL, TRUE, is_last, xlim_domain) |
    row_fun("total_26", dom, if (is_first) "Total production" else NULL, FALSE, is_last, xlim_domain)
}

loso_rows <- lapply(seq_along(domain_levels), function(i) {
  make_loso_domain_row(
    domain_levels[[i]],
    is_first = i == 1,
    is_last = i == length(domain_levels)
  )
})
names(loso_rows) <- domain_levels
loso_heights <- sapply(domain_levels, function(dom) {
  n_dom <- ord %>% filter(as.character(domain) == dom) %>% nrow()
  if (dom %in% c("COVID", "HPAI")) 0.72 else max(0.86, min(1.75, 0.42 + 0.035 * n_dom))
})
p_loso_combined <- wrap_plots(loso_rows, ncol = 1, heights = loso_heights) &
  theme(legend.position = "none")
loso_out <- file.path(fig, "supp_point1_exwas_loso_robustness_combined.svg")
ggsave(loso_out, p_loso_combined, width = 7.2, height = 9.3, units = "in", bg = "transparent")
clean_svg(loso_out)
loso_wo <- file.path(fig, "supp_point1_exwas_loso_robustness_combined_wo_legend.svg")
ggsave(loso_wo, p_loso_combined & theme(legend.position = "none"), width = 7.2, height = 9.3, units = "in", bg = "transparent")
clean_svg(loso_wo)

make_modelspec_domain_endpoint <- function(scope, dom, label = NULL, show_y = TRUE,
                                           show_x_title = FALSE, show_legend = FALSE,
                                           xlim_override = NULL) {
  d_spec <- spec_long %>%
    filter(
      phenotype_scope == scope,
      as.character(domain) == dom,
      is.finite(y_order)
    ) %>%
    arrange(desc(y_order)) %>%
    mutate(y_local = dense_rank(desc(y_order)))
  d_primary <- d_spec %>% filter(spec_id == "primary")
  d_nonprimary <- d_spec %>% filter(spec_id != "primary")
  d_range <- d_spec %>%
    group_by(phenotype_scope, domain, source_class, exposure, y_order, y_local) %>%
    summarise(
      n_spec = sum(is.finite(beta)),
      beta_mean = mean(beta, na.rm = TRUE),
      beta_se = sd(beta, na.rm = TRUE) / sqrt(n_spec),
      beta_lo = beta_mean - qt(0.975, pmax(n_spec - 1, 1)) * beta_se,
      beta_hi = beta_mean + qt(0.975, pmax(n_spec - 1, 1)) * beta_se,
      .groups = "drop"
    )
  scale_source <- bind_rows(
    d_range %>% transmute(value = beta_lo),
    d_range %>% transmute(value = beta_hi),
    d_spec %>% transmute(value = beta)
  ) %>%
    filter(is.finite(value))
  lim <- if (is.null(xlim_override)) max(abs(scale_source$value), na.rm = TRUE) else xlim_override
  lim <- ceiling(lim * 20) / 20
  if (!is.finite(lim) || lim == 0) lim <- 0.1
  mid <- (min(d_spec$y_local, na.rm = TRUE) + max(d_spec$y_local, na.rm = TRUE)) / 2
  ggplot() +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_errorbarh(
      data = d_range,
      aes(y = y_local, xmin = beta_lo, xmax = beta_hi),
      height = 0,
      linewidth = 0.32,
      color = "#8a8a8a",
      alpha = 0.70
    ) +
    geom_point(
      data = d_nonprimary,
      aes(x = beta, y = y_local, fill = spec_label),
      shape = 21,
      size = 1.05,
      color = "#111111",
      stroke = 0.12,
      alpha = 0.94
    ) +
    geom_point(
      data = d_primary,
      aes(x = beta, y = y_local, fill = domain),
      shape = 24,
      size = 1.25,
      color = "#111111",
      stroke = 0.14,
      alpha = 0.98,
      show.legend = FALSE
    ) +
    scale_fill_manual(
      values = c(spec_pal, domain_pal),
      name = "Model specification",
      breaks = names(spec_labels_compact),
      labels = spec_labels_compact,
      drop = TRUE,
      guide = if (show_legend) guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(size = 2.4)
      ) else "none"
    ) +
    scale_y_continuous(
      breaks = mid,
      labels = if (show_y) domain_labels[[dom]] else NULL,
      expand = expansion(mult = c(0.18, 0.18))
    ) +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 3), expand = expansion(mult = c(0.03, 0.03))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(
      title = label,
      x = if (show_x_title) "Specification estimate\n(standardized β or event effect %)" else NULL,
      y = NULL
    ) +
    theme_p1 +
    theme(
      legend.position = if (show_legend) "bottom" else "none",
      plot.margin = margin(2, 12, 2, 12),
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_modelspec_event_domain_endpoint <- function(scope, dom, label = NULL, show_y = TRUE,
                                                 show_x_title = FALSE, show_legend = FALSE,
                                                 xlim_override = NULL) {
  d_event <- event_sens %>%
    filter(
      phenotype_scope == scope,
      as.character(domain) == dom,
      is.finite(effect_pct)
    ) %>%
    mutate(y_local = 1)
  d_primary <- d_event %>% filter(as.character(sensitivity_label) == "Primary definition")
  d_nonprimary <- d_event %>% filter(as.character(sensitivity_label) != "Primary definition")
  d_range <- d_event %>%
    group_by(domain, y_local) %>%
    summarise(
      n_spec = sum(is.finite(effect_pct)),
      effect_mean = mean(effect_pct, na.rm = TRUE),
      effect_se = sd(effect_pct, na.rm = TRUE) / sqrt(n_spec),
      effect_lo = effect_mean - qt(0.975, pmax(n_spec - 1, 1)) * effect_se,
      effect_hi = effect_mean + qt(0.975, pmax(n_spec - 1, 1)) * effect_se,
      .groups = "drop"
    )
  scale_source <- c(d_event$effect_pct, d_range$effect_lo, d_range$effect_hi)
  lim <- if (is.null(xlim_override)) max(abs(scale_source), na.rm = TRUE) else xlim_override
  lim <- ceiling(lim * 1.15 * 10) / 10
  if (!is.finite(lim) || lim == 0) lim <- 1
  ggplot(d_event, aes(y = y_local)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.25) +
    geom_errorbarh(
      data = d_range,
      aes(y = y_local, xmin = effect_lo, xmax = effect_hi),
      inherit.aes = FALSE,
      height = 0,
      linewidth = 0.32,
      color = "#8a8a8a",
      alpha = 0.70
    ) +
    geom_point(
      data = d_nonprimary,
      aes(x = effect_pct, fill = sensitivity_label),
      shape = 21,
      size = 1.25,
      color = "#111111",
      stroke = 0.12,
      alpha = 0.94
    ) +
    geom_point(
      data = d_primary,
      aes(x = effect_pct, fill = domain),
      shape = 24,
      size = 1.35,
      color = "#111111",
      stroke = 0.14,
      alpha = 0.98,
      show.legend = FALSE
    ) +
    scale_fill_manual(
      values = c(event_sens_pal, domain_pal),
      name = "Event-study sensitivity",
      breaks = names(event_sens_labels_compact),
      labels = event_sens_labels_compact,
      drop = TRUE,
      guide = if (show_legend) guide_legend(
        nrow = 1,
        byrow = TRUE,
        override.aes = list(size = 2.4)
      ) else "none"
    ) +
    scale_y_continuous(
      breaks = 1,
      labels = if (show_y) domain_labels[[dom]] else NULL,
      limits = c(0.6, 1.4),
      expand = expansion(mult = 0)
    ) +
    scale_x_continuous(breaks = pretty(c(-lim, lim), n = 3), expand = expansion(mult = c(0.03, 0.03))) +
    coord_cartesian(xlim = c(-lim, lim), clip = "on") +
    labs(
      title = label,
      x = if (show_x_title) "Specification estimate\n(standardized β or event effect %)" else NULL,
      y = NULL
    ) +
    theme_p1 +
    theme(
      legend.position = if (show_legend) "bottom" else "none",
      plot.margin = margin(2, 12, 2, 12),
      axis.text.y = if (show_y) element_text(size = 9, family = "Arial", color = "#111111") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.25, color = "#111111") else element_blank()
    )
}

make_modelspec_domain_row <- function(dom, is_first = FALSE, is_last = FALSE) {
  row_fun <- if (dom %in% c("COVID", "HPAI")) make_modelspec_event_domain_endpoint else make_modelspec_domain_endpoint
  xlim_domain <- if (dom %in% c("COVID", "HPAI")) {
    event_sens %>%
      filter(
        phenotype_scope %in% c("per_cow_26", "total_26"),
        as.character(domain) == dom,
        is.finite(effect_pct)
      ) %>%
      pull(effect_pct) %>%
      abs() %>%
      max(na.rm = TRUE)
  } else {
    spec_long %>%
      filter(
        phenotype_scope %in% c("per_cow_26", "total_26"),
        as.character(domain) == dom,
        status == "ok",
        is.finite(beta)
      ) %>%
      pull(beta) %>%
      abs() %>%
      max(na.rm = TRUE)
  }
  row_fun(
    "per_cow_26", dom, if (is_first) "Milk per cow" else NULL, TRUE, is_last,
    show_legend = dom == "Heat",
    xlim_override = xlim_domain
  ) |
    row_fun(
      "total_26", dom, if (is_first) "Total production" else NULL, FALSE, is_last,
      show_legend = dom == "COVID",
      xlim_override = xlim_domain
    )
}

model_rows <- lapply(seq_along(domain_levels), function(i) {
  make_modelspec_domain_row(
    domain_levels[[i]],
    is_first = i == 1,
    is_last = i == length(domain_levels)
  )
})
names(model_rows) <- domain_levels
p_modelspec_combined <- wrap_plots(model_rows, ncol = 1, heights = loso_heights) +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "center",
    legend.box.spacing = unit(0, "pt"),
    legend.spacing.y = unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 0),
    legend.key.height = unit(9, "pt")
  )
model_out <- file.path(fig, "supp_point1_exwas_modelspec_robustness_combined.svg")
ggsave(model_out, p_modelspec_combined, width = 8.2, height = 9.3, units = "in", bg = "transparent")
clean_svg(model_out)
model_wo <- file.path(fig, "supp_point1_exwas_modelspec_robustness_combined_wo_legend.svg")
ggsave(model_wo, p_modelspec_combined & theme(legend.position = "none"), width = 8.2, height = 9.3, units = "in", bg = "transparent")
clean_svg(model_wo)

weighting <- read_csv(file.path(tab, "point1_exwas_weighting_robustness_beta.csv"), show_col_types = FALSE) %>%
  filter(
    analysis %in% c("unweighted_model", "state_equal_model"),
    main_status == "ok",
    sensitivity_status == "ok",
    is.finite(main_beta),
    is.finite(sensitivity_beta)
  ) %>%
  mutate(
    domain = factor(domain, levels = domain_levels),
    phenotype_label = recode(phenotype_scope, total_26 = "Total production", per_cow_26 = "Milk per cow"),
    analysis_label = recode(
      analysis,
      unweighted_model = "Unweighted model",
      state_equal_model = "State-equal model"
    )
  )

stats <- weighting %>%
  group_by(analysis_label) %>%
  summarise(
    same_sign = mean(sign_concordant, na.rm = TRUE),
    cor = suppressWarnings(cor(main_beta, sensitivity_beta, use = "complete.obs")),
    p_value = suppressWarnings(cor.test(main_beta, sensitivity_beta)$p.value),
    .groups = "drop"
  ) %>%
  mutate(
    p_label = if_else(p_value < 0.001, "P < 0.001", sprintf("P = %.3f", p_value)),
    label = sprintf("R² = %.2f, %s\nsame sign = %.0f%%", cor^2, p_label, 100 * same_sign)
  )

make_weight_panel <- function(analysis_name) {
  d <- weighting %>% filter(analysis_label == analysis_name)
  ann <- stats %>% filter(analysis_label == analysis_name)
  lim <- max(abs(c(d$main_beta, d$sensitivity_beta)), na.rm = TRUE)
  lim <- ceiling(lim * 20) / 20
  if (!is.finite(lim) || lim == 0) lim <- 0.1
  ann <- ann %>% mutate(x = -lim * 0.95, y = lim * 0.95)
  ggplot(d, aes(main_beta, sensitivity_beta, color = domain, shape = phenotype_label)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.28) +
    geom_hline(yintercept = 0, color = "#d0d0d0", linewidth = 0.2) +
    geom_vline(xintercept = 0, color = "#d0d0d0", linewidth = 0.2) +
    geom_point(size = 1.8, alpha = 0.86) +
    geom_text(data = ann, aes(x = x, y = y, label = label), inherit.aes = FALSE,
              hjust = 0, vjust = 1, size = 3, family = "Arial", color = "#222222") +
    scale_color_manual(values = domain_pal, labels = domain_labels, name = "Domain", drop = TRUE) +
    scale_shape_manual(values = c("Milk per cow" = 16, "Total production" = 1), name = "Phenotype") +
    guides(color = guide_legend(nrow = 1, byrow = TRUE), shape = guide_legend(nrow = 1)) +
    coord_equal(xlim = c(-lim, lim), ylim = c(-lim, lim), clip = "off") +
    labs(title = analysis_name, x = "Primary cow-weighted estimate (β)", y = "Sensitivity-model estimate (β)") +
    theme_p1 +
    theme(
      legend.position = "bottom",
      panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.28),
      axis.line = element_blank()
    )
}

p_unweighted <- make_weight_panel("Unweighted model")
p_state_equal <- make_weight_panel("State-equal model")
p_weight <- (p_unweighted | p_state_equal) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
weight_out <- file.path(fig, "supp_point1_exwas_weighting_robustness.svg")
ggsave(weight_out, p_weight, width = 9.2, height = 4.8, units = "in", bg = "transparent")
clean_svg(weight_out)
weight_wo <- file.path(fig, "supp_point1_exwas_weighting_robustness_wo_legend.svg")
ggsave(weight_wo, p_weight & theme(legend.position = "none"), width = 9.2, height = 4.8, units = "in", bg = "transparent")
clean_svg(weight_wo)

message("Wrote Point 1 ExWAS robustness figures.")
