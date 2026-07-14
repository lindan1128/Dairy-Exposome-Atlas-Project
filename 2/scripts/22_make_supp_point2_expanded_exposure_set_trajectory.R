#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
  library(tidyr)
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
dir_cols <- c(
  "> 2000 level" = "#B1C99C",
  "< 2000 level" = "#FFCC8D",
  "2000 baseline" = "#BDBDBD"
)

expanded_idx <- read_csv(
  file.path(tab, "point2_exposure_signal_vs_all_clean_translation_index.csv"),
  show_col_types = FALSE
) %>%
  filter(exposure_pool == "All clean native variables", domain %in% domain_levels) %>%
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
  group_by(domain) %>%
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

highlight_rows <- data.frame(
  xmin = which(levels(expanded_idx$year_f) == "2001") - 0.5,
  xmax = length(levels(expanded_idx$year_f)) + 0.5,
  ymin = match(c("Heat", "Severe weather"), levels(expanded_idx$domain_label)) - 0.5,
  ymax = match(c("Heat", "Severe weather"), levels(expanded_idx$domain_label)) + 0.5
)

base_theme <- theme_bw(base_size = 9) +
  theme(
    text = element_text(size = 9, face = "plain", color = "#222222"),
    axis.title = element_text(size = 9, face = "plain", color = "#222222"),
    axis.title.x.top = element_text(size = 9, face = "plain", color = "#222222", margin = margin(b = 5)),
    axis.title.x.bottom = element_blank(),
    axis.title.y = element_blank(),
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
    plot.margin = margin(t = 2, r = 7, b = 4, l = 7)
  )

panel_a <- ggplot(expanded_idx, aes(year_f, domain_label)) +
  annotate(
    "rect",
    xmin = seq(1.5, length(unique(expanded_idx$year_f)) - 0.5, by = 1),
    xmax = seq(2.5, length(unique(expanded_idx$year_f)) + 0.5, by = 1),
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
    xintercept = seq(1.5, length(unique(expanded_idx$year_f)) - 0.5, by = 1),
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
  labs(x = "Year") +
  guides(
    fill = guide_legend(override.aes = list(size = 3.4, alpha = 1), order = 1),
    size = guide_legend(override.aes = list(fill = "#111111", color = "#111111", alpha = 1), order = 2)
  ) +
  base_theme

component_order <- c("Humid heat", "Dry heat", "Night heat", "Flood", "Fire")
plot_component_order <- c("Humid heat", "Dry heat", "Flood", "Fire")

heat_clean <- read_csv(
  file.path(tab, "point2_heat_clean_full_paired_translation_trajectory.csv"),
  show_col_types = FALSE
)

heat_clean_full <- heat_clean %>%
  filter(
    pool == "clean_full_heat",
    subform %in% c("Humid/wet-bulb heat", "Dry/VPD heat", "Night/no-relief heat")
  ) %>%
  mutate(
    component = recode(
      subform,
      "Humid/wet-bulb heat" = "Humid heat",
      "Dry/VPD heat" = "Dry heat",
      "Night/no-relief heat" = "Night heat"
    )
  ) %>%
  transmute(component, year, exposure_burden = exposure_burden_index_2000, loss_translation)

severe_clean <- read_csv(
  file.path(tab, "point2_severe_type_exposure_translation_trajectory.csv"),
  show_col_types = FALSE
) %>%
  filter(pool == "clean natural type events", disaster_type %in% c("Flood events", "Fire events")) %>%
  mutate(
    component = recode(
      disaster_type,
      "Flood events" = "Flood",
      "Fire events" = "Fire"
    )
  ) %>%
  transmute(component, year, exposure_burden = exposure_burden_index_2000, loss_translation)

expanded_traj <- bind_rows(heat_clean_full, severe_clean) %>%
  mutate(component = factor(component, levels = component_order)) %>%
  group_by(component) %>%
  mutate(
    loss_translation_index = {
      baseline_year <- if (as.character(component[1]) == "Fire") 2015 else 2000
      base <- loss_translation[year == baseline_year][1]
      if (is.finite(base) & base != 0) {
        out <- loss_translation / base
        if (as.character(component[1]) == "Fire") {
          out[year < baseline_year] <- NA_real_
        }
        out
      } else {
        NA_real_
      }
    }
  ) %>%
  ungroup()

three_nice_half_breaks <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(c(0, 0.5, 1.0))
  rng <- range(x)
  if (!is.finite(rng[1]) || !is.finite(rng[2])) return(c(0, 0.5, 1.0))
  if (rng[1] == rng[2]) {
    mid <- round(rng[1] * 2) / 2
    return(mid + c(-0.5, 0, 0.5))
  }
  raw_step <- (rng[2] - rng[1]) / 2
  steps <- c(0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100)
  for (step in steps) {
    if (step < raw_step) next
    high <- ceiling(rng[2] / step) * step
    low <- high - 2 * step
    if (low <= rng[1]) return(c(low, low + step, high))
  }
  high <- ceiling(rng[2] / 100) * 100
  c(high - 200, high - 100, high)
}

panel_scale <- function(x, ticks, out_min, out_max) {
  if (length(ticks) < 2 || !all(is.finite(ticks)) || ticks[1] == ticks[length(ticks)]) {
    return(rep((out_min + out_max) / 2, length(x)))
  }
  out_min + (x - ticks[1]) / (ticks[length(ticks)] - ticks[1]) * (out_max - out_min)
}

p_label <- function(p) {
  if (!is.finite(p)) {
    return("P = NA")
  }
  if (p < 0.001) {
    "P < 0.001"
  } else {
    sprintf("P = %.3f", p)
  }
}

trend_label_row <- function(d, metric_name, y_pos) {
  z <- d %>% filter(metric == metric_name, is.finite(year), is.finite(value))
  if (nrow(z) < 3 || sd(z$year, na.rm = TRUE) == 0 || sd(z$value, na.rm = TRUE) == 0) {
    return(tibble(metric = metric_name, label = NA_character_, y = y_pos))
  }
  ct <- suppressWarnings(cor.test(z$year, z$value))
  tibble(
    metric = metric_name,
    label = sprintf("R² = %.2f, %s", unname(ct$estimate)^2, p_label(ct$p.value)),
    y = y_pos
  )
}

make_component_block <- function(d, component_name, top_limits_global) {
  exposure_color <- "#4d908e"
  translation_color <- "#274c77"
  component_label <- recode(
    component_name,
    "Flood" = "Flood events",
    "Fire" = "Fire events",
    .default = component_name
  )
  top <- d %>% filter(component == component_name, metric == "Exposure burden")
  bottom <- d %>% filter(component == component_name, metric == "Loss translation")

  top_ticks <- c(0.8, 1.0, 1.2)
  bottom_ticks <- three_nice_half_breaks(bottom$value)
  top_limits <- top_limits_global
  bottom_limits <- range(c(bottom$value, bottom_ticks), na.rm = TRUE)
  if (!all(is.finite(bottom_limits)) || bottom_limits[1] == bottom_limits[2]) {
    center <- ifelse(is.finite(bottom_limits[1]), bottom_limits[1], 0)
    bottom_limits <- center + c(-0.5, 0.5)
  }
  top_min <- 0.58
  top_max <- 0.96
  bottom_min <- 0.07
  bottom_max <- 0.45
  split_y <- 0.52
  x_ticks <- seq(2000, 2025, by = 5)
  x_min <- 1999.3
  x_max <- 2025.7
  top_baseline_y <- panel_scale(1, top_limits, top_min, top_max)
  bottom_baseline_y <- panel_scale(1, bottom_limits, bottom_min, bottom_max)

  line_df <- bind_rows(
    top %>% transmute(year, value, panel = "top", y = panel_scale(value, top_limits, top_min, top_max)),
    bottom %>% transmute(year, value, panel = "bottom", y = panel_scale(value, bottom_limits, bottom_min, bottom_max))
  )

  stat_labels <- bind_rows(
    trend_label_row(d %>% filter(component == component_name), "Exposure burden", top_max - 0.018),
    trend_label_row(d %>% filter(component == component_name), "Loss translation", bottom_max - 0.018)
  ) %>%
    filter(!is.na(label)) %>%
    mutate(
      x = x_min + 0.45,
      y = ifelse(metric == "Exposure burden", top_max - 0.018, bottom_max - 0.018)
    )

  smooth_df <- line_df %>%
    group_by(panel) %>%
    group_modify(function(.x, .y) {
      .x <- .x %>% filter(is.finite(year), is.finite(y))
      if (nrow(.x) < 3) {
        return(tibble(year = numeric(0), y = numeric(0), ymin = numeric(0), ymax = numeric(0)))
      }
      pred_year <- seq(min(.x$year), max(.x$year), length.out = 80)
      fit <- lm(y ~ year, data = .x)
      pred <- predict(fit, newdata = tibble(year = pred_year), interval = "confidence")
      tibble(
        year = pred_year,
        y = as.numeric(pred[, "fit"]),
        ymin = as.numeric(pred[, "lwr"]),
        ymax = as.numeric(pred[, "upr"])
      )
    }) %>%
    ungroup()

  y_axis <- bind_rows(
    tibble(side = "left", value = top_ticks,
           y = panel_scale(top_ticks, top_limits, top_min, top_max),
           label = label_number(accuracy = 0.1)(top_ticks)),
    tibble(side = "right", value = bottom_ticks,
           y = panel_scale(bottom_ticks, bottom_limits, bottom_min, bottom_max),
           label = label_number(accuracy = 0.1)(bottom_ticks))
  )

  ggplot() +
    annotate(
      "rect",
      xmin = x_min,
      xmax = x_max,
      ymin = 1.035,
      ymax = 1.125,
      fill = "#F1F1F2",
      color = "#111111",
      linewidth = 0.28
    ) +
    annotate(
      "text",
      x = (x_min + x_max) / 2,
      y = 1.080,
      label = component_label,
      size = 9 / .pt,
      color = "#222222"
    ) +
    geom_vline(xintercept = x_ticks, color = "#e8e8e8", linewidth = 0.18) +
    geom_hline(yintercept = y_axis$y, color = "#e8e8e8", linewidth = 0.18) +
    geom_hline(yintercept = split_y, color = "#e8e8e8", linewidth = 0.22) +
    geom_segment(aes(x = x_min, xend = x_max, y = top_baseline_y, yend = top_baseline_y),
                 color = "#9e9e9e", linewidth = 1, linetype = "dashed") +
    geom_segment(aes(x = x_min, xend = x_max, y = bottom_baseline_y, yend = bottom_baseline_y),
                 color = "#9e9e9e", linewidth = 1, linetype = "dashed") +
    geom_ribbon(data = smooth_df, aes(year, ymin = ymin, ymax = ymax, fill = panel),
                alpha = 0.22, color = NA, na.rm = TRUE) +
    geom_point(data = line_df, aes(year, y, fill = panel),
               shape = 21, color = "#111111", size = 2.1, stroke = 0.14, na.rm = TRUE) +
    geom_line(data = smooth_df, aes(year, y, group = panel, color = panel),
              linewidth = 0.65, na.rm = TRUE) +
    geom_text(
      data = stat_labels,
      aes(x = x, y = y, label = label),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 1,
      size = 3.0,
      color = "#111111"
    ) +
    scale_color_manual(values = c("top" = exposure_color, "bottom" = translation_color), guide = "none") +
    scale_fill_manual(values = c("top" = exposure_color, "bottom" = translation_color), guide = "none") +
    geom_segment(aes(x = x_min, xend = x_max, y = 0, yend = 0),
                 color = "#111111", linewidth = 0.28) +
    geom_segment(aes(x = x_min, xend = x_max, y = 1, yend = 1),
                 color = "#111111", linewidth = 0.28) +
    geom_segment(aes(x = x_min, xend = x_min, y = split_y, yend = 1),
                 color = "#111111", linewidth = 0.28) +
    geom_segment(aes(x = x_max, xend = x_max, y = 0, yend = split_y),
                 color = "#111111", linewidth = 0.28) +
    geom_segment(data = y_axis %>% filter(side == "left"),
                 aes(x = x_min, xend = x_min - 0.35, y = y, yend = y),
                 color = "#111111", linewidth = 0.22) +
    geom_segment(data = y_axis %>% filter(side == "right"),
                 aes(x = x_max, xend = x_max + 0.35, y = y, yend = y),
                 color = "#111111", linewidth = 0.22) +
    geom_text(data = y_axis %>% filter(side == "left"),
              aes(x = x_min - 0.55, y = y, label = label),
              hjust = 1, size = 9 / .pt, color = "#222222") +
    geom_text(data = y_axis %>% filter(side == "right"),
              aes(x = x_max + 0.55, y = y, label = label),
              hjust = 0, size = 9 / .pt, color = "#222222") +
    geom_segment(data = tibble(x = x_ticks),
                 aes(x = x, xend = x, y = 0, yend = -0.035),
                 color = "#111111", linewidth = 0.22) +
    geom_text(data = tibble(x = x_ticks, label = as.character(x_ticks)),
              aes(x = x, y = -0.07, label = label),
              angle = 90, hjust = 1, vjust = 0.5,
              size = 9 / .pt, color = "#222222") +
    coord_cartesian(
      xlim = c(x_min - 1.0, x_max + 1.35),
      ylim = c(-0.14, 1.14),
      clip = "off"
    ) +
    theme_void(base_size = 9) +
    theme(
      text = element_text(size = 9, face = "plain", color = "#222222"),
      plot.background = element_rect(fill = "transparent", color = NA),
      plot.margin = margin(t = 2, r = 15, b = 12, l = 16)
    )
}

expanded_plot_df <- expanded_traj %>%
  select(component, year, exposure_burden, loss_translation_index) %>%
  filter(component %in% plot_component_order) %>%
  complete(component = factor(plot_component_order, levels = plot_component_order), year = 2000:2025) %>%
  mutate(component = factor(component, levels = plot_component_order))

panel_b_df <- bind_rows(
  expanded_plot_df %>% transmute(component, year, metric = "Exposure burden", value = exposure_burden),
  expanded_plot_df %>% transmute(component, year, metric = "Loss translation", value = loss_translation_index)
) %>%
  mutate(metric = factor(metric, levels = c("Exposure burden", "Loss translation")))

top_limits_global <- range(
  c(panel_b_df$value[panel_b_df$metric == "Exposure burden"], 0.8, 1.0, 1.2),
  na.rm = TRUE
)

panel_b_blocks <- lapply(plot_component_order, function(x) {
  make_component_block(panel_b_df, x, top_limits_global)
})

y_title_block <- ggplot() +
  annotate(
    "text",
    x = 0.60,
    y = 0.70,
    label = "Exposure burden index",
    angle = 90,
    size = 9 / .pt,
    color = "#222222"
  ) +
  annotate(
    "text",
    x = 0.60,
    y = 0.30,
    label = "Milk loss index",
    angle = 90,
    size = 9 / .pt,
    color = "#222222"
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  theme_void(base_size = 9) +
  theme(plot.background = element_rect(fill = "transparent", color = NA))

panel_b <- wrap_plots(c(list(y_title_block), panel_b_blocks), nrow = 1) +
  plot_layout(widths = c(0.18, rep(1, length(plot_component_order)))) +
  plot_annotation(caption = "Year") &
  theme(
    plot.background = element_rect(fill = "transparent", color = NA),
    plot.caption = element_text(size = 9, face = "plain", color = "#222222", hjust = 0.5, margin = margin(t = 2))
  )

combined <- panel_a / wrap_elements(full = panel_b) +
  plot_layout(heights = c(1.05, 1.95)) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(size = 9, face = "plain", color = "#222222"),
    plot.tag.position = c(0.005, 0.995),
    plot.background = element_rect(fill = "transparent", color = NA)
  )

out <- file.path(fig, "supp_point2_expanded_exposure_set_trajectory.svg")
ggsave(out, combined, width = 9.2, height = 6.8, bg = "transparent")
clean_svg(out)

out_wo <- file.path(fig, "supp_point2_expanded_exposure_set_trajectory_wo_legend.svg")
ggsave(
  out_wo,
  combined & theme(legend.position = "none"),
  width = 9.2,
  height = 6.8,
  bg = "transparent"
)
clean_svg(out_wo)
