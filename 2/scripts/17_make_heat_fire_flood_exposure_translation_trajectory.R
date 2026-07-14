#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
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

component_order <- c("Humid heat", "Dry heat", "Night heat", "Flood", "Fire")
plot_component_order <- c("Humid heat", "Dry heat", "Flood", "Fire")
version_order <- c("chord_signal")
version_labels <- c("chord_signal" = "Chord-signal")

heat_subform <- read_csv(
  file.path(tab, "point2_heat_subform_exposure_translation_trajectory.csv"),
  show_col_types = FALSE
) %>%
  filter(subform %in% c("Humid threshold heat", "Dry heat", "Night heat / no relief")) %>%
  mutate(
    version = "chord_signal",
    component = recode(
      subform,
      "Humid threshold heat" = "Humid heat",
      "Dry heat" = "Dry heat",
      "Night heat / no relief" = "Night heat"
    )
  ) %>%
  transmute(
    version, component, year,
    exposure_burden = exposure_burden_index_2000,
    loss_translation = loss_translation
  )

severe <- read_csv(
  file.path(tab, "point2_severe_type_exposure_translation_trajectory.csv"),
  show_col_types = FALSE
) %>%
  filter(disaster_type %in% c("Flood events", "Fire events")) %>%
  filter(pool == "chord-signal natural events") %>%
  mutate(
    version = "chord_signal",
    component = recode(
      disaster_type,
      "Flood events" = "Flood",
      "Fire events" = "Fire"
    )
  ) %>%
  filter(version %in% version_order) %>%
  transmute(
    version, component, year,
    exposure_burden = exposure_burden_index_2000,
    loss_translation = loss_translation
  )

traj <- bind_rows(heat_subform, severe) %>%
  mutate(
    version = factor(version, levels = version_order),
    version_label = factor(version_labels[as.character(version)], levels = version_labels[version_order]),
    component = factor(component, levels = component_order)
  ) %>%
  group_by(version, component) %>%
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
  ungroup() %>%
  arrange(version, component, year)

write_csv(traj, file.path(tab, "point2_heat_fire_flood_exposure_translation_trajectory.csv"))

trend <- traj %>%
  group_by(version, version_label, component) %>%
  summarize(
    n_years = sum(is.finite(exposure_burden) | is.finite(loss_translation_index)),
    exposure_start = exposure_burden[year == min(year, na.rm = TRUE)][1],
    exposure_end = exposure_burden[year == max(year, na.rm = TRUE)][1],
    exposure_pct_change = ifelse(
      is.finite(exposure_start) & exposure_start != 0 & is.finite(exposure_end),
      100 * (exposure_end / exposure_start - 1),
      NA_real_
    ),
    translation_start = loss_translation_index[year == min(year, na.rm = TRUE)][1],
    translation_end = loss_translation_index[year == max(year, na.rm = TRUE)][1],
    translation_pct_change = ifelse(
      is.finite(translation_start) & translation_start != 0 & is.finite(translation_end),
      100 * (translation_end / translation_start - 1),
      NA_real_
    ),
    .groups = "drop"
  )
write_csv(trend, file.path(tab, "point2_heat_fire_flood_exposure_translation_trend_summary.csv"))

three_breaks <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(numeric(0))
  }
  rng <- range(x)
  if (!is.finite(rng[1]) || !is.finite(rng[2])) {
    return(numeric(0))
  }
  if (rng[1] == rng[2]) {
    return(rng[1])
  }
  seq(rng[1], rng[2], length.out = 3)
}

three_nice_half_breaks <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) {
    return(c(0, 0.5, 1.0))
  }
  rng <- range(x)
  if (!is.finite(rng[1]) || !is.finite(rng[2])) {
    return(c(0, 0.5, 1.0))
  }
  if (rng[1] == rng[2]) {
    mid <- round(rng[1] * 2) / 2
    return(mid + c(-0.5, 0, 0.5))
  }
  raw_step <- (rng[2] - rng[1]) / 2
  steps <- c(0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100)
  for (step in steps) {
    if (step < raw_step) {
      next
    }
    high <- ceiling(rng[2] / step) * step
    low <- high - 2 * step
    if (low <= rng[1]) {
      return(c(low, low + step, high))
    }
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

p_stars <- function(p) {
  case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ ""
  )
}

trend_label_row <- function(d, panel_name, y_pos) {
  z <- d %>% filter(metric == panel_name, is.finite(year), is.finite(value))
  if (nrow(z) < 3 || sd(z$year, na.rm = TRUE) == 0) {
    return(tibble(panel = panel_name, label = NA_character_, y = y_pos))
  }
  fit <- lm(value ~ year, data = z)
  p <- coef(summary(fit))["year", "Pr(>|t|)"]
  tibble(
    panel = panel_name,
    label = paste0(ifelse(p_stars(p) == "", "", paste0(p_stars(p), " ")), "R²=", sprintf("%.2f", summary(fit)$r.squared)),
    y = y_pos
  )
}

make_component_block <- function(d, component_name, top_limits_global, show_stats = FALSE) {
  exposure_color <- "#4d908e"
  translation_color <- "#274c77"
  top <- d %>% filter(component == component_name, metric == "Exposure burden")
  bottom <- d %>% filter(component == component_name, metric == "Loss translation")

  top_ticks <- c(0.8, 1.0, 1.2)
  bottom_ticks <- three_nice_half_breaks(bottom$value)
  top_limits <- top_limits_global
  bottom_limits <- range(c(bottom$value, bottom_ticks), na.rm = TRUE)
  if (!all(is.finite(top_limits))) {
    top_limits <- c(0.8, 1.2)
  }
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
    top %>%
      transmute(year, value, panel = "top",
                y = panel_scale(value, top_limits, top_min, top_max)),
    bottom %>%
      transmute(year, value, panel = "bottom",
                y = panel_scale(value, bottom_limits, bottom_min, bottom_max))
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

  stat_labels <- bind_rows(
    trend_label_row(d %>% filter(component == component_name), "Exposure burden", top_max),
    trend_label_row(d %>% filter(component == component_name), "Loss translation", bottom_max)
  ) %>%
    filter(is.finite(y), !is.na(label)) %>%
    mutate(
      x = x_max - 0.6,
      y = ifelse(panel == "Exposure burden", top_max - 0.015, bottom_max - 0.015)
    )

  y_axis <- bind_rows(
    tibble(side = "left", value = top_ticks,
           y = panel_scale(top_ticks, top_limits, top_min, top_max),
           label = label_number(accuracy = 0.1)(top_ticks)),
    tibble(side = "right", value = bottom_ticks,
           y = panel_scale(bottom_ticks, bottom_limits, bottom_min, bottom_max),
           label = label_number(accuracy = 0.1)(bottom_ticks))
  )

  ggplot() +
    geom_vline(xintercept = x_ticks, color = "#e8e8e8", linewidth = 0.18) +
    geom_hline(yintercept = y_axis$y, color = "#e8e8e8", linewidth = 0.18) +
    geom_hline(yintercept = split_y, color = "#e8e8e8", linewidth = 0.22) +
    geom_segment(aes(x = x_min, xend = x_max, y = top_baseline_y, yend = top_baseline_y),
                 color = "#9e9e9e", linewidth = 1, linetype = "dashed") +
    {
      geom_segment(aes(x = x_min, xend = x_max, y = bottom_baseline_y, yend = bottom_baseline_y),
                   color = "#9e9e9e", linewidth = 1, linetype = "dashed")
    } +
    geom_ribbon(data = smooth_df, aes(year, ymin = ymin, ymax = ymax, fill = panel),
                alpha = 0.22, color = NA, na.rm = TRUE) +
    geom_point(data = line_df, aes(year, y, fill = panel),
               shape = 21, color = "#111111", size = 2.1, stroke = 0.14, na.rm = TRUE) +
    geom_line(data = smooth_df, aes(year, y, group = panel, color = panel),
              linewidth = 0.65, na.rm = TRUE) +
    {
      if (show_stats) {
        geom_text(
          data = stat_labels,
          aes(x = x, y = y, label = label),
          inherit.aes = FALSE,
          hjust = 1,
          vjust = 1,
          size = 2.6,
          color = "#111111"
        )
      }
    } +
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
    coord_cartesian(xlim = c(x_min - 1.0, x_max + 1.0), ylim = c(-0.12, 1), clip = "off") +
    theme_void(base_size = 9) +
    theme(
      text = element_text(size = 9, face = "plain", color = "#222222"),
      plot.background = element_rect(fill = "transparent", color = NA),
      plot.margin = margin(t = 1, r = 12, b = 9, l = 14)
    )
}

make_plot <- function(version_name, outfile_prefix) {
  d <- traj %>%
    filter(version == version_name) %>%
    select(version, component, year, exposure_burden, loss_translation_index) %>%
    filter(component %in% plot_component_order) %>%
    tidyr::complete(component = factor(plot_component_order, levels = plot_component_order), year = 2000:2025) %>%
    mutate(component = factor(component, levels = plot_component_order))

  plot_df <- bind_rows(
    d %>%
      transmute(component, year, metric = "Exposure burden", value = exposure_burden),
    d %>%
      transmute(component, year, metric = "Loss translation", value = loss_translation_index)
  ) %>%
    mutate(metric = factor(metric, levels = c("Exposure burden", "Loss translation")))

  top_limits_global <- range(
    c(plot_df$value[plot_df$metric == "Exposure burden"], 0.8, 1.0, 1.2),
    na.rm = TRUE
  )
  if (!all(is.finite(top_limits_global))) {
    top_limits_global <- c(0.8, 1.2)
  }

  blocks <- lapply(plot_component_order, function(x) {
    make_component_block(plot_df, x, top_limits_global, show_stats = version_name == "chord_signal")
  })
  p <- wrap_plots(blocks, ncol = length(plot_component_order)) +
    plot_layout(widths = rep(1, length(plot_component_order))) &
    theme(plot.background = element_rect(fill = "transparent", color = NA))

  out <- file.path(fig, paste0(outfile_prefix, ".svg"))
  ggsave(out, p, width = 8.5, height = 3.7, bg = "transparent")
  clean_svg(out)

  out2 <- file.path(fig, paste0(outfile_prefix, "_wo_legend.svg"))
  ggsave(out2, p + theme(legend.position = "none"), width = 8.5, height = 3.7, bg = "transparent")
  clean_svg(out2)
}

make_plot("chord_signal", "main_point2_heat_flood_fire_chord_signal_exposure_translation_trajectory")
