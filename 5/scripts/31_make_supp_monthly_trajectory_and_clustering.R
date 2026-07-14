#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(patchwork)
  library(reticulate)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- if (length(file_arg)) normalizePath(sub("^--file=", "", file_arg[1])) else normalizePath(".")
root <- normalizePath(file.path(dirname(script_path), "..", "..", "..", "..", ".."), mustWork = TRUE)
tab5 <- file.path(root, "analysis/statistics/5/tables")
fig5 <- file.path(root, "analysis/statistics/5/figures")

region_map <- c(
  "AZ" = "West", "CA" = "West", "CO" = "West", "ID" = "West", "NM" = "West", "OR" = "West", "UT" = "West", "WA" = "West",
  "IA" = "Midwest", "IL" = "Midwest", "IN" = "Midwest", "KS" = "Midwest", "MI" = "Midwest", "MN" = "Midwest", "MO" = "Midwest", "OH" = "Midwest", "SD" = "Midwest", "WI" = "Midwest",
  "NY" = "Northeast", "PA" = "Northeast", "VT" = "Northeast",
  "FL" = "South", "GA" = "South", "KY" = "South", "TX" = "South", "VA" = "South"
)
region_order <- c("South", "West", "Midwest", "Northeast")
region_cols <- c("South" = "#F06F26", "West" = "#60BFA4", "Midwest" = "#1487CA", "Northeast" = "#D77291")
cluster_subdomains <- c(
  "Cold: nighttime / no-thaw",
  "Feed market",
  "Heat: humid (wet-bulb)",
  "Milk price, dairy market and milk-feed price ratio",
  "Storm / disaster events"
)

theme_supp <- theme_bw(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(size = 9, color = "black", family = "Arial"),
    axis.text = element_text(size = 9, color = "#222222", family = "Arial"),
    axis.title = element_text(size = 9, color = "#222222", family = "Arial"),
    legend.title = element_blank(),
    legend.text = element_text(size = 9, color = "black", family = "Arial"),
    legend.key = element_blank(),
    legend.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, color = "#111111", linewidth = 0.28),
    axis.ticks = element_line(linewidth = 0.22, color = "#111111"),
    axis.ticks.length = unit(4.2, "pt"),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA)
  )

weighted_mean <- function(x, w) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  if (!any(ok)) return(NA_real_)
  weighted.mean(x[ok], w[ok])
}

roll12 <- function(x) {
  out <- rep(NA_real_, length(x))
  for (i in seq_along(x)) {
    lo <- max(1, i - 11)
    out[i] <- mean(x[lo:i], na.rm = TRUE)
  }
  out
}

boot_weighted_ci <- function(x, w, n_boot = 2000) {
  ok <- is.finite(x) & is.finite(w) & w > 0
  x <- x[ok]
  w <- w[ok]
  n <- length(x)
  if (n == 0) return(c(NA_real_, NA_real_))
  if (n == 1) return(c(x[1], x[1]))
  idx <- matrix(sample.int(n, size = n * n_boot, replace = TRUE), nrow = n)
  vals <- colSums(matrix(x[idx], nrow = n) * matrix(w[idx], nrow = n)) / colSums(matrix(w[idx], nrow = n))
  as.numeric(quantile(vals, c(0.025, 0.975), na.rm = TRUE))
}

make_monthly_panel <- function() {
  set.seed(20260709)
  pred <- read_csv(file.path(tab5, "point5_exposome_milk_loss_risk_predictions.csv"), show_col_types = FALSE) %>%
    mutate(date_x = year + (month - 1) / 12) %>%
    filter(year >= 2014, year <= 2025)

  monthly <- pred %>%
    group_by(year, month, date_x) %>%
    summarise(
      observed_loss_pct = weighted_mean(next_loss_pct, milk_cows_head),
      baseline_predicted_loss_pct = weighted_mean(baseline_predicted_loss_pct, milk_cows_head),
      exposome_predicted_loss_pct = weighted_mean(exposome_predicted_loss_pct, milk_cows_head),
      .groups = "drop"
    ) %>%
    arrange(date_x) %>%
    mutate(
      observed_loss_pct = roll12(observed_loss_pct),
      baseline_predicted_loss_pct = roll12(baseline_predicted_loss_pct),
      exposome_predicted_loss_pct = roll12(exposome_predicted_loss_pct)
    )

  ci_df <- pred %>%
    group_by(year, month, date_x) %>%
    summarise(
      baseline_low = boot_weighted_ci(baseline_predicted_loss_pct, milk_cows_head)[1],
      baseline_high = boot_weighted_ci(baseline_predicted_loss_pct, milk_cows_head)[2],
      exposome_low = boot_weighted_ci(exposome_predicted_loss_pct, milk_cows_head)[1],
      exposome_high = boot_weighted_ci(exposome_predicted_loss_pct, milk_cows_head)[2],
      .groups = "drop"
    ) %>%
    arrange(date_x) %>%
    mutate(
      baseline_low = roll12(baseline_low),
      baseline_high = roll12(baseline_high),
      exposome_low = roll12(exposome_low),
      exposome_high = roll12(exposome_high)
    ) %>%
    pivot_longer(
      cols = c(baseline_low, baseline_high, exposome_low, exposome_high),
      names_to = c("model", ".value"),
      names_pattern = "(baseline|ridge)_(low|high)"
    ) %>%
    transmute(
      year, month, date_x,
      series = if_else(model == "baseline", "Prediction based on phenotype history", "Prediction based on phenotype history and exposome"),
      ci_low = low,
      ci_high = high
    ) %>%
    mutate(series = factor(series, levels = c("Prediction based on phenotype history", "Prediction based on phenotype history and exposome")))

  plot_df <- monthly %>%
    filter(year >= 2015, year <= 2025) %>%
    pivot_longer(
      cols = c(observed_loss_pct, baseline_predicted_loss_pct, exposome_predicted_loss_pct),
      names_to = "series",
      values_to = "loss_pct"
    ) %>%
    mutate(
      series = factor(
        series,
        levels = c("observed_loss_pct", "baseline_predicted_loss_pct", "exposome_predicted_loss_pct"),
        labels = c("Observed", "Prediction based on phenotype history", "Prediction based on phenotype history and exposome")
      ),
      loss_pct = tanh(loss_pct)
    )

  ci_df <- ci_df %>%
    filter(year >= 2015, year <= 2025) %>%
    mutate(ci_low = tanh(ci_low), ci_high = tanh(ci_high))

  series_cols <- c(
    "Observed" = "#8a8a8a",
    "Prediction based on phenotype history" = "#1487CA",
    "Prediction based on phenotype history and exposome" = "#D77291"
  )
  series_lwd <- c(
    "Observed" = 0.50,
    "Prediction based on phenotype history" = 0.50,
    "Prediction based on phenotype history and exposome" = 0.80
  )
  ribbon_cols <- c(
    "Prediction based on phenotype history" = "#1487CA",
    "Prediction based on phenotype history and exposome" = "#D77291"
  )

  ggplot(plot_df, aes(date_x, loss_pct, color = series, linewidth = series)) +
    geom_ribbon(
      data = ci_df,
      aes(x = date_x, ymin = ci_low, ymax = ci_high, fill = series, group = series),
      inherit.aes = FALSE,
      alpha = 0.72,
      color = NA
    ) +
    geom_line(lineend = "butt") +
    scale_color_manual(values = series_cols, drop = FALSE) +
    scale_fill_manual(values = ribbon_cols, guide = "none") +
    scale_linewidth_manual(values = series_lwd, drop = FALSE) +
    scale_x_continuous(
      breaks = 2015:2025,
      labels = function(x) ifelse(x %in% c(2015, 2020, 2025), as.character(x), ""),
      limits = c(2015, 2025 + 11 / 12),
      expand = expansion(mult = c(0.02, 0.02))
    ) +
    scale_y_continuous(
      breaks = c(-1.0, 0.0, 1.0),
      labels = sprintf("%.1f", c(-1.0, 0.0, 1.0)),
      limits = c(-1, 1),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(x = "Year", y = "Next-month milk-loss risk (%)") +
    guides(
      color = guide_legend(nrow = 1, byrow = TRUE, override.aes = list(linewidth = c(0.60, 0.60, 0.95))),
      linewidth = "none"
    ) +
    theme_supp +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      axis.title.x = element_text(margin = margin(t = 6)),
      legend.position = "bottom",
      legend.key.width = unit(18, "pt"),
      legend.key.height = unit(7, "pt"),
      legend.margin = margin(t = 0),
      legend.box.margin = margin(t = -4, b = 0),
      plot.margin = margin(5, 6, 2, 6)
    )
}

make_cluster_panel <- function() {
  hierarchy <- import("scipy.cluster.hierarchy")

  panel <- read_csv(file.path(tab5, "point5_forecast_state_month_feature_panel.csv"), show_col_types = FALSE)
  meta <- read_csv(file.path(tab5, "point5_forecast_feature_dictionary.csv"), show_col_types = FALSE)
  features <- meta %>%
    filter(feature_group == "exposure_node", mechanistic_domain_en %in% cluster_subdomains, feature %in% names(panel)) %>%
    pull(feature)

  annual <- panel %>%
    filter(state_alpha %in% names(region_map), year >= 2000, year <= 2014) %>%
    group_by(state_alpha, year) %>%
    summarise(across(all_of(features), ~mean(.x, na.rm = TRUE)), .groups = "drop")

  states <- sort(unique(annual$state_alpha))
  x_list <- lapply(states, function(st) {
    dat <- annual %>% filter(state_alpha == st) %>% arrange(year)
    dat <- right_join(tibble(year = 2000:2014), dat, by = "year") %>% arrange(year)
    mat <- as.matrix(dat[, features, drop = FALSE])
    apply(mat, 2, function(v) {
      if (all(!is.finite(v))) return(rep(NA_real_, length(v)))
      idx <- which(is.finite(v))
      if (length(idx) == 1) return(rep(v[idx], length(v)))
      approx(x = idx, y = v[idx], xout = seq_along(v), rule = 2)$y
    }) %>% as.vector()
  })
  x <- do.call(rbind, x_list)
  keep <- colSums(is.finite(x)) > 0
  x <- x[, keep, drop = FALSE]
  x <- scale(x)
  x[!is.finite(x)] <- 0
  rownames(x) <- states

  z <- py_to_r(hierarchy$linkage(x, method = "ward", metric = "euclidean"))
  cl <- as.integer(py_to_r(hierarchy$fcluster(z, t = 4L, criterion = "maxclust")))
  true <- match(region_map[states], region_order)

  best_accuracy <- function(true, pred) {
    tab <- table(true, pred)
    perms <- as.matrix(expand.grid(rep(list(seq_len(ncol(tab))), nrow(tab))))
    perms <- perms[apply(perms, 1, function(z) length(unique(z)) == length(z)), , drop = FALSE]
    best <- max(apply(perms, 1, function(p) sum(tab[cbind(seq_len(nrow(tab)), p)])))
    best / sum(tab)
  }
  comb2 <- function(n) n * (n - 1) / 2
  ari_score <- function(true, pred) {
    tab <- table(true, pred)
    sum_c <- sum(comb2(as.vector(tab)))
    sum_t <- sum(comb2(rowSums(tab)))
    sum_p <- sum(comb2(colSums(tab)))
    total <- comb2(length(true))
    expected <- sum_t * sum_p / total
    max_index <- (sum_t + sum_p) / 2
    if (max_index == expected) return(0)
    (sum_c - expected) / (max_index - expected)
  }
  entropy <- function(counts) {
    p <- counts[counts > 0] / sum(counts)
    -sum(p * log(p))
  }
  nmi_score <- function(true, pred) {
    tab <- table(true, pred)
    n <- sum(tab)
    pi <- rowSums(tab)
    pj <- colSums(tab)
    mi <- 0
    for (i in seq_len(nrow(tab))) {
      for (j in seq_len(ncol(tab))) {
        if (tab[i, j] > 0) mi <- mi + tab[i, j] / n * log((tab[i, j] * n) / (pi[i] * pj[j]))
      }
    }
    mi / sqrt(entropy(pi) * entropy(pj))
  }
  acc <- best_accuracy(true, cl)
  ari <- ari_score(true, cl)
  nmi <- nmi_score(true, cl)

  n <- length(states)
  leaf_order <- as.integer(py_to_r(hierarchy$leaves_list(z))) + 1L
  ordered_states <- states[leaf_order]
  x_pos <- setNames(seq_along(leaf_order), leaf_order - 1L)
  node_x <- numeric(nrow(z))
  node_h <- z[, 3]
  segs <- list()
  for (i in seq_len(nrow(z))) {
    kids <- as.integer(z[i, 1:2])
    child <- lapply(kids, function(k) {
      if (k < n) {
        list(x = x_pos[as.character(k)], h = 0)
      } else {
        idx <- k - n + 1L
        list(x = node_x[idx], h = node_h[idx])
      }
    })
    node_x[i] <- mean(c(child[[1]]$x, child[[2]]$x))
    h <- z[i, 3]
    segs[[length(segs) + 1]] <- tibble(x = child[[1]]$x, xend = child[[1]]$x, y = child[[1]]$h, yend = h)
    segs[[length(segs) + 1]] <- tibble(x = child[[2]]$x, xend = child[[2]]$x, y = child[[2]]$h, yend = h)
    segs[[length(segs) + 1]] <- tibble(x = child[[1]]$x, xend = child[[2]]$x, y = h, yend = h)
  }
  seg_df <- bind_rows(segs)
  label_df <- tibble(
    state_alpha = ordered_states,
    x = seq_along(ordered_states),
    region = factor(region_map[ordered_states], levels = region_order)
  )
  strip_df <- label_df %>% mutate(ymin = -16, ymax = -11, xmin = x - 0.42, xmax = x + 0.42)

  ggplot() +
    geom_segment(data = seg_df, aes(x = x, xend = xend, y = y, yend = yend), color = "#555555", linewidth = 0.45) +
    geom_rect(data = strip_df, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = region), color = NA) +
    geom_text(data = label_df, aes(x = x, y = -20, label = state_alpha, color = region), angle = 90, hjust = 1, vjust = 0.5, size = 9 / .pt, family = "Arial") +
    annotate("text", x = 1, y = max(z[, 3]) * 1.15, label = "Pre-2015 annual exposure trajectory clustering", hjust = 0, size = 9 / .pt, family = "Arial") +
    annotate("text", x = 1, y = max(z[, 3]) * 1.06, label = sprintf("Euclidean + Ward linkage; best-match accuracy = %.2f; ARI = %.2f; NMI = %.2f", acc, ari, nmi), hjust = 0, size = 9 / .pt, family = "Arial") +
    scale_fill_manual(values = region_cols, breaks = region_order, drop = FALSE) +
    scale_color_manual(values = region_cols, breaks = region_order, guide = "none") +
    scale_x_continuous(limits = c(0.25, n + 0.75), expand = expansion(mult = c(0, 0))) +
    scale_y_continuous(limits = c(-34, max(z[, 3]) * 1.22), expand = expansion(mult = c(0, 0))) +
    labs(x = NULL, y = "Euclidean distance") +
    theme_supp +
    theme(
      panel.border = element_blank(),
      axis.line.x = element_line(linewidth = 0.28, color = "#111111"),
      axis.line.y = element_line(linewidth = 0.28, color = "#111111"),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      legend.position = "bottom",
      legend.key.width = unit(10, "pt"),
      legend.key.height = unit(7, "pt"),
      legend.box.margin = margin(t = -2),
      plot.margin = margin(6, 6, 2, 6)
    ) +
    guides(fill = guide_legend(nrow = 1, byrow = TRUE))
}

p_a <- make_monthly_panel()
p_b <- make_cluster_panel()

combined <- p_a / p_b +
  plot_layout(heights = c(1.0, 1.34)) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(size = 9, face = "bold", family = "Arial", color = "black"),
    plot.tag.position = c(0, 1)
  )

out <- file.path(fig5, "supp_point5_monthly_risk_trajectory_and_exposure_clustering")
ggsave(paste0(out, ".svg"), combined, width = 7.2, height = 5.7)
ggsave(paste0(out, ".png"), combined, width = 7.2, height = 5.7, dpi = 450, bg = "transparent")
message("Wrote combined R supplementary figure SVG/PNG")
