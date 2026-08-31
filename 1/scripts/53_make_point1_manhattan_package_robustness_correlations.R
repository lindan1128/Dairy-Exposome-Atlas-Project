#!/usr/bin/env Rscript

# Package-based robustness comparisons against the Point 1 Manhattan betas.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(fixest)
})

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)][1])))
here <- normalizePath(file.path(script_dir, ".."))
tab <- file.path(here, "tables")
fig <- file.path(here, "figures")
dir.create(fig, recursive = TRUE, showWarnings = FALSE)

clean_svg <- function(path) {
  x <- readLines(path, warn = FALSE)
  x <- gsub("stroke: #FFFFFF; fill: #FFFFFF;", "stroke: none; fill: none;", x, fixed = TRUE)
  x <- gsub("fill: #FFFFFF;", "fill: none;", x, fixed = TRUE)
  writeLines(x, path)
}

endpoint_labels <- c(
  "total_same_50" = "Total production",
  "per_cow_50" = "Milk per cow"
)
endpoint_line_pal <- c(
  "Total production" = "#4C93AD",
  "Milk per cow" = "#CF625D"
)
endpoint_lty <- c(
  "Total production" = "solid",
  "Milk per cow" = "22"
)
model_order <- c(
  "fixest_state_trend",
  "unweighted_fe",
  "loso",
  "mgcv_gam",
  "geepack_gee_independence",
  "rexposome_residual_exwas"
)
model_labels <- c(
  "fixest_state_trend" = "A  fixest FE + state trend",
  "unweighted_fe" = "B  Unweighted FE",
  "loso" = "C  Leave-one-state-out",
  "mgcv_gam" = "D  mgcv GAM",
  "geepack_gee_independence" = "E  geepack GEE",
  "rexposome_residual_exwas" = "F  rexposome ExWAS"
)

p_stars <- function(p) {
  case_when(
    is.na(p) ~ "",
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ ""
  )
}

robustness_input <- read_csv(file.path(tab, "point1_manhattan_robustness_input.csv"), show_col_types = FALSE) %>%
  distinct(phenotype_scope, exposure, .keep_all = TRUE) %>%
  select(
    phenotype,
    phenotype_scope,
    phenotype_label,
    outcome_col,
    exposure,
    exposure_zh,
    domain,
    mechanistic_domain_en,
    source_class,
    construct,
    window,
    form,
    is_dairy_weighted_exposure,
    measurement_support_variable,
    main_beta = beta,
    main_p = plot_p,
    main_incr_r2 = incr_r2
  )

standardize <- function(x) {
  x <- as.numeric(x)
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 0) return(rep(0, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

transform_y <- function(x) {
  x <- as.numeric(x)
  if (min(x, na.rm = TRUE) > 0) log(x) else x
}

weighted_r2 <- function(y, yhat, w = NULL) {
  ok <- is.finite(y) & is.finite(yhat)
  if (!is.null(w)) ok <- ok & is.finite(w) & w > 0
  y <- y[ok]
  yhat <- yhat[ok]
  w <- if (is.null(w)) rep(1, length(y)) else w[ok]
  if (length(y) < 3) return(NA_real_)
  ybar <- sum(w * y) / sum(w)
  tss <- sum(w * (y - ybar)^2)
  rss <- sum(w * (y - yhat)^2)
  if (!is.finite(tss) || tss <= 0) return(NA_real_)
  1 - rss / tss
}

adjust_r2 <- function(r2, n, p) {
  if (!is.finite(r2) || !is.finite(n) || !is.finite(p) || n <= p + 1) return(NA_real_)
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

fit_unweighted_fe_one <- function(r, panel) {
  needed <- c("state_alpha", "year_month", r$outcome_col, r$exposure)
  if (!all(needed %in% names(panel))) {
    return(tibble(
      phenotype = r$phenotype, phenotype_scope = r$phenotype_scope,
      phenotype_label = r$phenotype_label, outcome_col = r$outcome_col,
      exposure = r$exposure, exposure_zh = r$exposure_zh, domain = r$domain,
      mechanistic_domain_en = r$mechanistic_domain_en, source_class = r$source_class,
      construct = r$construct, window = r$window, form = r$form,
      is_dairy_weighted_exposure = r$is_dairy_weighted_exposure,
      measurement_support_variable = r$measurement_support_variable,
      main_beta = r$main_beta, main_p = r$main_p, main_incr_r2 = r$main_incr_r2,
      native_signal_tier = NA_character_, model_id = "unweighted_fe",
      model_label = "Unweighted FE", status = "missing_columns", note = "",
      beta = NA_real_, se = NA_real_, p = NA_real_, r2 = NA_real_, adjusted_r2 = NA_real_,
      n = NA_integer_, n_states = NA_integer_, n_clusters = NA_integer_, r2_type = "unweighted_fixest_global_r2"
    ))
  }
  dd <- panel[, needed] %>%
    mutate(
      state_alpha = factor(as.character(state_alpha)),
      year_month = factor(as.character(year_month)),
      y_raw = as.numeric(.data[[r$outcome_col]]),
      x_raw = as.numeric(.data[[r$exposure]])
    ) %>%
    filter(is.finite(y_raw), is.finite(x_raw)) %>%
    mutate(y_z = standardize(transform_y(y_raw)), x_z = standardize(x_raw))
  base <- tibble(
    phenotype = r$phenotype, phenotype_scope = r$phenotype_scope,
    phenotype_label = r$phenotype_label, outcome_col = r$outcome_col,
    exposure = r$exposure, exposure_zh = r$exposure_zh, domain = r$domain,
    mechanistic_domain_en = r$mechanistic_domain_en, source_class = r$source_class,
    construct = r$construct, window = r$window, form = r$form,
    is_dairy_weighted_exposure = r$is_dairy_weighted_exposure,
    measurement_support_variable = r$measurement_support_variable,
    main_beta = r$main_beta, main_p = r$main_p, main_incr_r2 = r$main_incr_r2,
    native_signal_tier = NA_character_, model_id = "unweighted_fe",
    model_label = "Unweighted FE", status = "ok", note = "unweighted state + year-month FE refit",
    beta = NA_real_, se = NA_real_, p = NA_real_, r2 = NA_real_, adjusted_r2 = NA_real_,
    n = nrow(dd), n_states = n_distinct(dd$state_alpha), n_clusters = n_distinct(dd$state_alpha),
    r2_type = "unweighted_fixest_global_r2"
  )
  if (nrow(dd) < 60 || n_distinct(dd$state_alpha) < 3) {
    base$status <- "too_few"
    return(base)
  }
  fit <- tryCatch(
    feols(y_z ~ x_z | state_alpha + year_month, data = dd, cluster = ~state_alpha, warn = FALSE, notes = FALSE),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    base$status <- "failed"
    base$note <- conditionMessage(fit)
    return(base)
  }
  ct <- tryCatch(coeftable(fit), error = function(e) NULL)
  base$beta <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Estimate"] else NA_real_
  base$se <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Std. Error"] else NA_real_
  base$p <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Pr(>|t|)"] else NA_real_
  base$r2 <- as.numeric(tryCatch(fixest::r2(fit, type = "r2"), error = function(e) NA_real_))[1]
  base$adjusted_r2 <- as.numeric(tryCatch(fixest::r2(fit, type = "ar2"), error = function(e) NA_real_))[1]
  base$n <- nobs(fit)
  base
}

panel_for_unweighted <- read_csv(file.path(tab, "point1_manhattan_robustness_panel_for_r.csv"), show_col_types = FALSE)
unweighted_pairs <- bind_rows(lapply(seq_len(nrow(robustness_input)), function(i) fit_unweighted_fe_one(robustness_input[i, ], panel_for_unweighted))) %>%
  filter(
    phenotype_scope %in% names(endpoint_labels),
    status == "ok",
    is.finite(main_beta),
    is.finite(beta)
  )

package_pairs <- read_csv(file.path(tab, "point1_manhattan_package_robustness_beta_r2.csv"), show_col_types = FALSE) %>%
  filter(
    model_id %in% model_order,
    phenotype_scope %in% names(endpoint_labels),
    status == "ok",
    is.finite(main_beta),
    is.finite(beta)
  )

loso_pairs <- read_csv(file.path(tab, "point1_exwas_class_matched_loso_beta_robustness.csv"), show_col_types = FALSE) %>%
  inner_join(robustness_input, by = c("phenotype_scope", "exposure"), suffix = c("_loso", "")) %>%
  transmute(
    phenotype = phenotype,
    phenotype_scope = phenotype_scope,
    phenotype_label = phenotype_label,
    outcome_col = NA_character_,
    exposure = exposure,
    exposure_zh = coalesce(exposure_zh, exposure_zh_loso),
    domain = domain,
    mechanistic_domain_en = mechanistic_domain_en,
    source_class = source_class,
    construct = construct,
    window = window,
    form = form,
    is_dairy_weighted_exposure = is_dairy_weighted_exposure,
    measurement_support_variable = measurement_support_variable,
    main_beta = main_beta,
    main_p = main_p,
    main_incr_r2 = main_incr_r2,
    native_signal_tier = NA_character_,
    model_id = "loso",
    model_label = "Leave-one-state-out",
    status = "ok",
    note = "LOSO median beta from point1_exwas_class_matched_loso_beta_robustness.csv",
    beta = loso_beta_median,
    se = NA_real_,
    p = NA_real_,
    r2 = NA_real_,
    adjusted_r2 = NA_real_,
    n = loso_n,
    n_states = loso_n,
    n_clusters = loso_n,
    r2_type = NA_character_
  ) %>%
  filter(
    phenotype_scope %in% names(endpoint_labels),
    is.finite(main_beta),
    is.finite(beta)
  )

pairs <- bind_rows(package_pairs, unweighted_pairs, loso_pairs) %>%
  mutate(
    sensitivity_id = factor(model_id, levels = model_order),
    phenotype = factor(phenotype_scope, levels = names(endpoint_labels), labels = endpoint_labels[names(endpoint_labels)])
  )

flag_fit_outliers <- function(d) {
  if (nrow(d) < 5) {
    d$fit_outlier <- FALSE
    d$cooks_d <- NA_real_
    return(d)
  }
  fit <- lm(beta ~ main_beta, data = d)
  cd <- cooks.distance(fit)
  d$cooks_d <- as.numeric(cd)
  d$fit_outlier <- is.finite(d$cooks_d) & d$cooks_d > (4 / nrow(d))
  d
}

pairs <- pairs %>%
  group_by(model_id, phenotype) %>%
  group_modify(~ flag_fit_outliers(.x)) %>%
  ungroup()

fit_pairs <- pairs %>% filter(!fit_outlier)
plot_scale <- as.numeric(quantile(abs(c(fit_pairs$main_beta, fit_pairs$beta)), 0.95, na.rm = TRUE)) / 0.8
if (!is.finite(plot_scale) || plot_scale <= 0) plot_scale <- 1
pairs <- pairs %>%
  mutate(
    plot_main_beta = main_beta / plot_scale,
    plot_beta = beta / plot_scale,
    plot_range_excluded = abs(plot_main_beta) > 0.8 | abs(plot_beta) > 0.8
  )
plot_pairs <- pairs %>% filter(!plot_range_excluded)
fit_pairs <- pairs %>% filter(!fit_outlier, !plot_range_excluded)

write_csv(
  pairs %>%
    transmute(
      phenotype_scope,
      phenotype,
      domain,
      exposure,
      model_id,
      model_label,
      primary_beta = main_beta,
      sensitivity_beta = beta,
      model_r2 = r2,
      model_adjusted_r2 = adjusted_r2,
      n,
      n_states,
      status,
      cooks_d,
      fit_outlier,
      plot_range_excluded,
      plot_scale,
      plot_primary_beta = plot_main_beta,
      plot_sensitivity_beta = plot_beta
    ) %>%
    arrange(model_id, phenotype_scope, domain, exposure),
  file.path(tab, "point1_manhattan_package_robustness_correlation_pairs.csv")
)

stats <- fit_pairs %>%
  group_by(sensitivity_id, model_id, phenotype) %>%
  summarise(
    n = n(),
    n_outlier_removed = sum(pairs$model_id == first(model_id) & pairs$phenotype == first(phenotype) & pairs$fit_outlier),
    r = cor(main_beta, beta),
    r2 = r^2,
    p = cor.test(main_beta, beta)$p.value,
    .groups = "drop"
  ) %>%
  mutate(stars = p_stars(p), label = paste0("r=", sprintf("%.2f", r), stars))
write_csv(stats, file.path(tab, "point1_manhattan_package_robustness_correlation_statistics.csv"))

limit <- 1

base_theme <- theme_bw(base_size = 9, base_family = "Arial") +
  theme(
    text = element_text(size = 9, family = "Arial", face = "plain", color = "#222222"),
    plot.title = element_text(size = 9, family = "Arial", face = "plain", hjust = 0),
    axis.title = element_text(size = 9, family = "Arial", face = "plain"),
    axis.text = element_text(size = 9, family = "Arial", color = "#222222"),
    legend.title = element_text(size = 9, family = "Arial", face = "plain"),
    legend.text = element_text(size = 9, family = "Arial", face = "plain"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    axis.line = element_line(linewidth = 0.28, color = "#111111"),
    panel.border = element_rect(color = "#111111", fill = NA, linewidth = 0.28)
  )

make_panel <- function(model, show_y_title = FALSE, show_x_title = FALSE) {
  d <- plot_pairs %>% filter(model_id == model)
  ann <- stats %>%
    filter(model_id == model) %>%
    mutate(
      x = -0.95 * limit,
      y = c(0.93, 0.81) * limit
    )
  ggplot(d, aes(plot_main_beta, plot_beta, color = phenotype, linetype = phenotype)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#777777", linewidth = 0.28) +
    geom_hline(yintercept = 0, color = "#d0d0d0", linewidth = 0.20) +
    geom_vline(xintercept = 0, color = "#d0d0d0", linewidth = 0.20) +
    geom_point(size = 1.55, alpha = 0.74, stroke = 0) +
    geom_smooth(
      data = fit_pairs %>% filter(model_id == model),
      method = "lm",
      formula = y ~ x,
      se = FALSE,
      linewidth = 1.55,
      alpha = 1
    ) +
    geom_text(
      data = ann,
      aes(x = x, y = y, label = label, color = phenotype),
      inherit.aes = FALSE,
      hjust = 0,
      vjust = 1,
      size = 2.55,
      show.legend = FALSE
    ) +
    scale_color_manual(values = endpoint_line_pal, name = NULL, drop = FALSE) +
    scale_linetype_manual(values = endpoint_lty, name = NULL, drop = FALSE) +
    scale_x_continuous(breaks = c(-1, 0, 1)) +
    scale_y_continuous(breaks = c(-1, 0, 1)) +
    coord_fixed(xlim = c(-limit, limit), ylim = c(-limit, limit), clip = "on") +
    labs(
      title = model_labels[[model]],
      x = if (show_x_title) "Scaled primary standardized beta" else NULL,
      y = if (show_y_title) "Scaled robustness standardized beta" else NULL
    ) +
    base_theme +
    theme(
      legend.position = "bottom",
      legend.key.width = grid::unit(1.2, "lines"),
      panel.spacing = grid::unit(0.45, "lines"),
      plot.margin = margin(4, 4, 4, 4)
    )
}

plots <- list(
  make_panel("fixest_state_trend", TRUE, FALSE),
  make_panel("unweighted_fe", FALSE, FALSE),
  make_panel("loso", FALSE, FALSE),
  make_panel("mgcv_gam", TRUE, TRUE),
  make_panel("geepack_gee_independence", FALSE, TRUE),
  make_panel("rexposome_residual_exwas", FALSE, TRUE)
)

p_wo_theme <- theme(
  legend.position = "none",
  plot.title = element_blank(),
  axis.title = element_blank()
)

top <- (plots[[1]] | plots[[2]] | plots[[3]]) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
bottom <- (plots[[4]] | plots[[5]] | plots[[6]]) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

out_top <- file.path(fig, "main_point1_exwas_package_robustness_correlations_top.svg")
ggsave(out_top, top, width = 4.7, height = 2.35, units = "in", bg = "transparent")
clean_svg(out_top)

out_bottom <- file.path(fig, "main_point1_exwas_package_robustness_correlations_bottom.svg")
ggsave(out_bottom, bottom, width = 4.7, height = 2.35, units = "in", bg = "transparent")
clean_svg(out_bottom)

top_wo <- (
  (plots[[1]] + p_wo_theme + theme(plot.margin = margin(4, 4, 4, 4))) |
    (plots[[2]] + p_wo_theme + theme(plot.margin = margin(4, 4, 4, 4))) |
    (plots[[3]] + p_wo_theme + theme(plot.margin = margin(4, 4, 4, 4)))
) +
  plot_layout(guides = "collect")
bottom_wo <- (
  (plots[[4]] + p_wo_theme + theme(plot.margin = margin(4, 4, 4, 4))) |
    (plots[[5]] + p_wo_theme + theme(plot.margin = margin(4, 4, 4, 4))) |
    (plots[[6]] + p_wo_theme + theme(plot.margin = margin(4, 4, 4, 4)))
) +
  plot_layout(guides = "collect")

out_wo_top <- file.path(fig, "main_point1_exwas_package_robustness_correlations_wo_legend_top.svg")
ggsave(out_wo_top, top_wo, width = 4.7, height = 1.85, units = "in", bg = "transparent")
clean_svg(out_wo_top)

out_wo_bottom <- file.path(fig, "main_point1_exwas_package_robustness_correlations_wo_legend_bottom.svg")
ggsave(out_wo_bottom, bottom_wo, width = 4.7, height = 1.85, units = "in", bg = "transparent")
clean_svg(out_wo_bottom)

message("Wrote Point 1 package robustness correlation figures.")
