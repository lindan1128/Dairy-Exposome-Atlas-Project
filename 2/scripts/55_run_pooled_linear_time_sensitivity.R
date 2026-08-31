#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  .libPaths(c("/private/tmp/Rlib_point2", .libPaths()))
  library(data.table)
  library(fixest)
})

cmd <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", cmd[grep("--file=", cmd)][1])
point <- normalizePath(file.path(dirname(script_path), ".."))
tab <- file.path(point, "tables")

panel_file <- file.path(tab, "point2_expanded_sensitivity_panel_for_r.csv")
selection_file <- file.path(tab, "point2_common_sense_expanded_kept_variables.csv")
out_long <- file.path(tab, "point2_four_regression_variant_yearly_sensitivity_long.csv")
out_by_year <- file.path(tab, "point2_four_regression_variant_yearly_sensitivity_by_year.csv")
out_n <- file.path(tab, "point2_four_regression_variant_yearly_sensitivity_n_exposures.csv")
out_fit_stats <- file.path(tab, "point2_four_regression_variant_yearly_fit_stats.csv")

variant <- "fixest_pooled_linear_time"
variant_label <- "FE + x×year"
domain_order <- c("Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand")
lb_to_kg <- 0.45359237
years <- 2000:2025

z <- function(x) {
  x <- as.numeric(x)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 1e-12) return(rep(NA_real_, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

weighted_r2 <- function(y, yhat, w) {
  ok <- is.finite(y) & is.finite(yhat) & is.finite(w) & w > 0
  if (sum(ok) < 3) return(NA_real_)
  y <- y[ok]; yhat <- yhat[ok]; w <- w[ok]
  ybar <- stats::weighted.mean(y, w)
  sst <- sum(w * (y - ybar)^2)
  if (!is.finite(sst) || sst <= 1e-12) return(NA_real_)
  1 - sum(w * (y - yhat)^2) / sst
}

adj_r2 <- function(r2, n, p) {
  if (!is.finite(r2) || !is.finite(n) || !is.finite(p) || n <= p + 1) return(NA_real_)
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

make_work_data <- function(panel, exposure) {
  d <- panel[, .(
    state_alpha, year, month, milk_cows_head, log_per_cow,
    x_raw = get(exposure)
  )]
  d <- d[is.finite(log_per_cow) & is.finite(x_raw) & milk_cows_head > 0]
  d[, x_z := z(x_raw)]
  d <- d[is.finite(x_z)]
  d[, state_alpha := droplevels(factor(state_alpha))]
  d[, year_f := droplevels(factor(year, levels = years))]
  d[, month_f := droplevels(factor(month))]
  d[, year_num := as.numeric(year)]
  d[, year_scaled := as.numeric(scale(year_num))]
  d
}

fit_pooled <- function(d) {
  d <- copy(d)
  d[, x_year_scaled := x_z * year_scaled]
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6 || uniqueN(d$x_z) <= 1) return(NULL)
  fml <- log_per_cow ~ x_z + x_year_scaled | state_alpha + month_f + year_f
  fit <- tryCatch({
    tmp <- NULL
    capture.output(tmp <- feols(fml, data = d, weights = ~milk_cows_head, cluster = ~state_alpha, warn = FALSE))
    tmp
  }, error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  co <- coef(fit)
  b0 <- if ("x_z" %in% names(co)) unname(co[["x_z"]]) else NA_real_
  b1 <- if ("x_year_scaled" %in% names(co)) unname(co[["x_year_scaled"]]) else 0
  year_center <- mean(d$year_num, na.rm = TRUE)
  year_sd <- stats::sd(d$year_num, na.rm = TRUE)
  if (!is.finite(year_sd) || year_sd <= 1e-12) return(NULL)
  year_scaled_grid <- (years - year_center) / year_sd
  beta <- b0 + b1 * year_scaled_grid
  beta_obs <- b0 + b1 * d$year_scaled
  model_df <- tryCatch(as.numeric(fit$nparams), error = function(e) length(coef(fit)))
  if (length(model_df) != 1 || !is.finite(model_df)) model_df <- length(coef(fit))
  list(
    beta = beta,
    fitted = as.numeric(fitted(fit)),
    exposure_contribution = as.numeric(beta_obs * d$x_z),
    model_df = model_df,
    exposure_df = 2,
    global_r2 = tryCatch(as.numeric(fitstat(fit, "r2")[[1]]), error = function(e) NA_real_),
    global_adj_r2 = tryCatch(as.numeric(fitstat(fit, "ar2")[[1]]), error = function(e) NA_real_),
    r2_type = "fixest_pooled_linear_time_interaction_model_r2_and_yearly_weighted_prediction_r2"
  )
}

yearly_fit_stats <- function(d, fit_result, class_label, exposure) {
  rbindlist(lapply(years, function(yy) {
    idx <- d$year == yy & is.finite(d$log_per_cow) & is.finite(fit_result$fitted) &
      is.finite(d$milk_cows_head) & d$milk_cows_head > 0
    n_obs <- sum(idx)
    r2 <- weighted_r2(d$log_per_cow[idx], fit_result$fitted[idx], d$milk_cows_head[idx])
    y <- d$log_per_cow[idx]
    w <- d$milk_cows_head[idx]
    ybar <- if (sum(idx) > 0) stats::weighted.mean(y, w) else NA_real_
    sst <- if (sum(idx) > 0) sum(w * (y - ybar)^2) else NA_real_
    full_sse <- sum(d$milk_cows_head[idx] * (d$log_per_cow[idx] - fit_result$fitted[idx])^2)
    reduced_fitted <- fit_result$fitted[idx] - fit_result$exposure_contribution[idx]
    reduced_sse <- sum(d$milk_cows_head[idx] * (d$log_per_cow[idx] - reduced_fitted)^2)
    reduced_r2 <- if (is.finite(sst) && sst > 1e-12) 1 - reduced_sse / sst else NA_real_
    full_adj_r2 <- adj_r2(r2, n_obs, fit_result$model_df)
    reduced_adj_r2 <- adj_r2(reduced_r2, n_obs, fit_result$model_df - fit_result$exposure_df)
    partial_r2 <- if (is.finite(reduced_sse) && reduced_sse > 1e-12) max(0, (reduced_sse - full_sse) / reduced_sse) else NA_real_
    partial_adj_r2 <- if (
      is.finite(full_sse) && is.finite(reduced_sse) && reduced_sse > 1e-12 &&
      is.finite(fit_result$model_df) && n_obs > fit_result$model_df + 1
    ) {
      1 - (full_sse / (n_obs - fit_result$model_df)) /
        (reduced_sse / (n_obs - fit_result$model_df + fit_result$exposure_df))
    } else {
      NA_real_
    }
    incremental_r2 <- if (is.finite(r2) && is.finite(reduced_r2)) max(0, r2 - reduced_r2) else NA_real_
    adjusted_incremental_r2 <- if (is.finite(full_adj_r2) && is.finite(reduced_adj_r2)) full_adj_r2 - reduced_adj_r2 else NA_real_
    data.table(
      variant = variant,
      variant_label = variant_label,
      class_label = class_label,
      exposure = exposure,
      year = yy,
      r2_year = r2,
      adj_r2_year = full_adj_r2,
      reduced_r2_year = reduced_r2,
      reduced_adj_r2_year = reduced_adj_r2,
      incremental_r2_year = incremental_r2,
      adjusted_incremental_r2_year = adjusted_incremental_r2,
      partial_r2_year = partial_r2,
      partial_adj_r2_year = partial_adj_r2,
      n_obs_year = n_obs,
      model_df = fit_result$model_df,
      global_r2 = fit_result$global_r2,
      global_adj_r2 = fit_result$global_adj_r2,
      r2_type = fit_result$r2_type,
      status = ifelse(is.finite(fit_result$beta[match(yy, years)]), "ok", "failed")
    )
  }), fill = TRUE)
}

selected <- fread(selection_file)
if (!"class_label" %in% names(selected) && "domain_label" %in% names(selected)) selected[, class_label := domain_label]
if (!"domain_label" %in% names(selected) && "class_label" %in% names(selected)) selected[, domain_label := class_label]
selected <- selected[startsWith(expanded_selection_status, "kept_expanded")]
selected <- selected[class_label %in% domain_order]

panel <- fread(panel_file)
panel <- panel[year %between% c(2000, 2025)]
if (!"milk_per_cow_kg" %in% names(panel)) {
  panel[, milk_per_cow_kg := fifelse(milk_production_lb > 0 & milk_cows_head > 0, milk_production_lb * lb_to_kg / milk_cows_head, NA_real_)]
}
panel[, log_per_cow := log(fifelse(milk_per_cow_kg > 0, milk_per_cow_kg, NA_real_))]
setorder(panel, state_alpha, year, month)

new_rows <- list()
new_stats <- list()
for (ii in seq_len(nrow(selected))) {
  exposure <- selected$exposure[ii]
  class_label <- selected$class_label[ii]
  if (!exposure %in% names(panel)) next
  message(sprintf("[%s/%s] pooled x×year %s", ii, nrow(selected), exposure))
  d <- make_work_data(panel, exposure)
  fit_result <- fit_pooled(d)
  if (is.null(fit_result)) next
  new_rows[[length(new_rows) + 1]] <- data.table(
    variant = variant,
    variant_label = variant_label,
    class_label = class_label,
    exposure = exposure,
    year = years,
    beta_log_per_1sd_exposure = as.numeric(fit_result$beta),
    status = fifelse(is.finite(as.numeric(fit_result$beta)), "ok", "failed")
  )
  new_stats[[length(new_stats) + 1]] <- yearly_fit_stats(d, fit_result, class_label, exposure)
}

new_long <- rbindlist(new_rows, fill = TRUE)
new_long[, abs_beta_log := abs(beta_log_per_1sd_exposure)]
new_fit_stats <- rbindlist(new_stats, fill = TRUE)

old_long <- if (file.exists(out_long)) fread(out_long) else data.table()
old_fit_stats <- if (file.exists(out_fit_stats)) fread(out_fit_stats) else data.table()
old_long <- old_long[variant != "fixest_pooled_linear_time"]
old_fit_stats <- old_fit_stats[variant != "fixest_pooled_linear_time"]
long <- rbindlist(list(old_long, new_long), fill = TRUE)
fit_stats <- rbindlist(list(old_fit_stats, new_fit_stats), fill = TRUE)

fwrite(long, out_long)
fwrite(fit_stats, out_fit_stats)

summary <- long[status == "ok" & year %between% c(2000, 2024),
  .(median_abs_beta = median(abs_beta_log, na.rm = TRUE)),
  by = .(variant, variant_label, year, class_label)
]
summary[, class_label := factor(class_label, levels = domain_order)]
setorder(summary, variant, year, class_label)
fwrite(summary, out_by_year)

n_exp <- long[status == "ok" & year %between% c(2000, 2024),
  .(n_exposures = uniqueN(exposure)),
  by = .(variant, variant_label, year, class_label)
]
fwrite(n_exp, out_n)

cat("Wrote pooled linear time variant and updated combined variant tables\n")
print(n_exp[variant == "fixest_pooled_linear_time", .(min_n = min(n_exposures), max_n = max(n_exposures)), by = class_label])
