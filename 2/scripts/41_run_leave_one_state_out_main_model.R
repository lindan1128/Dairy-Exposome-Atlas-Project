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
out_long <- file.path(tab, "point2_leave_one_state_out_main_model_yearly_sensitivity_long.csv")
out_by_year <- file.path(tab, "point2_leave_one_state_out_main_model_yearly_sensitivity_by_year.csv")
out_fit_stats <- file.path(tab, "point2_leave_one_state_out_main_model_yearly_fit_stats.csv")
out_r2_by_year <- file.path(tab, "point2_leave_one_state_out_main_model_yearly_r2_by_year.csv")

domain_order <- c("Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand")
years <- 2000:2025
lb_to_kg <- 0.45359237
variant <- "fixest_fe_leave_one_state_out"
variant_label <- "fixest FE leave-one-state-out"

z <- function(x) {
  x <- as.numeric(x)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 1e-12) return(rep(NA_real_, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

safe_coef <- function(coefs, names) {
  out <- rep(NA_real_, length(names))
  names(out) <- names
  hit <- intersect(names, names(coefs))
  out[hit] <- unname(coefs[hit])
  out
}

weighted_r2 <- function(y, yhat, w) {
  ok <- is.finite(y) & is.finite(yhat) & is.finite(w) & w > 0
  if (sum(ok) < 3) return(NA_real_)
  y <- y[ok]
  yhat <- yhat[ok]
  w <- w[ok]
  ybar <- stats::weighted.mean(y, w)
  sst <- sum(w * (y - ybar)^2)
  if (!is.finite(sst) || sst <= 1e-12) return(NA_real_)
  1 - sum(w * (y - yhat)^2) / sst
}

adj_r2 <- function(r2, n, p) {
  if (!is.finite(r2) || !is.finite(n) || !is.finite(p) || n <= p + 1) return(NA_real_)
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

failed_result <- function(n_obs) {
  list(
    beta = rep(NA_real_, length(years)),
    fitted = rep(NA_real_, n_obs),
    exposure_contribution = rep(NA_real_, n_obs),
    model_df = NA_real_,
    global_r2 = NA_real_,
    global_adj_r2 = NA_real_,
    r2_type = "not_estimated",
    status = "failed"
  )
}

make_work_data <- function(panel, exposure) {
  d <- panel[, .(
    state_alpha,
    year,
    month,
    milk_cows_head,
    log_per_cow,
    x_raw = get(exposure)
  )]
  d <- d[is.finite(log_per_cow) & is.finite(x_raw) & milk_cows_head > 0]
  d[, x_z := z(x_raw)]
  d <- d[is.finite(x_z)]
  d[, state_alpha := droplevels(factor(state_alpha))]
  d[, year_f := factor(year, levels = years)]
  d[, year_f := droplevels(year_f)]
  d[, month_f := factor(month)]
  d[, month_f := droplevels(month_f)]
  for (yy in years) {
    d[, paste0("x_y_", yy) := fifelse(year == yy, x_z, 0)]
  }
  d
}

fit_main_model <- function(d) {
  x_cols <- paste0("x_y_", years)
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6 || uniqueN(d$x_z) <= 1) {
    return(failed_result(nrow(d)))
  }
  fml <- as.formula(paste0(
    "log_per_cow ~ ", paste(x_cols, collapse = " + "),
    " | state_alpha + month_f + year_f"
  ))
  fit <- tryCatch({
    tmp <- NULL
    capture.output(tmp <- feols(fml, data = d, weights = ~milk_cows_head, cluster = ~state_alpha, warn = FALSE))
    tmp
  }, error = function(e) NULL)
  if (is.null(fit)) return(failed_result(nrow(d)))
  global_r2 <- tryCatch(as.numeric(fitstat(fit, "r2")[[1]]), error = function(e) NA_real_)
  global_adj_r2 <- tryCatch(as.numeric(fitstat(fit, "ar2")[[1]]), error = function(e) NA_real_)
  model_df <- tryCatch(as.numeric(fit$nparams), error = function(e) length(coef(fit)))
  if (length(model_df) != 1 || !is.finite(model_df)) model_df <- length(coef(fit))
  beta <- safe_coef(coef(fit), x_cols)
  list(
    beta = beta,
    fitted = as.numeric(fitted(fit)),
    exposure_contribution = beta[as.character(paste0("x_y_", d$year))] * d$x_z,
    model_df = model_df,
    global_r2 = global_r2,
    global_adj_r2 = global_adj_r2,
    r2_type = "fixest_model_r2_and_yearly_weighted_prediction_r2",
    status = "ok"
  )
}

yearly_fit_stats <- function(d, fit_result, omitted_state, domain, exposure) {
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
    reduced_adj_r2 <- adj_r2(reduced_r2, n_obs, fit_result$model_df - 1)
    partial_r2 <- if (is.finite(reduced_sse) && reduced_sse > 1e-12) {
      max(0, (reduced_sse - full_sse) / reduced_sse)
    } else {
      NA_real_
    }
    partial_adj_r2 <- if (
      is.finite(full_sse) && is.finite(reduced_sse) &&
        reduced_sse > 1e-12 &&
        is.finite(fit_result$model_df) &&
        n_obs > fit_result$model_df + 1
    ) {
      1 - (full_sse / (n_obs - fit_result$model_df)) /
        (reduced_sse / (n_obs - fit_result$model_df + 1))
    } else {
      NA_real_
    }
    incremental_r2 <- if (is.finite(r2) && is.finite(reduced_r2)) max(0, r2 - reduced_r2) else NA_real_
    adjusted_incremental_r2 <- if (is.finite(full_adj_r2) && is.finite(reduced_adj_r2)) {
      full_adj_r2 - reduced_adj_r2
    } else {
      NA_real_
    }
    data.table(
      variant = variant,
      variant_label = variant_label,
      omitted_state = omitted_state,
      domain_label = domain,
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
      status = fit_result$status
    )
  }), fill = TRUE)
}

selected <- fread(selection_file)
if (!"class_label" %in% names(selected) && "domain_label" %in% names(selected)) selected[, class_label := domain_label]
if (!"domain_label" %in% names(selected) && "class_label" %in% names(selected)) selected[, domain_label := class_label]
selected <- selected[startsWith(expanded_selection_status, "kept_expanded")]
selected <- selected[domain_label %in% domain_order]

panel <- fread(panel_file)
panel <- panel[year %between% c(2000, 2025)]
if (!"milk_per_cow_kg" %in% names(panel)) {
  panel[, milk_per_cow_kg := fifelse(
    milk_production_lb > 0 & milk_cows_head > 0,
    milk_production_lb * lb_to_kg / milk_cows_head,
    NA_real_
  )]
}
panel[, log_per_cow := log(fifelse(milk_per_cow_kg > 0, milk_per_cow_kg, NA_real_))]
panel[, state_alpha := factor(state_alpha)]
states <- sort(as.character(unique(panel$state_alpha)))

rows <- list()
fit_stats_rows <- list()

for (ii in seq_len(nrow(selected))) {
  exposure <- selected$exposure[ii]
  domain <- selected$domain_label[ii]
  if (!exposure %in% names(panel)) next
  message(sprintf("[%s/%s] %s", ii, nrow(selected), exposure))
  d_all <- make_work_data(panel, exposure)
  if (nrow(d_all) < 300 || uniqueN(d_all$state_alpha) < 6 || uniqueN(d_all$x_z) <= 1) next

  for (omitted_state in states) {
    d <- d_all[as.character(state_alpha) != omitted_state]
    d[, state_alpha := droplevels(state_alpha)]
    d[, year_f := droplevels(year_f)]
    d[, month_f := droplevels(month_f)]
    fit_result <- fit_main_model(d)
    vals <- fit_result$beta
    rows[[length(rows) + 1]] <- data.table(
      variant = variant,
      variant_label = variant_label,
      omitted_state = omitted_state,
      domain_label = domain,
      exposure = exposure,
      year = years,
      beta_log_per_1sd_exposure = as.numeric(vals),
      status = fifelse(is.finite(as.numeric(vals)), "ok", "failed")
    )
    fit_stats_rows[[length(fit_stats_rows) + 1]] <- yearly_fit_stats(
      d = d,
      fit_result = fit_result,
      omitted_state = omitted_state,
      domain = domain,
      exposure = exposure
    )
  }
}

long <- rbindlist(rows, fill = TRUE)
long[, abs_beta_log := abs(beta_log_per_1sd_exposure)]
fwrite(long, out_long)

fit_stats <- rbindlist(fit_stats_rows, fill = TRUE)
fwrite(fit_stats, out_fit_stats)

summary <- long[status == "ok" & year %between% c(2000, 2024),
  .(
    median_abs_beta = median(abs_beta_log, na.rm = TRUE),
    q05_abs_beta = quantile(abs_beta_log, 0.05, na.rm = TRUE, names = FALSE),
    q95_abs_beta = quantile(abs_beta_log, 0.95, na.rm = TRUE, names = FALSE),
    n_exposures = uniqueN(exposure),
    n_omitted_states = uniqueN(omitted_state)
  ),
  by = .(variant, variant_label, year, domain_label)
]
summary[, domain_label := factor(domain_label, levels = domain_order)]
setorder(summary, variant, year, domain_label)
fwrite(summary, out_by_year)

r2_summary <- fit_stats[status == "ok" & year %between% c(2000, 2024),
  .(
    median_partial_r2_year = median(partial_r2_year, na.rm = TRUE),
    q05_partial_r2_year = quantile(partial_r2_year, 0.05, na.rm = TRUE, names = FALSE),
    q95_partial_r2_year = quantile(partial_r2_year, 0.95, na.rm = TRUE, names = FALSE),
    median_partial_adj_r2_year = median(partial_adj_r2_year, na.rm = TRUE),
    median_incremental_r2_year = median(incremental_r2_year, na.rm = TRUE),
    median_adjusted_incremental_r2_year = median(adjusted_incremental_r2_year, na.rm = TRUE),
    median_full_model_r2_year = median(r2_year, na.rm = TRUE),
    median_full_model_adj_r2_year = median(adj_r2_year, na.rm = TRUE),
    n_exposures = uniqueN(exposure),
    n_omitted_states = uniqueN(omitted_state)
  ),
  by = .(variant, variant_label, year, domain_label)
]
r2_summary[, domain_label := factor(domain_label, levels = domain_order)]
setorder(r2_summary, variant, year, domain_label)
fwrite(r2_summary, out_r2_by_year)

cat("Wrote", out_long, "\n")
cat("Wrote", out_by_year, "\n")
cat("Wrote", out_fit_stats, "\n")
cat("Wrote", out_r2_by_year, "\n")
cat("\nLeave-one-state-out dimensions\n")
print(long[, .(
  n_exposures = uniqueN(exposure),
  n_omitted_states = uniqueN(omitted_state),
  min_year = min(year),
  max_year = max(year)
)])
