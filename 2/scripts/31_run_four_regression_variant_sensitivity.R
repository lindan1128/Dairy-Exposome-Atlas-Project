#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  .libPaths(c("/private/tmp/Rlib_point2", .libPaths()))
  library(data.table)
  library(fixest)
  library(mgcv)
  library(geepack)
  library(ggplot2)
})

cmd <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", cmd[grep("--file=", cmd)][1])
point <- normalizePath(file.path(dirname(script_path), ".."))
tab <- file.path(point, "tables")
fig <- file.path(point, "figures")

panel_file <- file.path(tab, "point2_expanded_sensitivity_panel_for_r.csv")
selection_file <- file.path(tab, "point2_common_sense_expanded_kept_variables.csv")
out_long <- file.path(tab, "point2_four_regression_variant_yearly_sensitivity_long.csv")
out_by_year <- file.path(tab, "point2_four_regression_variant_yearly_sensitivity_by_year.csv")
out_n <- file.path(tab, "point2_four_regression_variant_yearly_sensitivity_n_exposures.csv")
out_fit_stats <- file.path(tab, "point2_four_regression_variant_yearly_fit_stats.csv")

domain_order <- c("Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand")
lb_to_kg <- 0.45359237
colors <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage" = "#1E7A8D",
  "Feed market" = "#fbc4ab",
  "Dairy market" = "#E47666",
  "Market demand" = "#f09d51"
)
variant_labels <- c(
  "fixest_fe" = "fixest FE",
  "fixest_year_month_fe" = "fixest FE + year-month FE",
  "fixest_pooled_linear_time" = "FE + x×year",
  "mgcv_gam" = "mgcv GAM",
  "geepack_gee" = "geepack GEE"
)

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

make_result <- function(beta, fitted, exposure_contribution, model_df,
                        global_r2 = NA_real_, global_adj_r2 = NA_real_,
                        r2_type = "weighted_prediction_r2") {
  list(
    beta = beta,
    fitted = as.numeric(fitted),
    exposure_contribution = as.numeric(exposure_contribution),
    model_df = as.numeric(model_df),
    global_r2 = as.numeric(global_r2),
    global_adj_r2 = as.numeric(global_adj_r2),
    r2_type = r2_type,
    status = "ok"
  )
}

failed_result <- function(n_obs) {
  list(
    beta = rep(NA_real_, length(2000:2025)),
    fitted = rep(NA_real_, n_obs),
    exposure_contribution = rep(NA_real_, n_obs),
    model_df = NA_real_,
    global_r2 = NA_real_,
    global_adj_r2 = NA_real_,
    r2_type = "not_estimated",
    status = "failed"
  )
}

yearly_fit_stats <- function(d, fit_result, variant, variant_label, domain, exposure) {
  rbindlist(lapply(2000:2025, function(yy) {
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
      class_label = domain,
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

make_work_data <- function(panel, exposure) {
  d <- panel[, .(
    state_alpha,
    year,
    month,
    milk_cows_head,
    log_per_cow,
    lag12_log_per_cow,
    x_raw = get(exposure)
  )]
  d <- d[is.finite(log_per_cow) & is.finite(x_raw) & milk_cows_head > 0]
  d[, x_z := z(x_raw)]
  d <- d[is.finite(x_z)]
  d[, state_alpha := droplevels(factor(state_alpha))]
  d[, year_f := factor(year, levels = 2000:2025)]
  d[, year_f := droplevels(year_f)]
  d[, month_f := factor(month)]
  d[, month_f := droplevels(month_f)]
  d[, year_num := as.numeric(year)]
  d[, year_scaled := as.numeric(scale(year_num))]
  d[, ym_f := factor(sprintf("%04d_%02d", year, month))]
  d[, ym_f := droplevels(ym_f)]
  for (yy in 2000:2025) {
    d[, paste0("x_y_", yy) := fifelse(year == yy, x_z, 0)]
  }
  d
}

fit_fixest <- function(d, year_month_fe = FALSE) {
  x_cols <- paste0("x_y_", 2000:2025)
  rhs <- x_cols
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6) return(failed_result(nrow(d)))
  fe_part <- if (year_month_fe) {
    "state_alpha + ym_f"
  } else {
    "state_alpha + month_f + year_f"
  }
  fml <- as.formula(paste0(
    "log_per_cow ~ ", paste(rhs, collapse = " + "),
    " | ", fe_part
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
  make_result(
    beta = beta,
    fitted = fitted(fit),
    exposure_contribution = beta[as.character(paste0("x_y_", d$year))] * d$x_z,
    model_df = model_df,
    global_r2 = global_r2,
    global_adj_r2 = global_adj_r2,
    r2_type = "fixest_model_r2_and_yearly_weighted_prediction_r2"
  )
}


fit_fixest_pooled_linear_time <- function(d) {
  d <- copy(d)
  d[, x_year_scaled := x_z * year_scaled]
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6) return(failed_result(nrow(d)))
  fml <- log_per_cow ~ x_z + x_year_scaled | state_alpha + month_f + year_f
  fit <- tryCatch({
    tmp <- NULL
    capture.output(tmp <- feols(fml, data = d, weights = ~milk_cows_head, cluster = ~state_alpha, warn = FALSE))
    tmp
  }, error = function(e) NULL)
  if (is.null(fit)) return(failed_result(nrow(d)))
  co <- coef(fit)
  b0 <- if ("x_z" %in% names(co)) unname(co[["x_z"]]) else NA_real_
  b1 <- if ("x_year_scaled" %in% names(co)) unname(co[["x_year_scaled"]]) else 0
  year_center <- mean(d$year_num, na.rm = TRUE)
  year_sd <- stats::sd(d$year_num, na.rm = TRUE)
  if (!is.finite(year_sd) || year_sd <= 1e-12) return(failed_result(nrow(d)))
  year_scaled_grid <- ((2000:2025) - year_center) / year_sd
  beta <- b0 + b1 * year_scaled_grid
  beta_obs <- b0 + b1 * d$year_scaled
  global_r2 <- tryCatch(as.numeric(fitstat(fit, "r2")[[1]]), error = function(e) NA_real_)
  global_adj_r2 <- tryCatch(as.numeric(fitstat(fit, "ar2")[[1]]), error = function(e) NA_real_)
  model_df <- tryCatch(as.numeric(fit$nparams), error = function(e) length(coef(fit)))
  if (length(model_df) != 1 || !is.finite(model_df)) model_df <- length(coef(fit))
  make_result(
    beta = beta,
    fitted = fitted(fit),
    exposure_contribution = beta_obs * d$x_z,
    model_df = model_df,
    global_r2 = global_r2,
    global_adj_r2 = global_adj_r2,
    r2_type = "fixest_pooled_linear_time_interaction_model_r2_and_yearly_weighted_prediction_r2"
  )
}

fit_gee <- function(d) {
  x_cols <- paste0("x_y_", 2000:2025)
  fml <- as.formula(paste0(
    "log_per_cow ~ 0 + ", paste(x_cols, collapse = " + "),
    " + month_f + year_f"
  ))
  setorder(d, state_alpha, year, month)
  fit <- tryCatch(
    suppressWarnings({
      tmp <- NULL
      capture.output(tmp <- geeglm(fml, data = d, id = state_alpha, weights = milk_cows_head,
                                   corstr = "exchangeable", scale.fix = FALSE))
      tmp
    }),
    error = function(e) NULL
  )
  if (is.null(fit)) return(failed_result(nrow(d)))
  yhat <- fitted(fit)
  global_r2 <- weighted_r2(d$log_per_cow, yhat, d$milk_cows_head)
  model_df <- sum(is.finite(coef(fit)))
  beta <- safe_coef(coef(fit), x_cols)
  make_result(
    beta = beta,
    fitted = yhat,
    exposure_contribution = beta[as.character(paste0("x_y_", d$year))] * d$x_z,
    model_df = model_df,
    global_r2 = global_r2,
    global_adj_r2 = adj_r2(global_r2, length(yhat), model_df),
    r2_type = "gee_population_average_weighted_pseudo_r2"
  )
}

fit_gam <- function(d) {
  fit <- tryCatch(
    suppressWarnings(bam(
      log_per_cow ~ x_z +
        s(year_num, by = x_z, k = 8) +
        state_alpha + month_f,
      data = d,
      weights = milk_cows_head,
      method = "fREML",
      discrete = TRUE
    )),
    error = function(e) NULL
  )
  if (is.null(fit)) return(failed_result(nrow(d)))
  ref_state <- levels(factor(d$state_alpha))[1]
  years <- 2000:2025
  nd1 <- data.frame(
    x_z = 1,
    year_num = years,
    state_alpha = factor(ref_state, levels = levels(factor(d$state_alpha))),
    month_f = factor(6, levels = levels(d$month_f))
  )
  nd0 <- nd1
  nd0$x_z <- 0
  sm <- summary(fit)
  d0 <- copy(d)
  d0$x_z <- 0
  for (yy in 2000:2025) {
    d0[[paste0("x_y_", yy)]] <- 0
  }
  fitted_full <- fitted(fit)
  fitted_no_exposure <- as.numeric(predict(fit, newdata = d0, type = "link"))
  make_result(
    beta = as.numeric(predict(fit, newdata = nd1, type = "link") - predict(fit, newdata = nd0, type = "link")),
    fitted = fitted_full,
    exposure_contribution = fitted_full - fitted_no_exposure,
    model_df = sum(sm$edf, na.rm = TRUE),
    global_r2 = sm$dev.expl,
    global_adj_r2 = sm$r.sq,
    r2_type = "mgcv_deviance_explained_adjusted_r2_and_yearly_weighted_prediction_r2"
  )
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
panel[, state_alpha := factor(state_alpha)]
setorder(panel, state_alpha, year, month)

rows <- list()
fit_stats_rows <- list()
years <- 2000:2025
for (ii in seq_len(nrow(selected))) {
  exposure <- selected$exposure[ii]
  domain <- selected$class_label[ii]
  if (!exposure %in% names(panel)) next
  message(sprintf("[%s/%s] %s", ii, nrow(selected), exposure))
  d <- make_work_data(panel, exposure)
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6 || uniqueN(d$x_z) <= 1) next

  fits <- list(
    fixest_fe = list(result = fit_fixest(d), data = d),
    fixest_year_month_fe = list(result = fit_fixest(d, year_month_fe = TRUE), data = d),
    fixest_pooled_linear_time = list(result = fit_fixest_pooled_linear_time(d), data = d),
    mgcv_gam = list(result = fit_gam(d), data = d),
    geepack_gee = list(result = fit_gee(d), data = d)
  )
  for (variant in names(fits)) {
    fit_result <- fits[[variant]]$result
    fit_data <- fits[[variant]]$data
    vals <- fit_result$beta
    rows[[length(rows) + 1]] <- data.table(
      variant = variant,
      variant_label = variant_labels[[variant]],
      class_label = domain,
      exposure = exposure,
      year = years,
      beta_log_per_1sd_exposure = as.numeric(vals),
      status = fifelse(is.finite(as.numeric(vals)), "ok", "failed")
    )
    fit_stats_rows[[length(fit_stats_rows) + 1]] <- yearly_fit_stats(
      d = fit_data,
      fit_result = fit_result,
      variant = variant,
      variant_label = variant_labels[[variant]],
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

cat("Wrote", out_long, "\n")
cat("Wrote", out_by_year, "\n")
cat("Wrote", out_n, "\n")
cat("Wrote", out_fit_stats, "\n")
cat("\nVariant/domain counts\n")
print(n_exp[, .(min_n = min(n_exposures), max_n = max(n_exposures)), by = .(variant_label, class_label)])
