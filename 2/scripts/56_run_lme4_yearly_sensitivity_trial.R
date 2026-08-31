#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  .libPaths(c("/private/tmp/Rlib_point2", .libPaths()))
  library(data.table)
  library(lme4)
})

cmd <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", cmd[grep("--file=", cmd)][1])
point <- normalizePath(file.path(dirname(script_path), ".."))
tab <- file.path(point, "tables")

panel_file <- file.path(tab, "point2_expanded_sensitivity_panel_for_r.csv")
selection_file <- file.path(tab, "point2_common_sense_expanded_kept_variables.csv")
sd_file <- file.path(tab, "point2_milk_per_cow_yearly_log_sd_for_beta_std.csv")
retained_file <- file.path(tab, "point2_beta_std_two_stage_three_metric_by_variable.csv")
out_long <- file.path(tab, "point2_lme4_yearly_sensitivity_trial_long.csv")
out_summary <- file.path(tab, "point2_lme4_yearly_sensitivity_trial_correlation_summary.csv")
out_pairs <- file.path(tab, "point2_lme4_yearly_sensitivity_trial_correlation_pairs.csv")

lb_to_kg <- 0.45359237
domain_order <- c("Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand")

data_z <- function(x) {
  x <- as.numeric(x)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 1e-12) return(rep(NA_real_, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

failed_rows <- function(class_label, exposure, years, status, note = "") {
  data.table(
    variant = "lme4_random_intercept_slope",
    variant_label = "LME + random state slope",
    class_label = class_label,
    exposure = exposure,
    year = years,
    beta_log_per_1sd_exposure = NA_real_,
    beta_std_per_1sd_exposure = NA_real_,
    status = status,
    note = note,
    singular = NA,
    n = NA_integer_,
    n_states = NA_integer_
  )
}

fit_lme_one <- function(panel, class_label, exposure, sd_tbl) {
  years <- 2000:2025
  needed <- c("state_alpha", "year", "month", "milk_cows_head", "log_per_cow", exposure)
  if (!all(needed %in% names(panel))) return(failed_rows(class_label, exposure, years, "missing_columns"))
  d <- panel[, ..needed]
  setnames(d, exposure, "x_raw")
  d <- d[is.finite(log_per_cow) & is.finite(x_raw) & is.finite(milk_cows_head) & milk_cows_head > 0]
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6 || uniqueN(d$x_raw) <= 1) {
    return(failed_rows(class_label, exposure, years, "too_few"))
  }
  d[, x_z := data_z(x_raw)]
  d <- d[is.finite(x_z)]
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6) return(failed_rows(class_label, exposure, years, "too_few_after_z"))
  d[, state_alpha := droplevels(factor(state_alpha))]
  d[, month_f := droplevels(factor(month))]
  d[, year_f := droplevels(factor(year, levels = years))]
  for (yy in years) {
    d[, paste0("x_y_", yy) := fifelse(year == yy, x_z, 0)]
  }
  x_cols <- paste0("x_y_", years)
  rhs <- paste(c(x_cols, "month_f", "year_f", "(1 + x_z || state_alpha)"), collapse = " + ")
  fml <- as.formula(paste0("log_per_cow ~ 0 + ", rhs))
  fit <- tryCatch({
    suppressWarnings(lmer(
      fml,
      data = d,
      weights = milk_cows_head,
      REML = FALSE,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 20000), check.conv.singular = "ignore")
    ))
  }, error = function(e) e)
  if (inherits(fit, "error")) {
    return(failed_rows(class_label, exposure, years, "failed", conditionMessage(fit)))
  }
  co <- fixef(fit)
  vals <- rep(NA_real_, length(years)); names(vals) <- years
  hit <- intersect(x_cols, names(co))
  vals[sub("x_y_", "", hit)] <- unname(co[hit])
  out <- data.table(
    variant = "lme4_random_intercept_slope",
    variant_label = "LME + random state slope",
    class_label = class_label,
    exposure = exposure,
    year = years,
    beta_log_per_1sd_exposure = as.numeric(vals),
    status = fifelse(is.finite(as.numeric(vals)), "ok", "collinear"),
    note = "lmer: fixed exposure-by-year terms, month/year fixed effects, random state intercept and exposure slope",
    singular = isSingular(fit, tol = 1e-4),
    n = nobs(fit),
    n_states = uniqueN(d$state_alpha)
  )
  out <- merge(out, sd_tbl[, .(year, log_milk_per_cow_kg_weighted_sd)], by = "year", all.x = TRUE)
  out[, beta_std_per_1sd_exposure := beta_log_per_1sd_exposure / log_milk_per_cow_kg_weighted_sd]
  out[, log_milk_per_cow_kg_weighted_sd := NULL]
  out[]
}

selected <- fread(selection_file)
if (!"class_label" %in% names(selected) && "domain_label" %in% names(selected)) selected[, class_label := domain_label]
if (!"domain_label" %in% names(selected) && "class_label" %in% names(selected)) selected[, domain_label := class_label]
selected <- selected[startsWith(expanded_selection_status, "kept_expanded") & class_label %in% domain_order]
if (file.exists(retained_file)) {
  retained <- fread(retained_file)
  retained <- retained[!exposure %in% c("market_log_population_total", "storm_event_types")]
  selected <- selected[exposure %in% retained$exposure]
}
selected <- unique(selected[, .(class_label, exposure)])

panel <- fread(panel_file)
panel <- panel[year %between% c(2000, 2025)]
if (!"milk_per_cow_kg" %in% names(panel)) {
  panel[, milk_per_cow_kg := fifelse(milk_production_lb > 0 & milk_cows_head > 0, milk_production_lb * lb_to_kg / milk_cows_head, NA_real_)]
}
panel[, log_per_cow := log(fifelse(milk_per_cow_kg > 0, milk_per_cow_kg, NA_real_))]

sd_tbl <- fread(sd_file)
rows <- vector("list", nrow(selected))
for (ii in seq_len(nrow(selected))) {
  message(sprintf("[%s/%s] LME %s", ii, nrow(selected), selected$exposure[ii]))
  rows[[ii]] <- fit_lme_one(panel, selected$class_label[ii], selected$exposure[ii], sd_tbl)
}
long <- rbindlist(rows, fill = TRUE)
fwrite(long, out_long)

base <- fread(file.path(tab, "point2_herd_adjusted_yearly_sensitivity_beta_std.csv"))
base <- base[status == "ok" & year %between% c(2000, 2024) & class_label %in% domain_order]
base <- base[exposure %in% selected$exposure, .(class_label, exposure, year, main_beta_std = beta_std_per_1sd_exposure)]
alt <- long[status == "ok" & year %between% c(2000, 2024), .(class_label, exposure, year, model_beta_std = beta_std_per_1sd_exposure, singular)]
pairs <- merge(base, alt, by = c("class_label", "exposure", "year"))
fwrite(pairs, out_pairs)

summary <- pairs[, {
  ok <- is.finite(main_beta_std) & is.finite(model_beta_std)
  ct <- if (sum(ok) >= 3) suppressWarnings(cor.test(main_beta_std[ok], model_beta_std[ok])) else NULL
  .(
    n_points = sum(ok),
    n_exposures = uniqueN(exposure[ok]),
    n_years = uniqueN(year[ok]),
    pearson_r = if (!is.null(ct)) unname(ct$estimate) else NA_real_,
    pearson_p = if (!is.null(ct)) ct$p.value else NA_real_,
    singular_exposure_share = uniqueN(exposure[singular %in% TRUE]) / uniqueN(exposure),
    ok_exposure_count = uniqueN(exposure[ok])
  )
}]
summary[, `:=`(model = "lme4_random_intercept_slope", model_label = "LME + random state slope")]
setcolorder(summary, c("model", "model_label", "n_points", "n_exposures", "n_years", "pearson_r", "pearson_p", "singular_exposure_share", "ok_exposure_count"))
fwrite(summary, out_summary)
print(summary)
