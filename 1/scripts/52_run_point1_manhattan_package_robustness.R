#!/usr/bin/env Rscript

# Package-based robustness/sensitivity models for the Point 1 endpoint Manhattan screen.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(fixest)
  library(mgcv)
  library(geepack)
  library(rexposome)
  library(parallel)
})

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)][1])))
here <- normalizePath(file.path(script_dir, ".."))
tab <- file.path(here, "tables")

meta_path <- file.path(tab, "point1_manhattan_robustness_input.csv")
panel_path <- file.path(tab, "point1_manhattan_robustness_panel_for_r.csv")
out_path <- file.path(tab, "point1_manhattan_package_robustness_beta_r2.csv")
summary_path <- file.path(tab, "point1_manhattan_package_robustness_summary.csv")

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

safe_num <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  if (length(x) == 0) NA_real_ else x[[1]]
}

fixest_r2_value <- function(fit, type) {
  val <- tryCatch(fixest::r2(fit, type = type), error = function(e) NA_real_)
  safe_num(val)
}

base_row <- function(r, model_id, model_label, status = "ok", note = "") {
  tibble(
    phenotype = r$phenotype,
    phenotype_scope = r$phenotype_scope,
    phenotype_label = r$phenotype_label,
    outcome_col = r$outcome_col,
    exposure = r$exposure,
    exposure_zh = r$exposure_zh,
    domain = r$domain,
    mechanistic_domain_en = r$mechanistic_domain_en,
    source_class = r$source_class,
    construct = r$construct,
    window = r$window,
    form = r$form,
    is_dairy_weighted_exposure = r$is_dairy_weighted_exposure,
    measurement_support_variable = r$measurement_support_variable,
    main_beta = r$beta,
    main_p = r$plot_p,
    main_incr_r2 = r$incr_r2,
    native_signal_tier = r$native_signal_tier,
    model_id = model_id,
    model_label = model_label,
    status = status,
    note = note,
    beta = NA_real_,
    se = NA_real_,
    p = NA_real_,
    r2 = NA_real_,
    adjusted_r2 = NA_real_,
    n = NA_integer_,
    n_states = NA_integer_,
    n_clusters = NA_integer_,
    r2_type = NA_character_
  )
}

prepare_data <- function(panel, r) {
  x_col <- r$exposure
  y_col <- r$outcome_col
  needed <- c("state_alpha", "year", "month", "year_month", "time_index", "milk_cows_head", y_col, x_col)
  if (!all(needed %in% names(panel))) {
    miss <- paste(setdiff(needed, names(panel)), collapse = ", ")
    return(list(status = paste0("missing_columns: ", miss), data = NULL))
  }
  d <- panel[, needed] %>%
    mutate(
      year_num = as.numeric(year),
      year_scaled = as.numeric(scale(year_num)),
      w = as.numeric(milk_cows_head),
      y_raw = as.numeric(.data[[y_col]]),
      x_raw = as.numeric(.data[[x_col]])
    ) %>%
    filter(is.finite(y_raw), is.finite(x_raw), is.finite(w), w > 0) %>%
    mutate(
      state_alpha = droplevels(factor(as.character(state_alpha))),
      year_month = droplevels(factor(as.character(year_month))),
      month = droplevels(factor(as.character(month)))
    )
  if (nrow(d) < 60 || n_distinct(d$state_alpha) < 3) {
    return(list(status = "too_few", data = d))
  }
  d <- d %>%
    arrange(state_alpha, year_num, as.integer(as.character(month))) %>%
    mutate(
      y_z = standardize(transform_y(y_raw)),
      x_z = standardize(x_raw)
    ) %>%
    mutate(
      state_alpha = droplevels(state_alpha),
      year_month = droplevels(year_month),
      month = droplevels(month)
    )
  list(status = "ok", data = d)
}

fit_fixest <- function(d, r, model_id, model_label, formula_text, use_weights = TRUE) {
  out <- base_row(r, model_id, model_label)
  dd <- d
  if (nrow(dd) < 60 || n_distinct(dd$state_alpha) < 3) {
    out$status <- "too_few"
    out$n <- nrow(dd)
    out$n_states <- n_distinct(dd$state_alpha)
    return(out)
  }
  weight_formula <- if (isTRUE(use_weights)) ~w else NULL
  fit <- tryCatch(
    feols(as.formula(formula_text), data = dd, weights = weight_formula, cluster = ~state_alpha, warn = FALSE, notes = FALSE),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    out$status <- "failed"
    out$note <- conditionMessage(fit)
    out$n <- nrow(dd)
    out$n_states <- n_distinct(dd$state_alpha)
    return(out)
  }
  ct <- tryCatch(coeftable(fit), error = function(e) NULL)
  out$beta <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Estimate"] else NA_real_
  out$se <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Std. Error"] else NA_real_
  out$p <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Pr(>|t|)"] else NA_real_
  if (!is.finite(out$beta)) {
    out$status <- "collinear"
    out$note <- "x_z was removed or not estimable under this fixest specification"
  }
  out$r2 <- fixest_r2_value(fit, "r2")
  out$adjusted_r2 <- fixest_r2_value(fit, "ar2")
  out$n <- nobs(fit)
  out$n_states <- n_distinct(dd$state_alpha)
  out$n_clusters <- n_distinct(dd$state_alpha)
  out$r2_type <- "fixest_global_r2"
  out
}

fit_mgcv <- function(d, r) {
  out <- base_row(r, "mgcv_gam", "mgcv GAM")
  dd <- d %>% mutate(state_alpha = factor(state_alpha), month_num = as.numeric(as.character(month)))
  fit <- tryCatch(
    gam(
      y_z ~ x_z + s(time_index, k = 12) + s(month_num, bs = "cc", k = 12) + s(state_alpha, bs = "re"),
      data = dd,
      weights = w,
      method = "REML"
    ),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    out$status <- "failed"
    out$note <- conditionMessage(fit)
    out$n <- nrow(dd)
    out$n_states <- n_distinct(dd$state_alpha)
    return(out)
  }
  sm <- summary(fit)
  pt <- sm$p.table
  out$beta <- if ("x_z" %in% rownames(pt)) pt["x_z", "Estimate"] else NA_real_
  out$se <- if ("x_z" %in% rownames(pt)) pt["x_z", "Std. Error"] else NA_real_
  out$p <- if ("x_z" %in% rownames(pt)) pt["x_z", "Pr(>|t|)"] else NA_real_
  out$r2 <- safe_num(sm$dev.expl)
  out$adjusted_r2 <- safe_num(sm$r.sq)
  out$n <- nobs(fit)
  out$n_states <- n_distinct(dd$state_alpha)
  out$n_clusters <- n_distinct(dd$state_alpha)
  out$r2_type <- "mgcv_deviance_explained_and_adj_rsq"
  out
}

fit_gee <- function(d, r) {
  out <- base_row(r, "geepack_gee_independence", "geepack GEE independence")
  dd <- d %>% arrange(state_alpha, time_index)
  fit <- tryCatch(
    geeglm(
      y_z ~ x_z + state_alpha + factor(year) + month,
      data = dd,
      id = state_alpha,
      waves = time_index,
      weights = w,
      corstr = "independence"
    ),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    out$status <- "failed"
    out$note <- conditionMessage(fit)
    out$n <- nrow(dd)
    out$n_states <- n_distinct(dd$state_alpha)
    return(out)
  }
  ct <- tryCatch(summary(fit)$coefficients, error = function(e) NULL)
  out$beta <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Estimate"] else NA_real_
  out$se <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Std.err"] else NA_real_
  out$p <- if (!is.null(ct) && "x_z" %in% rownames(ct)) ct["x_z", "Pr(>|W|)"] else NA_real_
  yhat <- as.numeric(fitted(fit))
  out$r2 <- weighted_r2(dd$y_z, yhat, dd$w)
  out$adjusted_r2 <- adjust_r2(out$r2, nrow(dd), length(coef(fit)))
  out$n <- nrow(dd)
  out$n_states <- n_distinct(dd$state_alpha)
  out$n_clusters <- n_distinct(dd$state_alpha)
  out$r2_type <- "weighted_pseudo_r2_from_fitted_state_year_month"
  out
}

weighted_residuals <- function(y, mm, w) {
  sw <- sqrt(w / mean(w, na.rm = TRUE))
  fit <- lm.fit(mm * sw, y * sw)
  as.numeric(y - mm %*% fit$coefficients)
}

fit_rexposome <- function(d, r) {
  out <- base_row(r, "rexposome_residual_exwas", "rexposome residual ExWAS")
  dd <- d %>% filter(is.finite(y_z), is.finite(x_z), is.finite(w))
  mm <- model.matrix(~ state_alpha + year_month, data = dd)
  y_res <- weighted_residuals(dd$y_z, mm, dd$w)
  x_res <- weighted_residuals(dd$x_z, mm, dd$w)
  ok <- is.finite(y_res) & is.finite(x_res)
  y_res <- y_res[ok]
  x_res <- x_res[ok]
  if (length(y_res) < 60) {
    out$status <- "too_few"
    out$n <- length(y_res)
    out$n_states <- n_distinct(dd$state_alpha)
    return(out)
  }
  x_sd <- sd(x_res, na.rm = TRUE)
  y_sd <- sd(y_res, na.rm = TRUE)
  if (!is.finite(x_sd) || !is.finite(y_sd) || x_sd < 1e-8 || y_sd < 1e-8) {
    out$status <- "near_zero_residual_variance"
    out$n <- length(y_res)
    out$n_states <- n_distinct(dd$state_alpha)
    out$note <- sprintf("Residual sd too small after state + year-month FE: x_sd=%g, y_sd=%g", x_sd, y_sd)
    return(out)
  }
  y_res <- standardize(y_res)
  x_res <- standardize(x_res)
  ex <- data.frame(x_res = x_res)
  ph <- data.frame(y_res = y_res)
  rownames(ex) <- paste0("s", seq_along(y_res))
  rownames(ph) <- rownames(ex)
  desc <- data.frame(Family = r$domain, type = "numeric")
  rownames(desc) <- "x_res"
  exfit <- tryCatch({
    es <- loadExposome(exposures = ex, description = desc, phenotype = ph, description.famCol = "Family", warnings = FALSE)
    exwas(es, y_res ~ x_res, family = "gaussian", verbose = FALSE, warnings = FALSE)
  }, error = function(e) e)
  lmfit <- lm(y_res ~ x_res)
  ct <- summary(lmfit)$coefficients
  if (!inherits(exfit, "error")) {
    comp <- as.data.frame(slot(exfit, "comparison"))
    out$beta <- safe_num(comp["x_res", "effect"])
    out$p <- safe_num(comp["x_res", "pvalue"])
    out$note <- "beta/p from rexposome on state + year-month FE residuals; R2 from matching residual lm"
  } else {
    out$beta <- ct["x_res", "Estimate"]
    out$p <- ct["x_res", "Pr(>|t|)"]
    out$status <- "rexposome_failed_lm_fallback"
    out$note <- conditionMessage(exfit)
  }
  out$se <- ct["x_res", "Std. Error"]
  out$r2 <- summary(lmfit)$r.squared
  out$adjusted_r2 <- summary(lmfit)$adj.r.squared
  out$n <- length(y_res)
  out$n_states <- n_distinct(dd$state_alpha)
  out$n_clusters <- n_distinct(dd$state_alpha)
  out$r2_type <- "residualized_lm_r2"
  out
}

meta <- read_csv(meta_path, show_col_types = FALSE)
panel <- read_csv(panel_path, show_col_types = FALSE)
max_rows <- suppressWarnings(as.integer(Sys.getenv("POINT1_ROBUSTNESS_MAX_ROWS", "")))
if (is.finite(max_rows) && max_rows > 0) {
  meta <- meta %>% slice_head(n = max_rows)
  cat(sprintf("Debug row limit active: %d Manhattan rows\n", nrow(meta)))
}

model_defs <- list(
  list(id = "fixest_state_trend", label = "fixest FE + state-specific trend", type = "fixest", formula = "y_z ~ x_z | state_alpha + month + state_alpha[year_scaled]", use_weights = TRUE),
  list(id = "mgcv_gam", label = "mgcv GAM", type = "mgcv"),
  list(id = "geepack_gee_independence", label = "geepack GEE independence", type = "gee"),
  list(id = "rexposome_residual_exwas", label = "rexposome residual ExWAS", type = "rexposome")
)

fit_one_meta_row <- function(i) {
  r <- meta[i, ]
  prep <- prepare_data(panel, r)
  if (prep$status != "ok") {
    out_bad <- vector("list", length(model_defs))
    j <- 0L
    for (m in model_defs) {
      j <- j + 1L
      out_bad[[j]] <- base_row(r, m$id, m$label, status = prep$status)
      out_bad[[j]]$n <- if (is.null(prep$data)) NA_integer_ else nrow(prep$data)
      out_bad[[j]]$n_states <- if (is.null(prep$data)) NA_integer_ else n_distinct(prep$data$state_alpha)
    }
    return(bind_rows(out_bad))
  }
  d <- prep$data
  out_row <- vector("list", length(model_defs))
  j <- 0L
  for (m in model_defs) {
    j <- j + 1L
    out_row[[j]] <- tryCatch({
      if (m$type == "fixest") {
        fit_fixest(d, r, m$id, m$label, m$formula, use_weights = if (!is.null(m$use_weights)) m$use_weights else TRUE)
      } else if (m$type == "mgcv") {
        fit_mgcv(d, r)
      } else if (m$type == "gee") {
        fit_gee(d, r)
      } else {
        fit_rexposome(d, r)
      }
    }, error = function(e) {
      base_row(r, m$id, m$label, status = "failed", note = conditionMessage(e))
    })
  }
  if (i %% 25 == 0) cat(sprintf("processed %d / %d Manhattan rows\n", i, nrow(meta)))
  bind_rows(out_row)
}

cores_env <- suppressWarnings(as.integer(Sys.getenv("POINT1_ROBUSTNESS_CORES", "")))
cores <- if (is.finite(cores_env) && cores_env > 0) cores_env else max(1L, min(6L, detectCores() - 1L))
cat(sprintf("Running %d Manhattan rows on %d core(s)\n", nrow(meta), cores))
rows <- if (cores > 1) {
  mclapply(seq_len(nrow(meta)), fit_one_meta_row, mc.cores = cores, mc.preschedule = FALSE)
} else {
  lapply(seq_len(nrow(meta)), fit_one_meta_row)
}

out <- bind_rows(rows) %>%
  arrange(model_id, phenotype_scope, domain, exposure)
write_csv(out, out_path)

summary <- out %>%
  group_by(model_id, model_label, phenotype_scope, status) %>%
  summarise(
    n_rows = n(),
    n_ok_beta = sum(is.finite(beta)),
    median_r2 = median(r2, na.rm = TRUE),
    median_adjusted_r2 = median(adjusted_r2, na.rm = TRUE),
    .groups = "drop"
  )
write_csv(summary, summary_path)

print(summary)
cat(sprintf("Wrote %s rows=%d\n", out_path, nrow(out)))
