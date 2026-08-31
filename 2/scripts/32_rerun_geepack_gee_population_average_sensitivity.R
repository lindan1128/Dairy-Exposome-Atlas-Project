#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  .libPaths(c("/private/tmp/Rlib_point2", .libPaths()))
  library(data.table)
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
  d[, year_f := droplevels(factor(year, levels = 2000:2025))]
  d[, month_f := droplevels(factor(month))]
  for (yy in 2000:2025) {
    d[, paste0("x_y_", yy) := fifelse(year == yy, x_z, 0)]
  }
  d
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
  if (is.null(fit)) return(rep(NA_real_, length(x_cols)))
  safe_coef(coef(fit), x_cols)
}

selected <- fread(selection_file)
if (!"class_label" %in% names(selected) && "domain_label" %in% names(selected)) selected[, class_label := domain_label]
if (!"domain_label" %in% names(selected) && "class_label" %in% names(selected)) selected[, domain_label := class_label]
selected <- selected[startsWith(expanded_selection_status, "kept_expanded")]
selected <- selected[domain_label %in% domain_order]

panel <- fread(panel_file)
panel <- panel[year %between% c(2000, 2025)]
if (!"milk_per_cow_kg" %in% names(panel)) {
  panel[, milk_per_cow_kg := fifelse(milk_production_lb > 0 & milk_cows_head > 0, milk_production_lb * lb_to_kg / milk_cows_head, NA_real_)]
}
panel[, log_per_cow := log(fifelse(milk_per_cow_kg > 0, milk_per_cow_kg, NA_real_))]
panel[, state_alpha := factor(state_alpha)]

years <- 2000:2025
rows <- list()
for (ii in seq_len(nrow(selected))) {
  exposure <- selected$exposure[ii]
  domain <- selected$domain_label[ii]
  if (!exposure %in% names(panel)) next
  message(sprintf("[GEE %s/%s] %s", ii, nrow(selected), exposure))
  d <- make_work_data(panel, exposure)
  if (nrow(d) < 300 || uniqueN(d$state_alpha) < 6 || uniqueN(d$x_z) <= 1) next
  vals <- fit_gee(d)
  rows[[length(rows) + 1]] <- data.table(
    variant = "geepack_gee",
    variant_label = "geepack GEE",
    domain_label = domain,
    exposure = exposure,
    year = years,
    beta_log_per_1sd_exposure = as.numeric(vals),
    status = fifelse(is.finite(as.numeric(vals)), "ok", "failed")
  )
}

new_gee <- rbindlist(rows, fill = TRUE)
new_gee[, abs_beta_log := abs(beta_log_per_1sd_exposure)]

old <- fread(out_long)
old <- old[variant != "geepack_gee"]
long <- rbindlist(list(old, new_gee), fill = TRUE)
fwrite(long, out_long)

summary <- long[status == "ok" & year %between% c(2000, 2024),
  .(median_abs_beta = median(abs(beta_log_per_1sd_exposure), na.rm = TRUE)),
  by = .(variant, variant_label, year, domain_label)
]
fwrite(summary, out_by_year)

n_exp <- long[status == "ok" & year %between% c(2000, 2024),
  .(n_exposures = uniqueN(exposure)),
  by = .(variant, variant_label, year, domain_label)
]
fwrite(n_exp, out_n)

s <- summary[variant == "geepack_gee"]
s[, domain_label := factor(domain_label, levels = rev(domain_order))]
stem <- "point2_milk_per_cow_expanded_nonredundant_geepack_gee_yearly_point2_style"
for (show_legend in c(TRUE, FALSE)) {
  suffix <- if (show_legend) "" else "_wo_legend"
  height <- if (show_legend) 3.9 else 3.45
  p <- ggplot(s, aes(x = year, y = median_abs_beta * 100, fill = domain_label)) +
    geom_area(position = "stack", linewidth = 0) +
    scale_fill_manual(values = colors[rev(domain_order)], breaks = domain_order) +
    scale_x_continuous(limits = c(2000, 2025), breaks = c(2000, 2005, 2010, 2015, 2020, 2025)) +
    labs(x = "Year", y = "Stacked median |β| (%)", fill = NULL) +
    theme_classic(base_family = "Arial", base_size = 9) +
    theme(
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "transparent", color = NA),
      legend.position = if (show_legend) "top" else "none",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.text = element_text(size = 9),
      axis.text = element_text(size = 9),
      axis.title = element_text(size = 9)
    ) +
    guides(fill = guide_legend(nrow = 3, byrow = FALSE))
  ggsave(file.path(fig, paste0(stem, suffix, ".svg")), p, width = 8.25, height = height, bg = "transparent")
}

cat("Updated geepack GEE in", out_long, "\n")
cat("\nGEE/domain counts\n")
print(n_exp[variant == "geepack_gee", .(min_n = min(n_exposures), max_n = max(n_exposures)), by = domain_label])
