suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(maps)
  library(scales)
  library(svglite)
})

cmd <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", cmd[grep("--file=", cmd)][1])
point4_dir <- normalizePath(file.path(dirname(script_path), ".."), mustWork = FALSE)
base_dir <- dirname(point4_dir)

dir.create(file.path(point4_dir, "tables"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(point4_dir, "figures"), recursive = TRUE, showWarnings = FALSE)

domain_order <- c("Heat", "Cold", "Severe weather", "Forage", "Feed market", "Dairy market", "Market demand")
domain_cols <- c(
  "Heat" = "#34a6b5",
  "Cold" = "#39c7b4",
  "Severe weather" = "#d8efdc",
  "Forage" = "#1f7f91",
  "Feed market" = "#ffc5ad",
  "Dairy market" = "#e87365",
  "Market demand" = "#f7a34d"
)

excluded_priority_exposures <- c(
  "market_log_population_total",
  "storm_event_types"
)

state_lookup <- tibble(
  state_alpha = state.abb,
  state_name = tolower(state.name)
) %>%
  bind_rows(tibble(state_alpha = c("AK", "HI"), state_name = c("alaska", "hawaii"))) %>%
  distinct(state_alpha, .keep_all = TRUE)

minmax01 <- function(x) {
  if (all(is.na(x))) return(rep(NA_real_, length(x)))
  r <- range(x, na.rm = TRUE)
  if (!is.finite(r[1]) || !is.finite(r[2]) || diff(r) == 0) return(ifelse(is.na(x), NA_real_, 0.5))
  (x - r[1]) / diff(r)
}

pctl <- function(x) {
  if (all(is.na(x))) return(rep(NA_real_, length(x)))
  percent_rank(x)
}

top20_flag <- function(x) {
  ok <- is.finite(x)
  out <- rep(FALSE, length(x))
  if (!any(ok)) return(out)
  cutoff <- ceiling(0.2 * sum(ok))
  out[ok] <- min_rank(desc(x[ok])) <= cutoff
  out
}

read_csv <- function(path) {
  read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

z <- function(x) {
  x <- as.numeric(x)
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 1e-12) return(rep(NA_real_, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

weighted_r2 <- function(y, yhat, w) {
  ok <- is.finite(y) & is.finite(yhat) & is.finite(w) & w > 0
  if (sum(ok) < 10) return(NA_real_)
  y <- y[ok]
  yhat <- yhat[ok]
  w <- w[ok]
  ybar <- weighted.mean(y, w)
  sst <- sum(w * (y - ybar)^2)
  if (!is.finite(sst) || sst <= 1e-12) return(NA_real_)
  1 - sum(w * (y - yhat)^2) / sst
}

adj_r2_from_r2 <- function(r2, n, p) {
  if (!is.finite(r2) || !is.finite(n) || !is.finite(p) || n <= p + 1) return(NA_real_)
  1 - (1 - r2) * (n - 1) / (n - p - 1)
}

fit_state_exposure <- function(d) {
  d <- d %>%
    filter(
      is.finite(log_per_cow),
      is.finite(x_raw),
      is.finite(milk_cows_head),
      milk_cows_head > 0
    ) %>%
    mutate(
      x_z = z(x_raw),
      y_z = z(log_per_cow),
      year_f = factor(year),
      month_f = factor(month)
    ) %>%
    filter(is.finite(x_z), is.finite(y_z))
  if (nrow(d) < 48 || length(unique(d$year)) < 6 || length(unique(d$x_z)) < 3) {
    return(tibble(
      beta_std_2015_2024 = NA_real_,
      abs_beta_std_2015_2024 = NA_real_,
      adjusted_incremental_r2_2015_2024 = NA_real_,
      full_adjusted_r2_2015_2024 = NA_real_,
      n_obs = nrow(d),
      status = "too_few"
    ))
  }
  full <- tryCatch(
    lm(y_z ~ x_z + year_f + month_f, data = d, weights = milk_cows_head),
    error = function(e) NULL
  )
  reduced <- tryCatch(
    lm(y_z ~ year_f + month_f, data = d, weights = milk_cows_head),
    error = function(e) NULL
  )
  if (is.null(full) || is.null(reduced) || !"x_z" %in% names(coef(full))) {
    return(tibble(
      beta_std_2015_2024 = NA_real_,
      abs_beta_std_2015_2024 = NA_real_,
      adjusted_incremental_r2_2015_2024 = NA_real_,
      full_adjusted_r2_2015_2024 = NA_real_,
      n_obs = nrow(d),
      status = "failed"
    ))
  }
  full_r2 <- weighted_r2(d$y_z, fitted(full), d$milk_cows_head)
  reduced_r2 <- weighted_r2(d$y_z, fitted(reduced), d$milk_cows_head)
  full_p <- length(coef(full))
  reduced_p <- length(coef(reduced))
  full_adj <- adj_r2_from_r2(full_r2, nrow(d), full_p)
  reduced_adj <- adj_r2_from_r2(reduced_r2, nrow(d), reduced_p)
  beta <- unname(coef(full)[["x_z"]])
  tibble(
    beta_std_2015_2024 = beta,
    abs_beta_std_2015_2024 = abs(beta),
    adjusted_incremental_r2_2015_2024 = full_adj - reduced_adj,
    full_adjusted_r2_2015_2024 = full_adj,
    n_obs = nrow(d),
    status = "ok"
  )
}

point2 <- file.path(base_dir, "2")
point3 <- file.path(base_dir, "3")

selection <- read_csv(file.path(point2, "tables", "point2_beta_std_two_stage_three_metric_by_variable.csv")) %>%
  filter(class_label %in% domain_order) %>%
  filter(!exposure %in% excluded_priority_exposures) %>%
  distinct(class_label, exposure, .keep_all = TRUE)

panel <- read_csv(file.path(point2, "tables", "point2_expanded_sensitivity_panel_for_r.csv")) %>%
  filter(year >= 2015, year <= 2024)
panel <- panel %>%
  mutate(
    log_per_cow = log(ifelse(milk_per_cow_kg > 0, milk_per_cow_kg, NA_real_))
  )

available_vars <- intersect(selection$exposure, names(panel))
missing_vars <- setdiff(selection$exposure, names(panel))

variable_dictionary <- read_csv(file.path(point2, "tables", "point2_2015_2024_percow_clean_curated_7class_screen.csv")) %>%
  select(exposure, exposure_zh, definition_en) %>%
  distinct(exposure, .keep_all = TRUE)

long_exposure <- panel %>%
  select(state_alpha, year, month, all_of(available_vars)) %>%
  pivot_longer(
    cols = all_of(available_vars),
    names_to = "exposure",
    values_to = "value"
  ) %>%
  left_join(selection, by = "exposure") %>%
  filter(!is.na(class_label))

message("Fitting state-specific 2015-2024 exposure models...")
state_variable_models <- lapply(seq_len(nrow(selection)), function(i) {
  exposure_i <- selection$exposure[[i]]
  domain_i <- selection$class_label[[i]]
  if (!exposure_i %in% names(panel)) return(NULL)
  d <- panel %>%
    select(state_alpha, year, month, milk_cows_head, log_per_cow, x_raw = all_of(exposure_i))
  d %>%
    group_by(state_alpha) %>%
    group_modify(~ fit_state_exposure(.x)) %>%
    ungroup() %>%
    mutate(class_label = domain_i, exposure = exposure_i)
}) %>%
  bind_rows() %>%
  relocate(state_alpha, class_label, exposure)

write.csv(
  state_variable_models,
  file.path(point4_dir, "tables", "main_point4_state_variable_2015_2024_beta_r2.csv"),
  row.names = FALSE
)

state_sensitivity_metrics <- state_variable_models %>%
  filter(status == "ok") %>%
  group_by(state_alpha, class_label) %>%
  summarise(
    state_median_abs_beta_std_2015_2024 = median(abs_beta_std_2015_2024, na.rm = TRUE),
    state_median_adjusted_incremental_r2_2015_2024 = median(adjusted_incremental_r2_2015_2024, na.rm = TRUE),
    n_state_sensitivity_variables = n_distinct(exposure),
    .groups = "drop"
  ) %>%
  group_by(class_label) %>%
  mutate(
    beta_std_percentile_within_class = pctl(state_median_abs_beta_std_2015_2024),
    adjusted_incremental_r2_percentile_within_class = pctl(state_median_adjusted_incremental_r2_2015_2024),
    beta_std_top20 = top20_flag(state_median_abs_beta_std_2015_2024),
    adjusted_incremental_r2_top20 = top20_flag(state_median_adjusted_incremental_r2_2015_2024)
  ) %>%
  ungroup()

shap <- lapply(1:9, function(horizon) {
  read_csv(file.path(point3, "tables", paste0("point3_point4aligned_h", horizon, "_shap_state_month_class.csv"))) %>%
    mutate(horizon_months = horizon)
}) %>%
  bind_rows() %>%
  filter(class_label %in% domain_order) %>%
  mutate(class_abs_shap = as.numeric(class_abs_signed_shap)) %>%
  group_by(state_alpha, region, year, month, horizon_months) %>%
  mutate(total_abs_shap = sum(class_abs_shap, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(class_shap_share = ifelse(total_abs_shap > 0, class_abs_shap / total_abs_shap, NA_real_)) %>%
  group_by(state_alpha, region, class_label) %>%
  summarise(
    state_forecast_contribution = mean(class_shap_share, na.rm = TRUE),
    n_shap_records = sum(is.finite(class_shap_share)),
    .groups = "drop"
  ) %>%
  group_by(class_label) %>%
  mutate(
    forecast_percentile_within_class = pctl(state_forecast_contribution),
    forecast_top20 = top20_flag(state_forecast_contribution)
  ) %>%
  ungroup()

state_regions <- read_csv(file.path(point3, "tables", "point3_state_region_lookup.csv")) %>%
  distinct(state_alpha, region)

priority_grid <- expand_grid(
  state_alpha = sort(unique(panel$state_alpha)),
  class_label = domain_order
) %>%
  left_join(state_regions, by = "state_alpha")

priority <- priority_grid %>%
  left_join(state_sensitivity_metrics, by = c("state_alpha", "class_label")) %>%
  left_join(shap, by = c("state_alpha", "class_label")) %>%
  mutate(
    class_label = factor(class_label, levels = domain_order),
    region = coalesce(region.x, region.y, "Unassigned"),
    region.x = NULL,
    region.y = NULL,
    beta_std_top20 = coalesce(beta_std_top20, FALSE),
    adjusted_incremental_r2_top20 = coalesce(adjusted_incremental_r2_top20, FALSE),
    forecast_top20 = coalesce(forecast_top20, FALSE),
    top20_overlap_3part = as.integer(beta_std_top20) +
      as.integer(adjusted_incremental_r2_top20) +
      as.integer(forecast_top20),
    top20_overlap_4flag = top20_overlap_3part,
    priority_class = case_when(
      beta_std_top20 & adjusted_incremental_r2_top20 & forecast_top20 ~ "High priority",
      beta_std_top20 & adjusted_incremental_r2_top20 & !forecast_top20 ~ "Association-only",
      beta_std_top20 & forecast_top20 & !adjusted_incremental_r2_top20 ~ "Beta-forecast",
      adjusted_incremental_r2_top20 & forecast_top20 & !beta_std_top20 ~ "R2-forecast",
      top20_overlap_3part == 1 ~ "Watchlist",
      TRUE ~ "Low"
    ),
    consensus_score = rowMeans(
      cbind(beta_std_percentile_within_class, adjusted_incremental_r2_percentile_within_class, forecast_percentile_within_class),
      na.rm = TRUE
    ),
    beta_std_index = beta_std_percentile_within_class,
    adjusted_incremental_r2_index = adjusted_incremental_r2_percentile_within_class,
    forecast_index = forecast_percentile_within_class,
    priority_index_sum = rowSums(
      cbind(beta_std_index, adjusted_incremental_r2_index, forecast_index),
      na.rm = TRUE
    ),
    priority_index_mean = rowMeans(
      cbind(beta_std_index, adjusted_incremental_r2_index, forecast_index),
      na.rm = TRUE
    ),
    priority_index_n_components = rowSums(is.finite(cbind(beta_std_index, adjusted_incremental_r2_index, forecast_index))),
    equal_weighted_priority_raw = rowMeans(
      cbind(beta_std_index, adjusted_incremental_r2_index, forecast_index),
      na.rm = TRUE
    )
  ) %>%
  group_by(state_alpha) %>%
  mutate(
    equal_weighted_priority_share = equal_weighted_priority_raw / sum(equal_weighted_priority_raw, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  arrange(state_alpha, desc(equal_weighted_priority_share), desc(priority_index_sum), desc(top20_overlap_3part), desc(consensus_score))

write.csv(
  priority,
  file.path(point4_dir, "tables", "main_point4_state_class_priority_top20_overlap.csv"),
  row.names = FALSE
)

top_by_state <- priority %>%
  group_by(state_alpha) %>%
  slice_max(order_by = equal_weighted_priority_share, n = 1, with_ties = TRUE) %>%
  slice_max(order_by = top20_overlap_3part, n = 1, with_ties = TRUE) %>%
  slice_max(order_by = consensus_score, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  left_join(state_lookup, by = "state_alpha")

write.csv(
  top_by_state,
  file.path(point4_dir, "tables", "main_point4_state_top_priority_class.csv"),
  row.names = FALSE
)

variable_evidence <- state_variable_models %>%
  filter(status == "ok") %>%
  left_join(variable_dictionary, by = "exposure") %>%
  group_by(state_alpha, class_label) %>%
  mutate(
    beta_rank_in_state_class = min_rank(desc(abs_beta_std_2015_2024)),
    adjusted_incremental_r2_rank_in_state_class = min_rank(desc(adjusted_incremental_r2_2015_2024))
  ) %>%
  ungroup()

priority_variable_evidence <- priority %>%
  filter(top20_overlap_3part >= 2 | paste(state_alpha, class_label) %in% paste(top_by_state$state_alpha, top_by_state$class_label)) %>%
  select(state_alpha, region, class_label, priority_class, top20_overlap_3part, top20_overlap_4flag) %>%
  left_join(variable_evidence, by = c("state_alpha", "class_label")) %>%
  mutate(
    evidence_role = case_when(
      beta_rank_in_state_class == 1 ~ "highest state beta",
      adjusted_incremental_r2_rank_in_state_class == 1 ~ "highest state adjusted incremental R2",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(evidence_role)) %>%
  arrange(state_alpha, class_label, evidence_role, exposure)

write.csv(
  priority_variable_evidence,
  file.path(point4_dir, "tables", "main_point4_state_class_priority_variable_evidence.csv"),
  row.names = FALSE
)

summary_domain_region <- priority %>%
  group_by(region, class_label) %>%
  summarise(
    n_states = n_distinct(state_alpha),
    n_high_priority = sum(priority_class == "High priority", na.rm = TRUE),
    n_two_or_more_flags = sum(top20_overlap_3part >= 2, na.rm = TRUE),
    median_priority_index_sum = median(priority_index_sum, na.rm = TRUE),
    q25_priority_index_sum = quantile(priority_index_sum, 0.25, na.rm = TRUE),
    q75_priority_index_sum = quantile(priority_index_sum, 0.75, na.rm = TRUE),
    median_consensus_score = median(consensus_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(region, desc(median_priority_index_sum), desc(n_high_priority), desc(n_two_or_more_flags))

write.csv(
  summary_domain_region,
  file.path(point4_dir, "tables", "main_point4_region_class_priority_summary.csv"),
  row.names = FALSE
)

audit <- tibble(
  item = c(
    "available_selected_variables",
    "missing_selected_variables",
    "domains",
    "states_in_priority_table",
    "top20_rule"
  ),
  value = c(
    length(available_vars),
    length(missing_vars),
    paste(domain_order, collapse = "; "),
    n_distinct(priority$state_alpha),
    "Within-class percentiles for state median |standardized beta|, state median adjusted incremental R2 and 1-9 month forecast SHAP share; equal-weighted then normalized within state."
  )
)

write.csv(
  audit,
  file.path(point4_dir, "tables", "main_point4_priority_input_audit.csv"),
  row.names = FALSE
)

if (length(missing_vars) > 0) {
  write.csv(
    tibble(exposure = missing_vars),
    file.path(point4_dir, "tables", "main_point4_priority_missing_selected_exposures.csv"),
    row.names = FALSE
  )
}

plot_theme <- theme_classic(base_size = 16.25) +
  theme(
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    strip.background = element_blank(),
    strip.text = element_text(color = "black", face = "plain"),
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.background = element_rect(fill = NA, color = NA),
    panel.background = element_rect(fill = NA, color = NA),
    legend.background = element_rect(fill = NA, color = NA),
    legend.box.background = element_rect(fill = NA, color = NA)
  )

domain_distribution <- priority %>%
  filter(is.finite(priority_index_sum)) %>%
  mutate(class_label = factor(class_label, levels = domain_order))

domain_index_summary <- domain_distribution %>%
  group_by(class_label) %>%
  summarise(
    n_state_classes = n(),
    median_priority_index = median(priority_index_sum, na.rm = TRUE),
    q25_priority_index = quantile(priority_index_sum, 0.25, na.rm = TRUE),
    q75_priority_index = quantile(priority_index_sum, 0.75, na.rm = TRUE),
    min_priority_index = min(priority_index_sum, na.rm = TRUE),
    max_priority_index = max(priority_index_sum, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(median_priority_index))

write.csv(
  domain_index_summary,
  file.path(point4_dir, "tables", "main_point4_class_priority_index_distribution_summary.csv"),
  row.names = FALSE
)

state_index_summary <- priority %>%
  filter(is.finite(priority_index_sum)) %>%
  group_by(state_alpha, region) %>%
  summarise(
    n_classes = n(),
    top_class = as.character(class_label[which.max(priority_index_sum)]),
    top_priority_index = max(priority_index_sum, na.rm = TRUE),
    second_priority_index = sort(priority_index_sum, decreasing = TRUE)[pmin(2, n())],
    priority_gap_top_minus_second = top_priority_index - second_priority_index,
    median_priority_index = median(priority_index_sum, na.rm = TRUE),
    q25_priority_index = quantile(priority_index_sum, 0.25, na.rm = TRUE),
    q75_priority_index = quantile(priority_index_sum, 0.75, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(region, desc(top_priority_index))

write.csv(
  state_index_summary,
  file.path(point4_dir, "tables", "main_point4_state_priority_index_distribution_summary.csv"),
  row.names = FALSE
)

plot_shap_h1_h9 <- lapply(1:9, function(horizon) {
  read_csv(file.path(point3, "tables", paste0("point3_point4aligned_h", horizon, "_shap_state_month_class.csv"))) %>%
    mutate(horizon_months = horizon)
}) %>%
  bind_rows() %>%
  filter(class_label %in% domain_order) %>%
  mutate(class_abs_shap = as.numeric(class_abs_signed_shap)) %>%
  group_by(state_alpha, region, year, month, horizon_months) %>%
  mutate(total_abs_shap = sum(class_abs_shap, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(class_shap_share = ifelse(total_abs_shap > 0, class_abs_shap / total_abs_shap, NA_real_)) %>%
  group_by(state_alpha, class_label) %>%
  summarise(
    plot_forecast_contribution = mean(class_shap_share, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(class_label) %>%
  mutate(plot_forecast_index = pctl(plot_forecast_contribution)) %>%
  ungroup()

state_distribution <- priority %>%
  left_join(plot_shap_h1_h9, by = c("state_alpha", "class_label")) %>%
  mutate(forecast_index = coalesce(plot_forecast_index, forecast_index)) %>%
  filter(is.finite(priority_index_sum)) %>%
  mutate(
    region = factor(region, levels = c("South", "West", "Midwest", "Northeast", "Unassigned")),
    state_alpha = factor(
      state_alpha,
      levels = state_index_summary %>%
        arrange(region, desc(top_priority_index), state_alpha) %>%
        pull(state_alpha)
    ),
    class_label = factor(class_label, levels = domain_order)
  )

state_distribution_long <- state_distribution %>%
  select(
    state_alpha,
    region,
    class_label,
    beta_std_index,
    adjusted_incremental_r2_index,
    forecast_index
  ) %>%
  pivot_longer(
    cols = c(beta_std_index, adjusted_incremental_r2_index, forecast_index),
    names_to = "index_type",
    values_to = "index_value"
  ) %>%
  mutate(
    index_type = factor(
      index_type,
      levels = c("beta_std_index", "adjusted_incremental_r2_index", "forecast_index"),
      labels = c("Standardized β index", "Adjusted incremental R² index", "Forecast SHAP index")
    )
  ) %>%
  filter(is.finite(index_value))

state_distribution_long <- state_distribution_long %>%
  group_by(state_alpha, index_type) %>%
  mutate(index_share = index_value / sum(index_value, na.rm = TRUE)) %>%
  ungroup() %>%
  filter(!state_alpha %in% c("AK", "HI")) %>%
  filter(is.finite(index_share))

p_state_dist <- ggplot(state_distribution_long, aes(state_alpha, index_share, fill = class_label)) +
  geom_col(color = "black", linewidth = 0.12, width = 0.82) +
  facet_grid(index_type ~ region, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = domain_cols, drop = FALSE) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  coord_cartesian(ylim = c(0, 1), clip = "off") +
  labs(x = NULL, y = NULL) +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    panel.spacing.x = unit(0.35, "lines")
  )

ggsave(
  file.path(point4_dir, "figures", "supp_point4_state_class_priority_index_distribution.svg"),
  p_state_dist,
  width = 12.0,
  height = 9.4,
  bg = "transparent"
)

state_combined_distribution <- state_distribution_long %>%
  group_by(state_alpha, region, class_label) %>%
  summarise(
    combined_index_share = mean(index_share, na.rm = TRUE),
    n_indices = n_distinct(index_type),
    .groups = "drop"
  ) %>%
  group_by(state_alpha) %>%
  mutate(combined_index_share = combined_index_share / sum(combined_index_share, na.rm = TRUE)) %>%
  ungroup()

state_cluster_order <- state_combined_distribution %>%
  select(state_alpha, region, class_label, combined_index_share) %>%
  pivot_wider(
    names_from = class_label,
    values_from = combined_index_share,
    values_fill = 0
  ) %>%
  group_by(region) %>%
  group_modify(function(.x, .g) {
    mat <- as.matrix(.x[, domain_order, drop = FALSE])
    rownames(mat) <- as.character(.x$state_alpha)
    if (nrow(mat) >= 3) {
      ordered_states <- rownames(mat)[hclust(dist(mat), method = "average")$order]
    } else {
      ordered_states <- rownames(mat)
    }
    tibble(state_alpha = ordered_states, cluster_order = seq_along(ordered_states))
  }) %>%
  ungroup() %>%
  mutate(region = factor(region, levels = c("South", "West", "Midwest", "Northeast", "Unassigned"))) %>%
  arrange(region, cluster_order)

write.csv(
  state_cluster_order,
  file.path(point4_dir, "tables", "main_point4_state_equal_weighted_priority_cluster_order.csv"),
  row.names = FALSE
)

state_combined_distribution <- state_combined_distribution %>%
  mutate(
    state_alpha = factor(state_alpha, levels = state_cluster_order$state_alpha),
    region = factor(region, levels = c("South", "West", "Midwest", "Northeast", "Unassigned"))
  )

p_state_combined <- ggplot(state_combined_distribution, aes(state_alpha, combined_index_share, fill = class_label)) +
  geom_col(color = "black", linewidth = 0.12, width = 0.82) +
  facet_grid(. ~ region, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = domain_cols, drop = FALSE) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  coord_cartesian(ylim = c(0, 1), clip = "off") +
  labs(x = NULL, y = "Equal-weighted priority share") +
  plot_theme +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    panel.spacing.x = unit(0.35, "lines")
  )

ggsave(
  file.path(point4_dir, "figures", "main_point4_state_class_equal_weighted_priority_index_distribution.svg"),
  p_state_combined,
  width = 12.0,
  height = 3.8,
  bg = "transparent"
)

message("Done. Wrote point4 priority tables and SVG figures.")
