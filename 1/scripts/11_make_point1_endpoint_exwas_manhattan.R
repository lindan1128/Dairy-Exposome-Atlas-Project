#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)][1])))
here <- normalizePath(file.path(script_dir, ".."))
tab <- file.path(here, "tables")
fig <- file.path(here, "figures")
dir.create(fig, recursive = TRUE, showWarnings = FALSE)

class_levels <- c(
  "Nature and climate",
  "Forage and pasture condition",
  "Chemical and pollution exposome",
  "Market and production-system",
  "Epidemic and infectious shocks"
)
class_pal <- c(
  "Nature and climate" = "#60BFA4",
  "Forage and pasture condition" = "#1E7A8D",
  "Chemical and pollution exposome" = "#c79fa8",
  "Market and production-system" = "#F06F26",
  "Epidemic and infectious shocks" = "#deab90"
)
domain_pal <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage condition" = "#1E7A8D",
  "Agricultural pesticides" = "#c79fa8",
  "Feed market" = "#fbc4ab",
  "Milk price / dairy market" = "#E47666",
  "Dairy market" = "#E47666",
  "Market demand" = "#f09d51",
  "Dairy scale" = "#fec89a",
  "Herd structure / scale" = "#fec89a",
  "Production system context" = "#ECBF51",
  "COVID" = "#d2b48c",
  "HPAI" = "#9d6b53"
)
domain_levels <- c(
  "Heat", "Cold", "Drought", "Severe weather", "Wildfire smoke",
  "Forage condition",
  "Air pollution", "Agricultural pesticides", "Industrial chemicals",
  "Feed market", "Milk price / dairy market", "Dairy market", "Market demand",
  "Dairy scale", "Herd structure / scale", "Production system context",
  "COVID", "HPAI"
)
excluded_domains <- c(
  "Drought", "Wildfire smoke", "Air pollution", "Industrial chemicals",
  "Production system context"
)
plot_domain_levels <- setdiff(domain_levels, excluded_domains)
domain_labels <- c(
  "Heat" = "Heat",
  "Cold" = "Cold",
  "Drought" = "Drought",
  "Severe weather" = "Severe weather",
  "Wildfire smoke" = "Wildfire smoke",
  "Forage condition" = "Forage",
  "Air pollution" = "Air pollution",
  "Agricultural pesticides" = "Pesticides",
  "Industrial chemicals" = "Industrial\nchemicals",
  "Feed market" = "Feed market",
  "Milk price / dairy market" = "Dairy market",
  "Dairy market" = "Dairy market",
  "Market demand" = "Market demand",
  "Dairy scale" = "Dairy scale",
  "Herd structure / scale" = "Dairy scale",
  "Production system context" = "Production\nsystem",
  "COVID" = "COVID-19",
  "HPAI" = "HPAI"
)
scope_levels <- c("total_26", "per_cow_26")
scope_labels <- c(
  "per_cow_26" = "Milk per cow\n26 states",
  "total_50" = "Total production\n50 states",
  "total_26" = "Total production\nsame 26 states"
)
tier_levels <- c("Bonferroni", "BY-FDR", "p<0.05", "n.s.")
tier_pal <- c(
  "Bonferroni" = "#e07a5f",
  "BY-FDR" = "#81b29a",
  "p<0.05" = "#f2cc8f",
  "n.s." = "#bdbdbd"
)
tier_legend_labels <- c(
  "Bonferroni" = "Bonferroni",
  "BY-FDR" = "BY-FDR",
  "p<0.05" = "P<0.05",
  "n.s." = "n.s."
)

clean_svg <- function(path) {
  x <- readLines(path, warn = FALSE)
  x <- gsub("stroke: #FFFFFF; fill: #FFFFFF;", "stroke: none; fill: none;", x, fixed = TRUE)
  x <- gsub("fill: #FFFFFF;", "fill: none;", x, fixed = TRUE)
  writeLines(x, path)
}

base_theme <- theme_classic(base_size = 9) +
  theme(
    text = element_text(size = 9, face = "plain", color = "#222222"),
    plot.title = element_text(size = 9, face = "plain"),
    plot.subtitle = element_text(size = 9, face = "plain", color = "#555555"),
    axis.title = element_text(size = 9, face = "plain"),
    axis.text = element_text(size = 9, color = "#222222"),
    strip.text = element_text(size = 9, face = "plain"),
    legend.title = element_text(size = 9, face = "plain"),
    legend.text = element_text(size = 9, face = "plain"),
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.key = element_rect(fill = "transparent", color = NA),
    axis.line = element_line(linewidth = 0.28, color = "#111111")
  )

d <- read_csv(file.path(tab, "point1_native_only_endpoint_exwas_associations.csv"), show_col_types = FALSE) %>%
  mutate(
    domain = case_when(
      domain == "Pandemic shock" & mechanistic_domain_en == "COVID" ~ "COVID",
      domain == "Pandemic shock" & mechanistic_domain_en == "HPAI" ~ "HPAI",
      TRUE ~ domain
    )
  ) %>%
  filter(
    domain %in% plot_domain_levels,
    window == "native",
    phenotype_scope %in% scope_levels,
    is.finite(plot_p)
  ) %>%
  mutate(
    source_class = factor(source_class, levels = class_levels),
    domain = factor(domain, levels = plot_domain_levels),
    phenotype_scope = factor(phenotype_scope, levels = scope_levels, labels = scope_labels[scope_levels]),
    native_signal_tier_plot = case_when(
      as.character(native_signal_tier) %in% c("Bonferroni", "BY-FDR") ~ as.character(native_signal_tier),
      plot_p < 0.05 ~ "p<0.05",
      TRUE ~ "n.s."
    ),
    native_signal_tier_plot = factor(native_signal_tier_plot, levels = tier_levels),
    neglogp = -log10(pmax(plot_p, 1e-300)),
    r2_plot = pmax(plot_incr_r2, 0),
    alpha_plot = ifelse(native_signal_tier_plot == "n.s.", 0.42, 0.95),
    stroke_plot = case_when(
      native_signal_tier_plot == "Bonferroni" ~ 0.55,
      native_signal_tier_plot == "BY-FDR" ~ 0.42,
      native_signal_tier_plot == "p<0.05" ~ 0.30,
      TRUE ~ 0.10
    )
  )

event_domains <- c("COVID", "HPAI")

avg_r2 <- d %>%
  filter(!(as.character(domain) %in% c("COVID", "HPAI")), is.finite(r2_plot), r2_plot > 0) %>%
  summarise(x = median(r2_plot, na.rm = TRUE)) %>%
  pull(x)
if (!is.finite(avg_r2) || length(avg_r2) == 0) avg_r2 <- 0.001
n_exposures <- n_distinct(d$exposure)

ord <- d %>%
  distinct(exposure, domain, source_class) %>%
  arrange(source_class, domain, exposure) %>%
  mutate(x = row_number())
d <- d %>% left_join(ord, by = c("exposure", "domain", "source_class"))
n_order <- max(ord$x, na.rm = TRUE)
d <- d %>% mutate(y_order = n_order + 1 - x)
ord <- ord %>% mutate(y_order = n_order + 1 - x)
bounds <- ord %>%
  group_by(source_class, domain) %>%
  summarise(ymin = min(y_order), ymax = max(y_order), .groups = "drop") %>%
  mutate(sep = ymin - 0.5, mid = (ymin + ymax) / 2)
class_bounds <- bounds %>%
  group_by(source_class) %>%
  summarise(ymin = min(ymin), ymax = max(ymax), mid = (ymin + ymax) / 2, .groups = "drop") %>%
  mutate(sep = ymin - 0.5)
domain_seps <- head(bounds$sep, -1)
class_seps <- head(class_bounds$sep, -1)
top_axis_y <- max(ord$y_order, na.rm = TRUE) + 0.5

event_d <- d %>%
  filter(as.character(domain) %in% event_domains) %>%
  group_by(phenotype_scope, domain, source_class) %>%
  summarise(
    exposure = paste0(first(as.character(domain)), " event summary"),
    plot_p = min(plot_p, na.rm = TRUE),
    plot_incr_r2 = first(plot_incr_r2),
    neglogp = -log10(pmax(plot_p, 1e-300)),
    r2_plot = ifelse(plot_p < 1, avg_r2, 0),
    y_order = median(y_order, na.rm = TRUE),
    native_signal_tier_plot = case_when(
      plot_p < 0.05 ~ "p<0.05",
      TRUE ~ "n.s."
    ),
    alpha_plot = ifelse(native_signal_tier_plot == "n.s.", 0.42, 0.95),
    stroke_plot = case_when(
      native_signal_tier_plot == "Bonferroni" ~ 0.55,
      native_signal_tier_plot == "BY-FDR" ~ 0.42,
      native_signal_tier_plot == "p<0.05" ~ 0.30,
      TRUE ~ 0.10
    ),
    .groups = "drop"
  ) %>%
  mutate(
    domain = factor(as.character(domain), levels = plot_domain_levels),
    source_class = factor(as.character(source_class), levels = class_levels),
    native_signal_tier_plot = factor(as.character(native_signal_tier_plot), levels = tier_levels)
  )

d_plot <- bind_rows(
  d %>% filter(!(as.character(domain) %in% event_domains)),
  event_d
)

thr <- d %>%
  group_by(phenotype_scope) %>%
  summarise(bonf = -log10(0.05 / sum(is.finite(plot_p))), .groups = "drop")
d_plot <- d_plot %>% left_join(thr, by = "phenotype_scope")
ymax <- max(16, ceiling(max(d_plot$neglogp, na.rm = TRUE) + 0.5))

p <- ggplot(d_plot, aes(neglogp, y_order)) +
  geom_hline(yintercept = domain_seps, linetype = "dotted", color = "#c4c4c4", linewidth = 0.18) +
  geom_hline(yintercept = class_seps, linetype = "solid", color = "#777777", linewidth = 0.32) +
  geom_hline(yintercept = top_axis_y, color = "#111111", linewidth = 0.28) +
  geom_vline(aes(xintercept = bonf), linetype = "dashed", color = "#777777", linewidth = 0.25) +
  geom_point(
    aes(fill = native_signal_tier_plot, size = r2_plot),
    shape = 21,
    color = "#111111",
    stroke = 0.16
  ) +
  facet_grid(. ~ phenotype_scope, scales = "fixed") +
  scale_fill_manual(values = tier_pal, labels = tier_legend_labels, name = "Signal tier", drop = FALSE) +
  scale_size_area(
    max_size = 7.6,
    breaks = c(0.001, 0.005, 0.01),
    labels = c("0.1%", "0.5%", "1%"),
    name = expression(Delta * R^2)
  ) +
  guides(
    fill = guide_legend(order = 1, nrow = 2, byrow = TRUE),
    size = guide_legend(order = 2, nrow = 1)
  ) +
  scale_y_continuous(
    breaks = bounds$mid,
    labels = domain_labels[as.character(bounds$domain)],
    expand = expansion(mult = 0.01)
  ) +
  scale_x_continuous(breaks = seq(0, ymax, by = 5), expand = expansion(mult = c(0, 0.05))) +
  coord_cartesian(xlim = c(0, ymax), clip = "off") +
  labs(
    title = "Native-only dairy panel-ExWAS Manhattan",
    subtitle = paste0("Clean ", n_exposures, "-variable pool; domains are ordered and coloured by source class. Event and slow-context domains use their preferred statistics."),
    x = expression(-log[10](italic(P))),
    y = NULL
  ) +
  base_theme +
  theme(
    axis.text.y = element_text(size = 9, angle = 0, hjust = 1, vjust = 0.5, lineheight = 0.88),
    strip.background = element_blank(),
    strip.text = element_blank(),
    panel.spacing.x = unit(1.4, "lines"),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.spacing = unit(0.1, "lines"),
    legend.spacing.y = unit(0.1, "lines"),
    legend.key.size = unit(0.28, "lines"),
    legend.text = element_text(size = 8),
    legend.title = element_text(size = 8)
  )

out <- file.path(fig, "main_point1_endpoint_exwas_manhattan.svg")
ggsave(out, p, width = 3.45, height = 8, units = "in", bg = "transparent")
clean_svg(out)

p_wo <- p + labs(title = NULL, subtitle = NULL, x = NULL, y = NULL) + theme(legend.position = "none")
out_wo <- file.path(fig, "main_point1_endpoint_exwas_manhattan_wo_legend.svg")
ggsave(out_wo, p_wo, width = 3.45, height = 8, units = "in", bg = "transparent")
clean_svg(out_wo)

message("Wrote native-only endpoint-aware ExWAS Manhattan figures.")
