#!/usr/bin/env Rscript
# Native-only semicircle chord for Point 1.
# Domain arcs are grouped and coloured by source class; ribbons encode effect
# direction among native-only p<0.05 + direction-stable signals.
suppressPackageStartupMessages({library(dplyr); library(readr); library(ggplot2)})

args <- commandArgs(trailingOnly = FALSE)
sd_ <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)][1])))
here <- normalizePath(file.path(sd_, ".."))
tab <- file.path(here, "tables")
fig <- file.path(here, "figures")

clean <- function(o) {
  x <- readLines(o, warn = FALSE)
  x <- gsub("stroke: #FFFFFF; fill: #FFFFFF;", "stroke: none; fill: none;", x, fixed = TRUE)
  x <- gsub("fill: #FFFFFF;", "fill: none;", x, fixed = TRUE)
  writeLines(x, o)
}

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
class_lab <- c(
  "Nature and climate" = "Nature + climate",
  "Forage and pasture condition" = "Forage + pasture",
  "Chemical and pollution exposome" = "Chemical",
  "Market and production-system" = "Market + system",
  "Epidemic and infectious shocks" = "Epidemic shocks"
)
ribbon_col <- c(
  "negative" = "#FFCC8D",
  "positive" = "#B1C99C",
  "zero" = "#cfcfcf"
)
domain_pal <- c(
  "Heat" = "#32a4b4",
  "Cold" = "#33c5b2",
  "Severe weather" = "#d5eada",
  "Forage condition" = "#1E7A8D",
  "Agricultural pesticides" = "#c79fa8",
  "Feed market" = "#fbc4ab",
  "Milk price / dairy market" = "#E47666",
  "Market demand" = "#f09d51",
  "Dairy scale" = "#fec89a",
  "Herd structure / scale" = "#fec89a",
  "Production system context" = "#ECBF51",
  "COVID" = "#d2b48c",
  "HPAI" = "#9d6b53"
)
ord <- c(
  "Heat", "Cold", "Drought", "Severe weather", "Wildfire smoke",
  "Forage condition",
  "Air pollution", "Agricultural pesticides", "Industrial chemicals",
  "Feed market", "Milk price / dairy market", "Market demand",
  "Dairy scale", "Herd structure / scale", "Production system context",
  "COVID", "HPAI"
)
excluded_domains <- c(
  "Drought", "Wildfire smoke", "Air pollution", "Industrial chemicals",
  "Production system context"
)
ord <- setdiff(ord, excluded_domains)
short_lab <- c(
  "Heat" = "Heat", "Cold" = "Cold", "Drought" = "Drought",
  "Severe weather" = "Severe weather", "Wildfire smoke" = "Wildfire smoke",
  "Forage condition" = "Forage", "Air pollution" = "Air pollution",
  "Agricultural pesticides" = "Pesticides", "Industrial chemicals" = "Industrial",
  "Feed market" = "Feed market", "Milk price / dairy market" = "Dairy market",
  "Market demand" = "Market demand", "Dairy scale" = "Dairy scale",
  "Herd structure / scale" = "Dairy scale",
  "Production system context" = "Production", "COVID" = "COVID-19", "HPAI" = "HPAI"
)

d_all <- read_csv(file.path(tab, "point1_native_only_endpoint_exwas_associations.csv"), show_col_types = FALSE) %>%
  mutate(
    domain = case_when(
      domain == "Pandemic shock" & mechanistic_domain_en == "COVID" ~ "COVID",
      domain == "Pandemic shock" & mechanistic_domain_en == "HPAI" ~ "HPAI",
      domain == "Dairy market" ~ "Milk price / dairy market",
      TRUE ~ domain
    )
  ) %>%
  filter(window == "native", domain %in% ord, source_class %in% class_levels) %>%
  mutate(
    domain = factor(domain, levels = ord),
    source_class = factor(source_class, levels = class_levels)
  )

pollution_audit_path <- file.path(tab, "point1_pollution_direction_confounding_audit.csv")
pollution_direction_ok <- tibble()
if (file.exists(pollution_audit_path)) {
  pollution_direction_ok <- read_csv(pollution_audit_path, show_col_types = FALSE) %>%
    filter(model == "plus_herd_market_system", status == "ok") %>%
    mutate(
      audited_effect_direction = case_when(
        is.finite(beta) & beta < 0 ~ "negative",
        is.finite(beta) & beta > 0 ~ "positive",
        TRUE ~ "zero"
      ),
      pollution_direction_ok = p < 0.05 & n_clusters >= 10
    ) %>%
    select(domain, exposure, audited_effect_direction, pollution_direction_ok)
}

R <- 1.0; r1 <- 1.13; rlab <- 1.24
th_hi <- pi * 0.998; th_lo <- pi * 0.002; gA <- pi * 0.014
bxL <- -1.0; bxR <- 1.0; gB <- 0.012
bez <- function(ax, ay, bx, by, n = 46) {
  t <- seq(0, 1, length.out = n); cy <- ay + (by - ay) * 0.5
  x <- (1 - t)^3 * ax + 3 * (1 - t)^2 * t * ax + 3 * (1 - t) * t^2 * bx + t^3 * bx
  y <- (1 - t)^3 * ay + 3 * (1 - t)^2 * t * cy + 3 * (1 - t) * t^2 * cy + t^3 * by
  data.frame(x = x, y = y)
}
arcpts <- function(a0, a1, rr, n = 24) {
  a <- seq(a0, a1, length.out = n)
  data.frame(x = rr * cos(a), y = rr * sin(a))
}

build_data <- function(scope) {
  d <- d_all %>% filter(phenotype_scope == scope)
  if (scope == "per_cow_26" && nrow(pollution_direction_ok) > 0) {
    d <- d %>%
      left_join(pollution_direction_ok, by = c("domain", "exposure")) %>%
      mutate(
        pollution_domain = as.character(domain) %in% c("Air pollution", "Industrial chemicals"),
        pollution_pass = !pollution_domain |
          (!is.na(pollution_direction_ok) &
             pollution_direction_ok &
             audited_effect_direction == effect_direction)
      )
  } else {
    d <- d %>% mutate(pollution_pass = TRUE)
  }
  agg_long <- d %>%
    mutate(
      sig_dir = as.numeric(plot_p) < 0.05 &
        as.numeric(n_specs_same_sign) >= 3 &
        pollution_pass &
        effect_direction %in% c("negative", "positive"),
      dir = ifelse(sig_dir, effect_direction, "zero"),
      ribbon_key = dir,
      ribbon_order = case_when(dir == "negative" ~ 1, dir == "positive" ~ 2, TRUE ~ 3)
    ) %>%
    count(source_class, domain, ribbon_key, ribbon_order, dir, name = "n")
  agg <- agg_long %>%
    group_by(source_class, domain) %>%
    summarise(
      total = sum(n),
      negative = sum(n[dir == "negative"]),
      positive = sum(n[dir == "positive"]),
      zero = sum(n[dir == "zero"]),
      .groups = "drop"
    ) %>%
    mutate(domain = factor(domain, levels = ord), source_class = factor(source_class, levels = class_levels)) %>%
    arrange(source_class, domain)
  N <- nrow(agg); S <- sum(agg$total)
  availA <- (th_hi - th_lo) - (N - 1) * gA
  availB <- (bxR - bxL) - (N - 1) * gB
  RIB <- list(); BLK <- list(); LAB <- list(); ki <- 1
  curA <- th_hi; curB <- bxL
  for (i in seq_len(N)) {
    spA <- availA * agg$total[i] / S; spB <- availB * agg$total[i] / S
    aHi <- curA; aLo <- curA - spA; bL <- curB; bR <- curB + spB
    dm <- as.character(agg$domain[i]); cl <- as.character(agg$source_class[i])
    BLK[[i]] <- rbind(arcpts(aHi, aLo, r1), arcpts(aLo, aHi, R)) %>%
      mutate(grp = paste0("blk", i), domain = dm, source_class = cl)
    amid <- (aLo + aHi) / 2
    LAB[[i]] <- data.frame(
      x = rlab * cos(amid), y = rlab * sin(amid),
      lab = short_lab[[dm]], lab2 = paste0(agg$negative[i], "-/", agg$positive[i], "+"),
      domain = dm, source_class = cl
    )
    ca <- aHi; cb <- bL
    sub <- agg_long %>% filter(as.character(domain) == dm) %>% arrange(ribbon_order)
    for (j in seq_len(nrow(sub))) {
      cnt <- sub$n[j]; if (cnt <= 0) next
      fa <- spA * cnt / agg$total[i]; fb <- spB * cnt / agg$total[i]
      aA <- ca; aB <- ca - fa; bl <- cb; br <- cb + fb; ca <- aB; cb <- br
      poly <- rbind(
        bez(bl, 0, R * cos(aA), R * sin(aA)),
        arcpts(aA, aB, R),
        bez(R * cos(aB), R * sin(aB), br, 0),
        data.frame(x = bl, y = 0)
      )
      poly$grp <- paste0("r", ki); poly$ribbon_key <- sub$ribbon_key[j]; RIB[[ki]] <- poly; ki <- ki + 1
    }
    curA <- aLo - gA; curB <- bR + gB
  }
  RB <- bind_rows(RIB); RB$ribbon_key <- factor(RB$ribbon_key, levels = names(ribbon_col))
  BK <- bind_rows(BLK); BK$source_class <- factor(BK$source_class, levels = class_levels)
  LB <- bind_rows(LAB); LB$source_class <- factor(LB$source_class, levels = class_levels)
  list(RB = RB, BK = BK, LB = LB)
}

build_signal_only_data <- function(scope) {
  chord_anchor_r <- 0.872
  bx_sig_L <- -0.82
  bx_sig_R <- 0.82
  d <- d_all %>% filter(phenotype_scope == scope)
  if (scope == "per_cow_26" && nrow(pollution_direction_ok) > 0) {
    d <- d %>%
      left_join(pollution_direction_ok, by = c("domain", "exposure")) %>%
      mutate(
        pollution_domain = as.character(domain) %in% c("Air pollution", "Industrial chemicals"),
        pollution_pass = !pollution_domain |
          (!is.na(pollution_direction_ok) &
             pollution_direction_ok &
             audited_effect_direction == effect_direction)
      )
  } else {
    d <- d %>% mutate(pollution_pass = TRUE)
  }
  agg_long <- d %>%
    mutate(
      sig_dir = as.numeric(plot_p) < 0.05 &
        as.numeric(n_specs_same_sign) >= 3 &
        pollution_pass &
        effect_direction %in% c("negative", "positive"),
      dir = ifelse(sig_dir, effect_direction, "zero"),
      ribbon_key = dir,
      ribbon_order = case_when(dir == "negative" ~ 1, dir == "positive" ~ 2, TRUE ~ 3)
    ) %>%
    count(source_class, domain, ribbon_key, ribbon_order, dir, name = "n")
  agg <- agg_long %>%
    group_by(source_class, domain) %>%
    summarise(
      total = sum(n),
      negative = sum(n[dir == "negative"]),
      positive = sum(n[dir == "positive"]),
      zero = sum(n[dir == "zero"]),
      signal_total = negative + positive,
      .groups = "drop"
    ) %>%
    mutate(domain = factor(domain, levels = ord), source_class = factor(source_class, levels = class_levels)) %>%
    arrange(source_class, domain)

  N <- nrow(agg); S <- sum(agg$total)
  availA <- (th_hi - th_lo) - (N - 1) * gA
  availB <- (bx_sig_R - bx_sig_L) - (N - 1) * gB
  RIB <- list(); BLK <- list(); LAB <- list(); BG <- list(); ki <- 1
  curA <- th_hi; curB <- bx_sig_L
  for (i in seq_len(N)) {
    spA <- availA * agg$total[i] / S; spB <- availB * agg$total[i] / S
    aHi <- curA; aLo <- curA - spA; bL <- curB; bR <- curB + spB
    dm <- as.character(agg$domain[i]); cl <- as.character(agg$source_class[i])
    BLK[[i]] <- rbind(arcpts(aHi, aLo, r1), arcpts(aLo, aHi, R)) %>%
      mutate(grp = paste0("blk", i), domain = dm, source_class = cl)
    # A soft inner pool band preserves tested-variable mass without drawing
    # every nonsignificant grey ribbon.
    BG[[i]] <- rbind(arcpts(aHi, aLo, R * 0.965), arcpts(aLo, aHi, R * 0.915)) %>%
      mutate(grp = paste0("bg", i), domain = dm, source_class = cl)
    amid <- (aLo + aHi) / 2
    LAB[[i]] <- data.frame(
      x = rlab * cos(amid), y = rlab * sin(amid),
      lab = short_lab[[dm]], lab2 = paste0(agg$negative[i], "-/", agg$positive[i], "+"),
      domain = dm, source_class = cl
    )
    sub <- agg_long %>%
      filter(as.character(domain) == dm, dir %in% c("negative", "positive")) %>%
      arrange(ribbon_order)
    if (nrow(sub) > 0 && agg$signal_total[i] > 0) {
      sig_spA_raw <- spA * max(agg$signal_total[i], 1) / agg$total[i]
      edge_pad <- min(gA * 0.55, sig_spA_raw * 0.22)
      ca <- aHi - edge_pad; cb <- bL
      sig_spA <- max(sig_spA_raw - 2 * edge_pad, sig_spA_raw * 0.56)
      for (j in seq_len(nrow(sub))) {
        cnt <- sub$n[j]; if (cnt <= 0) next
        fa <- sig_spA * cnt / agg$signal_total[i]
        fb <- spB * cnt / agg$signal_total[i]
        aA <- ca; aB <- ca - fa; bl <- cb; br <- cb + fb; ca <- aB; cb <- br
        poly <- rbind(
          bez(bl, 0, chord_anchor_r * cos(aA), chord_anchor_r * sin(aA)),
          arcpts(aA, aB, chord_anchor_r),
          bez(chord_anchor_r * cos(aB), chord_anchor_r * sin(aB), br, 0),
          data.frame(x = bl, y = 0)
        )
        poly$grp <- paste0("rs", ki); poly$ribbon_key <- sub$ribbon_key[j]
        RIB[[ki]] <- poly; ki <- ki + 1
      }
    }
    curA <- aLo - gA; curB <- bR + gB
  }
  RB <- bind_rows(RIB)
  if (nrow(RB) > 0) RB$ribbon_key <- factor(RB$ribbon_key, levels = names(ribbon_col))
  BK <- bind_rows(BLK); BK$source_class <- factor(BK$source_class, levels = class_levels)
  LB <- bind_rows(LAB); LB$source_class <- factor(LB$source_class, levels = class_levels)
  GB <- bind_rows(BG); GB$source_class <- factor(GB$source_class, levels = class_levels)
  list(RB = RB, BK = BK, LB = LB, GB = GB)
}

th <- theme_void(base_size = 9) +
  theme(
    text = element_text(size = 9, color = "#222222"),
    legend.position = "none",
    plot.title = element_text(size = 10, hjust = .5),
    plot.background = element_rect(fill = "transparent", color = NA),
    panel.background = element_rect(fill = "transparent", color = NA)
  )

mk <- function(scope, baseline_label) {
  dd <- build_data(scope); RB <- dd$RB; BK <- dd$BK; LB <- dd$LB
  ggplot() +
    geom_polygon(data = RB, aes(x, y, group = grp, fill = ribbon_key), color = NA, alpha = 1) +
    geom_polygon(data = BK, aes(x, y, group = grp, fill = domain), color = "white", linewidth = 0.3) +
    geom_rect(aes(xmin = bxL - 0.01, xmax = bxR + 0.01, ymin = -0.085, ymax = 0), fill = "#222222") +
    annotate("text", x = 0, y = -0.16, label = baseline_label, size = 3, color = "#222222") +
    geom_text(data = LB, aes(x = x, y = y, label = lab, color = domain), size = 2.35, fontface = "plain") +
    scale_fill_manual(values = c(class_pal, domain_pal, ribbon_col), drop = FALSE) +
    scale_color_manual(values = c(class_pal, domain_pal)) +
    coord_fixed(xlim = c(-1.38, 1.38), ylim = c(-0.3, 1.38), clip = "off") + th
}

circle_half <- function(d, side = c("left", "right"), scale = 0.96) {
  side <- match.arg(side)
  if (side == "left") {
    d %>% mutate(x_old = x, y_old = y, x = -y_old * scale, y = -x_old * scale) %>%
      select(-x_old, -y_old)
  } else {
    d %>% mutate(x_old = x, y_old = y, x = y_old * scale, y = -x_old * scale) %>%
      select(-x_old, -y_old)
  }
}

class_arc_data <- function(side = c("left", "right"), scale = 0.96, gap = 0, rr = 1.25, n = 44) {
  side <- match.arg(side)
  agg <- d_all %>%
    filter(phenotype_scope == "per_cow_26") %>%
    count(source_class, domain, name = "total") %>%
    mutate(domain = factor(domain, levels = ord), source_class = factor(source_class, levels = class_levels)) %>%
    arrange(source_class, domain)
  N <- nrow(agg); S <- sum(agg$total)
  availA <- (th_hi - th_lo) - (N - 1) * gA
  curA <- th_hi
  pieces <- list()
  for (i in seq_len(N)) {
    spA <- availA * agg$total[i] / S
    aHi <- curA; aLo <- curA - spA
    pieces[[i]] <- tibble(source_class = as.character(agg$source_class[i]), a_hi = aHi, a_lo = aLo)
    curA <- aLo - gA
  }
  cls <- bind_rows(pieces) %>%
    group_by(source_class) %>%
    summarise(a_hi = max(a_hi), a_lo = min(a_lo), .groups = "drop") %>%
    mutate(source_class = factor(source_class, levels = class_levels)) %>%
    arrange(source_class)
  arcs <- lapply(seq_len(nrow(cls)), function(i) {
    a <- seq(cls$a_hi[i], cls$a_lo[i], length.out = n)
    base <- tibble(
      x = rr * cos(a),
      y = rr * sin(a),
      source_class = as.character(cls$source_class[i]),
      grp = paste0(side, "_class_", i)
    )
    circle_half(base, side, scale) %>% mutate(x = x + ifelse(side == "left", -gap, gap))
  }) %>% bind_rows()
  labs <- lapply(seq_len(nrow(cls)), function(i) {
    amid <- (cls$a_hi[i] + cls$a_lo[i]) / 2
    base <- tibble(
      x = (rr + 0.075) * cos(amid),
      y = (rr + 0.075) * sin(amid),
      source_class = as.character(cls$source_class[i]),
      lab = class_lab[[as.character(cls$source_class[i])]]
    )
    circle_half(base, side, scale) %>%
      mutate(
        x = x + ifelse(side == "left", -gap, gap),
        hjust = ifelse(side == "left", 1, 0)
      )
  }) %>% bind_rows()
  list(arcs = arcs, labels = labs)
}

class_band_data <- function(side = c("left", "right"), scale = 0.96, gap = 0,
                            r_outer = 0.965, r_inner = 0.875, n = 44) {
  side <- match.arg(side)
  agg <- d_all %>%
    filter(phenotype_scope == "per_cow_26") %>%
    count(source_class, domain, name = "total") %>%
    mutate(domain = factor(domain, levels = ord), source_class = factor(source_class, levels = class_levels)) %>%
    arrange(source_class, domain)
  N <- nrow(agg); S <- sum(agg$total)
  availA <- (th_hi - th_lo) - (N - 1) * gA
  curA <- th_hi
  pieces <- list()
  for (i in seq_len(N)) {
    spA <- availA * agg$total[i] / S
    aHi <- curA; aLo <- curA - spA
    pieces[[i]] <- tibble(source_class = as.character(agg$source_class[i]), a_hi = aHi, a_lo = aLo)
    curA <- aLo - gA
  }
  cls <- bind_rows(pieces) %>%
    group_by(source_class) %>%
    summarise(a_hi = max(a_hi), a_lo = min(a_lo), .groups = "drop") %>%
    mutate(source_class = factor(source_class, levels = class_levels)) %>%
    arrange(source_class)
  lapply(seq_len(nrow(cls)), function(i) {
    a <- seq(cls$a_hi[i], cls$a_lo[i], length.out = n)
    base <- rbind(
      tibble(x = r_outer * cos(a), y = r_outer * sin(a)),
      tibble(x = r_inner * cos(rev(a)), y = r_inner * sin(rev(a)))
    ) %>%
      mutate(source_class = as.character(cls$source_class[i]), grp = paste0(side, "_class_band_", i))
    circle_half(base, side, scale) %>% mutate(x = x + ifelse(side == "left", -gap, gap))
  }) %>% bind_rows()
}

mk_paired <- function() {
  left <- build_data("total_26")
  right <- build_data("per_cow_26")
  circle_scale <- 0.96
  rb <- bind_rows(
    circle_half(left$RB, "left", circle_scale) %>% mutate(endpoint = "Total production"),
    circle_half(right$RB, "right", circle_scale) %>% mutate(endpoint = "Milk per cow")
  )
  bk <- bind_rows(
    circle_half(left$BK, "left", circle_scale) %>% mutate(endpoint = "Total production"),
    circle_half(right$BK, "right", circle_scale) %>% mutate(endpoint = "Milk per cow")
  )
  lb <- bind_rows(
    circle_half(left$LB, "left", circle_scale) %>% mutate(endpoint = "Total production"),
    circle_half(right$LB, "right", circle_scale) %>% mutate(endpoint = "Milk per cow")
  ) %>%
    mutate(
      hjust = ifelse(endpoint == "Total production", 1, 0),
      x = x + ifelse(endpoint == "Total production", -0.03, 0.03)
    )
  center_line <- tibble(
    x = c(0, 0),
    y = c(-0.97, 0.97)
  )
  titles <- tibble(
    x = c(-0.55, 0.55),
    y = 1.23,
    lab = c("Total production", "Milk per cow")
  )
  ggplot() +
    geom_polygon(data = rb, aes(x, y, group = interaction(endpoint, grp), fill = ribbon_key),
                 color = NA, alpha = 1) +
    geom_polygon(data = bk, aes(x, y, group = interaction(endpoint, grp), fill = domain),
                 color = "white", linewidth = 0.26) +
    geom_path(data = center_line, aes(x = x, y = y),
              inherit.aes = FALSE, color = "#222222", linewidth = 0.3, lineend = "round") +
    geom_text(data = lb, aes(x = x, y = y, label = lab, color = domain, hjust = hjust),
              size = 2.05, fontface = "plain") +
    geom_text(data = titles, aes(x = x, y = y, label = lab),
              inherit.aes = FALSE, size = 3.6, color = "#222222") +
    scale_fill_manual(values = c(class_pal, domain_pal, ribbon_col), drop = FALSE) +
    scale_color_manual(values = c(class_pal, domain_pal)) +
    coord_fixed(xlim = c(-1.32, 1.32), ylim = c(-1.20, 1.32), clip = "off") +
    th +
    theme(plot.title = element_text(size = 10, hjust = 0.5))
}

mk_paired_signal_only <- function(show_endpoint_titles = TRUE) {
  left <- build_signal_only_data("total_26")
  right <- build_signal_only_data("per_cow_26")
  circle_scale <- 0.96
  half_gap <- 0.075
  rb <- bind_rows(
    circle_half(left$RB, "left", circle_scale) %>% mutate(x = x - half_gap, endpoint = "Total production"),
    circle_half(right$RB, "right", circle_scale) %>% mutate(x = x + half_gap, endpoint = "Milk per cow")
  )
  bk <- bind_rows(
    circle_half(left$BK, "left", circle_scale) %>% mutate(x = x - half_gap, endpoint = "Total production"),
    circle_half(right$BK, "right", circle_scale) %>% mutate(x = x + half_gap, endpoint = "Milk per cow")
  )
  gb <- bind_rows(
    circle_half(left$GB, "left", circle_scale) %>% mutate(x = x - half_gap, endpoint = "Total production"),
    circle_half(right$GB, "right", circle_scale) %>% mutate(x = x + half_gap, endpoint = "Milk per cow")
  )
  lb <- bind_rows(
    circle_half(left$LB, "left", circle_scale) %>% mutate(x = x - half_gap, endpoint = "Total production"),
    circle_half(right$LB, "right", circle_scale) %>% mutate(x = x + half_gap, endpoint = "Milk per cow")
  ) %>%
    mutate(
      x_center = ifelse(endpoint == "Total production", -half_gap, half_gap),
      dx = x - x_center,
      theta = atan2(y, dx) * 180 / pi,
      flip = theta > 90 | theta < -90,
      angle = ifelse(flip, theta + 180, theta),
      hjust = ifelse(flip, 1, 0),
      x = x_center + dx * 1.000,
      y = y * 1.000
    )
  center_line <- tibble(x = c(-0.012, -0.012, 0.012, 0.012), y = c(-0.97, 0.97, -0.97, 0.97),
                        grp = c("left", "left", "right", "right"))
  class_band <- bind_rows(
    class_band_data("left", circle_scale, half_gap),
    class_band_data("right", circle_scale, half_gap)
  )
  titles <- tibble(
    x = c(-0.58, 0.58),
    y = 1.30,
    lab = c("Total production", "Milk per cow")
  )
  p <- ggplot() +
    geom_polygon(data = rb, aes(x, y, group = interaction(endpoint, grp), fill = ribbon_key),
                 color = "#111111", linewidth = 0.05, alpha = 1) +
    geom_polygon(data = class_band, aes(x, y, group = grp, fill = source_class),
                 inherit.aes = FALSE, color = "#111111", linewidth = 0.08, alpha = 1) +
    geom_polygon(data = bk, aes(x, y, group = interaction(endpoint, grp), fill = domain),
                 color = "#111111", linewidth = 0.08, alpha = 0.95) +
    geom_text(data = lb, aes(x = x, y = y, label = lab, color = domain, hjust = hjust, angle = angle),
              size = 3.17, fontface = "plain") +
    scale_fill_manual(values = c(class_pal, domain_pal, ribbon_col), drop = FALSE) +
    scale_color_manual(values = c(class_pal, domain_pal)) +
    coord_fixed(xlim = c(-1.54, 1.54), ylim = c(-1.54, 1.42), clip = "off") +
    th +
    theme(
      plot.title = element_text(size = 10, hjust = 0.5),
      plot.margin = margin(8, 10, 14, 10)
    )
  if (show_endpoint_titles) {
    p <- p +
      geom_text(data = titles, aes(x = x, y = y, label = lab),
                inherit.aes = FALSE, size = 3.6, color = "#222222")
  }
  p
}

pp <- mk_paired() +
  labs(title = "Native-only exposome chord: total production vs milk per cow")
outp <- file.path(fig, "main_point1_total26_percow_paired_exposome_chord.svg")
ggsave(outp, pp, width = 9, height = 4.7, units = "in", bg = "transparent"); clean(outp)
ppw <- mk_paired() + theme(plot.title = element_blank())
outpw <- file.path(fig, "main_point1_total26_percow_paired_exposome_chord_wo_legend.svg")
ggsave(outpw, ppw, width = 9, height = 4.7, units = "in", bg = "transparent"); clean(outpw)

pps <- mk_paired_signal_only() +
  labs(title = "Native-only exposome chord: significant directions over tested pool")
outs <- file.path(fig, "main_point1_total26_percow_paired_exposome_chord_signal_only.svg")
ggsave(outs, pps, width = 7.75, height = 4.10, units = "in", bg = "transparent"); clean(outs)
ppsw <- mk_paired_signal_only(show_endpoint_titles = FALSE) + theme(plot.title = element_blank())
outsw <- file.path(fig, "main_point1_total26_percow_paired_exposome_chord_signal_only_wo_legend.svg")
ggsave(outsw, ppsw, width = 7.75, height = 4.10, units = "in", bg = "transparent"); clean(outsw)
message("wrote native-only semicircle chord figures")
