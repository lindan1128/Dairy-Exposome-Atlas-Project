#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(maps)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE)
root <- normalizePath(file.path(dirname(script_path), "../../../../.."), mustWork = TRUE)
tab <- file.path(root, "analysis", "statistics", "6", "tables")
dir.create(tab, recursive = TRUE, showWarnings = FALSE)

state_lookup <- tibble(
  state_name = tolower(state.name),
  state_alpha = state.abb
)

m <- maps::map("state", plot = FALSE, fill = TRUE)

is_break <- is.na(m$x) | is.na(m$y)
breaks <- which(is_break)
starts <- c(1L, breaks + 1L)
ends <- c(breaks - 1L, length(m$x))
valid <- starts <= ends
starts <- starts[valid]
ends <- ends[valid]

rows <- list()
seg_id <- 0L
for (i in seq_along(starts)) {
  nm <- sub(":.*$", "", m$names[[i]])
  ab <- state_lookup$state_alpha[match(nm, state_lookup$state_name)]
  if (is.na(ab)) next
  xs <- m$x[starts[[i]]:ends[[i]]]
  ys <- m$y[starts[[i]]:ends[[i]]]
  ok <- is.finite(xs) & is.finite(ys)
  if (sum(ok) < 3) next
  seg_id <- seg_id + 1L
  rows[[length(rows) + 1L]] <- tibble(
    state_alpha = ab,
    state_name = nm,
    segment_id = seg_id,
    point_id = seq_len(sum(ok)),
    x = xs[ok],
    y = ys[ok]
  )
}

poly <- bind_rows(rows)

poly_area <- function(x, y) {
  abs(sum(x * dplyr::lead(y, default = y[1]) - y * dplyr::lead(x, default = x[1])) / 2)
}

seg_summary <- poly %>%
  group_by(state_alpha, segment_id) %>%
  summarise(
    area = poly_area(x, y),
    cx = mean(x),
    cy = mean(y),
    .groups = "drop"
  )

centroids <- seg_summary %>%
  group_by(state_alpha) %>%
  slice_max(area, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(state_alpha, x = cx, y = cy, largest_segment_area = area)

write_csv(poly, file.path(tab, "point6_us_state_map_polygons.csv"))
write_csv(centroids, file.path(tab, "point6_us_state_centroids.csv"))

cat("wrote", nrow(poly), "polygon points and", nrow(centroids), "centroids\n")
