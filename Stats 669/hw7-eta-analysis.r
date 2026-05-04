## ============================================================
## Stats 669 – Homework 7: eta_j analysis on elevation data
## Parts (a)–(d): 50x50 subregion eta_j for j = 1,2,3,4
## ============================================================
library(terra)
library(ggplot2)
library(magrittr)
library(dplyr)
library(tidyr)
# ---------------------------------------------------------------
# 0.  LOAD YOUR RASTER  (edit path/filename to match your dataset)
# ---------------------------------------------------------------

ourdir <- "Stats 669/img/hw8"
dir.create(ourdir, showWarnings = FALSE, recursive = TRUE)

raster_data <- rast("Stats 669/data/USGS_13_n37w118_20260112.tif")

nr <- nrow(raster_data)
nc <- ncol(raster_data)

# center indices
center_row <- nr %/% 2
center_col <- nc %/% 3

# row/col window: 2000 x 2000
row_start <- center_row - 999
row_end   <- center_row + 1001   # inclusive
col_start <- center_col - 999
col_end   <- center_col + 1000   # inclusive

# convert those indices to coordinates for ext(xmin, xmax, ymin, ymax)
dem_2k <- crop(
  raster_data,
  ext(
    xFromCol(raster_data, col_start),
    xFromCol(raster_data, col_end + 1),
    yFromRow(raster_data, row_end + 1),
    yFromRow(raster_data, row_start)
  )
)

Z <- as.matrix(dem_2k, wide = TRUE)
dim(Z)  # should now be 2000 2000

# ---------------------------------------------------------------
# 1.  CORE FUNCTION: square increment  Delta_{j,j} Z at (r, c)
#     = Z[r,c] + Z[r+j, c+j] - Z[r+j, c] - Z[r, c+j]
# ---------------------------------------------------------------

# Vectorised eta_j for one 50x50 subgrid (already extracted)
eta_j_subgrid <- function(sub, j) {
  n <- nrow(sub)   # should be 50
  # interior index ranges (1-based, using the notation in eq. 1)
  # k = j+1 ... n1-j  =>  R index j+1 ... n-j
  r  <- (j + 1):(n - j)
  c  <- (j + 1):(n - j)

  # vectorised square increment
  inc <- sub[r,      c     ] +
         sub[r + j,  c + j ] -
         sub[r + j,  c     ] -
         sub[r,      c + j ]

  mean(inc^2, na.rm = TRUE)
}

# ---------------------------------------------------------------
# 2.  PART (a):  compute eta_j for j=1,2,3,4 on all 1600 subregions
# ---------------------------------------------------------------

sub_size  <- 50
n_row_sub <- 2000 / sub_size   # 40
n_col_sub <- 2000 / sub_size   # 40
j_vals    <- 1:4

# Storage:  1600 rows  x  4 columns
#           plus row/col indices for mapping
results <- expand.grid(
  row_blk = 1:n_row_sub,
  col_blk = 1:n_col_sub
)
eta_mat <- matrix(NA_real_, nrow = nrow(results), ncol = length(j_vals))
colnames(eta_mat) <- paste0("eta", j_vals)

cat("Computing eta_j for all 1600 subregions ...\n")

for (i in seq_len(nrow(results))) {
  rb <- results$row_blk[i]
  cb <- results$col_blk[i]

  # extract 50x50 subgrid
  r_idx <- ((rb - 1) * sub_size + 1):(rb * sub_size)
  c_idx <- ((cb - 1) * sub_size + 1):(cb * sub_size)
  sub   <- Z[r_idx, c_idx]

  for (j in j_vals) {
    eta_mat[i, j] <- eta_j_subgrid(sub, j)
  }
}

results <- cbind(results, as.data.frame(eta_mat))

# add log versions
results <- results %>%
  mutate(
    log_eta1 = log(eta1),
    log_eta2 = log(eta2),
    log_eta3 = log(eta3),
    log_eta4 = log(eta4)
  )

cat("Done.\n")

# ---------------------------------------------------------------
# 3. POWER-LAW FIT per subregion: log(eta_j) = a + alpha * log(j)
# ---------------------------------------------------------------

log_j <- log(j_vals)

alpha_vec     <- numeric(nrow(results))
intercept_vec <- numeric(nrow(results))
r2_vec        <- numeric(nrow(results))

for (i in seq_len(nrow(results))) {
  # response: log(eta_j) for j = 1,2,3,4 in this subregion
  y <- as.numeric(results[i, c("log_eta1", "log_eta2", "log_eta3", "log_eta4")])

  if (any(!is.finite(y))) {
    alpha_vec[i]     <- NA
    intercept_vec[i] <- NA
    r2_vec[i]        <- NA
  } else {
    fit_i <- lm(y ~ log_j)
    coefs <- coef(fit_i)

    intercept_vec[i] <- coefs[1]        # a
    alpha_vec[i]     <- coefs[2]        # slope = alpha

    ss_res <- sum(residuals(fit_i)^2)
    ss_tot <- sum((y - mean(y))^2)
    r2_vec[i] <- 1 - ss_res / ss_tot
  }
}

results$intercept <- intercept_vec
results$alpha     <- alpha_vec
results$r2        <- r2_vec

# optional: residuals at each j for diagnostics in part (c)
resid_mat <- t(apply(
  as.matrix(results[, c("log_eta1", "log_eta2", "log_eta3", "log_eta4")]),
  1,
  function(row) {
    if (any(!is.finite(row))) return(rep(NA, 4))
    residuals(lm(row ~ log_j))
  }
))
colnames(resid_mat) <- paste0("resid_j", 1:4)
results <- cbind(results, resid_mat)

# quick check
head(results[, c("row_blk", "col_blk", "alpha", "r2")])

# ---------------------------------------------------------------
# 4. PART (b): PLOTS
# ---------------------------------------------------------------

## --- helper for 40x40 matrices ---

make_matrix_40 <- function(vec) {
  matrix(vec, nrow = 40, ncol = 40, byrow = FALSE)
}

plot_heatmap_key <- function(mat40, main_title, xlab, ylab, key_title, cols) {
  filled.contour(
    x = 1:40,
    y = 1:40,
    z = mat40,
    color.palette = function(n) cols,
    xlab = xlab,
    ylab = ylab,
    main = main_title,
    key.title = title(main = key_title, cex.main = 0.8),
    plot.axes = {
      axis(1)
      axis(2)
      box()
    }
  )
}

## create matrix of log eta values for later use
log_eta_matrix <- as.matrix(
  results[, c("log_eta1", "log_eta2", "log_eta3", "log_eta4")]
)

## --- 4a. Spatial heatmaps of log(eta_j) for each j ---

for (j in 1:4) {
  png(file.path(ourdir, paste0("hw8_log_eta", j, "_heatmap.png")),
      width = 900, height = 700)

  mat <- make_matrix_40(results[[paste0("log_eta", j)]])
  plot_heatmap_key(
    mat40 = mat,
    main_title = bquote("Spatial map of " * log(eta[.(j)]) * " over 50×50 subregions"),
    xlab = "Column block (Easting, 50-pixel subregions)",
    ylab = "Row block (Northing, 50-pixel subregions)",
    key_title = bquote(log(eta[.(j)])),
    cols = terrain.colors(64)
  )

  mtext(
    paste0(
      "Figure: Heatmap of log(eta_", j, "). ",
      "Each cell is one 50×50 subregion. ",
      "The color bar on the right gives the numeric range: lower values indicate smoother local terrain, ",
      "while higher values indicate rougher local terrain at lag j = ", j, "."
    ),
    side = 1, line = 4, cex = 0.8
  )

  dev.off()
}

## --- 4b. Spatial heatmap of alpha_hat ---

png(file.path(ourdir, "hw8_alpha_heatmap.png"), width = 900, height = 700)

mat_alpha <- make_matrix_40(results$alpha)
plot_heatmap_key(
  mat40 = mat_alpha,
  main_title = expression("Spatial map of local smoothness estimate " * hat(alpha)),
  xlab = "Column block (Easting, 50-pixel subregions)",
  ylab = "Row block (Northing, 50-pixel subregions)",
  key_title = expression(hat(alpha)),
  cols = colorRampPalette(c("navy", "white", "firebrick"))(64)
)

mtext(
  "Figure: Heatmap of the local power-law slope alpha. The color bar gives the numeric range of alpha values. Lower alpha indicates smoother local behavior, while higher alpha indicates rougher local behavior.",
  side = 1, line = 4, cex = 0.8
)

dev.off()

## --- 4c. Overlaid density plots of log(eta_j), j = 1,2,3,4 ---

results_long <- data.frame(
  j = factor(rep(1:4, each = nrow(results))),
  log_eta = c(results$log_eta1,
              results$log_eta2,
              results$log_eta3,
              results$log_eta4)
)

png(file.path(ourdir, "hw8_log_eta_density.png"), width = 900, height = 600)
print(
  ggplot(results_long, aes(x = log_eta, fill = j, color = j)) +
    geom_density(alpha = 0.25, linewidth = 1) +
    labs(
      title = expression("Distribution of " * log(eta[j]) * " across 1600 subregions"),
      subtitle = "Each curve corresponds to one lag value j; curves further to the right indicate greater roughness at that scale.",
      x = expression(log(eta[j])),
      y = "Density over 50×50 subregions",
      fill = "Lag j",
      color = "Lag j",
      caption = "Figure 3: Overlaid densities show how the distribution of local roughness changes as lag increases."
    ) +
    scale_fill_brewer(palette = "Set1") +
    scale_color_brewer(palette = "Set1") +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "right"
    )
)
dev.off()

## --- 4d. Histogram of R^2 values ---

df_r2 <- data.frame(r2 = results$r2)

png(file.path(ourdir, "hw8_r2_hist.png"), width = 900, height = 600)
print(
  ggplot(df_r2, aes(x = r2)) +
    geom_histogram(bins = 40, fill = "steelblue", color = "white") +
    geom_vline(xintercept = 0.95, linetype = "dashed", color = "red", linewidth = 1) +
    labs(
      title = expression("Goodness-of-fit of local power-law model"),
      subtitle = expression("R^2 from regressions of log(eta[j]) on log(j) for each 50×50 subregion"),
      x = expression(R^2 ~ " from log(eta[j]) vs log(j)"),
      y = "Number of subregions",
      caption = "Figure 4: Most subregions have high R^2, indicating that a power-law in j explains the scaling of eta_j locally."
    ) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold"))
)
dev.off()

## --- 4e. Log-log plots: overlay 100 sampled subregions + median ---

set.seed(669)
idx_sample <- sample(nrow(results), 100)

png(file.path(ourdir, "hw8_loglog_overlay.png"), width = 900, height = 650)
plot(log_j, rep(NA, 4),
     ylim = range(log_eta_matrix, na.rm = TRUE),
     xlab = expression(log(j)),
     ylab = expression(log(eta[j])),
     main = "Log–log behavior of eta_j across sampled subregions",
     sub  = "Gray: 100 sampled subregions; Red: median log(eta_j) over all 1600 subregions")

for (i in idx_sample) {
  lines(log_j, log_eta_matrix[i, ], col = rgb(0.5, 0.5, 0.5, 0.25))
}

median_log_eta <- apply(log_eta_matrix, 2, median, na.rm = TRUE)
lines(log_j, median_log_eta, col = "red", lwd = 2)
points(log_j, median_log_eta, col = "red", pch = 19, cex = 1.2)

legend("topleft",
       legend = c("Sampled subregions", "Median across all subregions"),
       col = c(rgb(0.5, 0.5, 0.5, 0.7), "red"),
       lwd = c(1, 2),
       pch = c(NA, 19),
       bty = "n")
mtext("Figure 5: Log(eta_j) versus log(j). A nearly straight median line supports a power-law relationship between eta_j and j.",
      side = 1, line = 4, cex = 0.8)
dev.off()

## --- 4f. Violin + boxplot of alpha_hat ---

df_alpha <- data.frame(alpha = results$alpha)

png(file.path(ourdir, "hw8_alpha_violin.png"), width = 800, height = 600)
print(
  ggplot(df_alpha, aes(x = "All subregions", y = alpha)) +
    geom_violin(fill = "lightblue", color = "navy", alpha = 0.7) +
    geom_boxplot(width = 0.12, fill = "white", outlier.color = "red") +
    labs(
      title = expression("Distribution of local smoothness estimates " * hat(alpha)),
      subtitle = expression("Each point is the slope of log(eta[j]) on log(j) from one 50×50 subregion"),
      x = "",
      y = expression(hat(alpha)),
      caption = "Figure 6: Violin and boxplot summarizing the variability in local power-law exponents across the region."
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title  = element_text(face = "bold"),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank()
    )
)
dev.off()


# ---------------------------------------------------------------
# 5. PART (c): LINEARITY DIAGNOSTICS
# ---------------------------------------------------------------

## --- 5a. Mean residual profile across all subregions ---

mean_resid <- colMeans(resid_mat, na.rm = TRUE)
se_resid   <- apply(
  resid_mat, 2,
  function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))
)

png(file.path(ourdir, "hw8_mean_residuals.png"), width = 900, height = 600)
plot(1:4, mean_resid,
     xlab = "Lag j",
     ylab = "Mean residual",
     main = "Average residuals from local log–log regressions",
     sub  = "Error bars show ±2 standard errors over 1600 subregions",
     ylim = range(c(mean_resid - 2 * se_resid,
                    mean_resid + 2 * se_resid)),
     pch = 19, cex = 1.3, col = "steelblue", type = "b", lwd = 2)
abline(h = 0, lty = 2, col = "gray40")
arrows(1:4, mean_resid - 2 * se_resid,
       1:4, mean_resid + 2 * se_resid,
       angle = 90, code = 3, length = 0.07, col = "steelblue", lwd = 1.5)
mtext("Figure 7: Mean residuals indicate whether log(eta_j) vs log(j) is systematically curved (departing from a pure power law).",
      side = 1, line = 4, cex = 0.8)
dev.off()

## --- 5b. Spatial heatmap of residual at j = 3 ---

png(file.path(ourdir, "hw8_resid_j3_heatmap.png"), width = 900, height = 700)

mat_resid3 <- make_matrix_40(results$resid_j3)
plot_heatmap_key(
  mat40 = mat_resid3,
  main_title = "Spatial pattern of residuals at lag j = 3",
  xlab = "Column block (Easting, 50-pixel subregions)",
  ylab = "Row block (Northing, 50-pixel subregions)",
  key_title = "Residual at j = 3",
  cols = colorRampPalette(c("blue", "white", "red"))(64)
)

mtext(
  "Figure: Heatmap of residuals at j = 3 from the local log-log regression. The color bar gives the numeric range of residual values; blue means observed log(eta_3) is below the fitted line, red means it is above the fitted line.",
  side = 1, line = 4, cex = 0.8
)

dev.off()

## --- 5c. Sample of 9 local log–log plots ---

png(file.path(ourdir, "hw8_sample_loglog.png"), width = 1100, height = 1000)
set.seed(42)
samp9 <- sample(nrow(results), 9)
par(mfrow = c(3, 3), mar = c(4, 4, 3, 1))

for (i in samp9) {
  y <- log_eta_matrix[i, ]
  fit_i <- lm(y ~ log_j)
  alpha_i <- round(coef(fit_i)[2], 3)
  r2_i    <- round(results$r2[i], 3)

  plot(log_j, y,
       xlab = expression(log(j)),
       ylab = expression(log(eta[j])),
       main = paste("Subregion", i),
       pch = 19, col = "steelblue", cex = 1.2)
  abline(fit_i, col = "red", lwd = 2)

  legend("topleft",
         legend = c(
           paste("alpha =", alpha_i),
           paste("R^2 =", r2_i)
         ),
         bty = "n", cex = 0.9)
}

par(mfrow = c(1, 1))
mtext("Figure 9: Examples of local log(eta_j) vs log(j) fits. Each panel shows one 50×50 subregion with its alpha and R^2.",
      side = 1, line = -1.5, cex = 0.8, outer = TRUE)
dev.off()

# ---------------------------------------------------------------
# 6.  NUMERICAL SUMMARIES (printed to console)
# ---------------------------------------------------------------

cat("\n===== NUMERICAL SUMMARY =====\n")
cat("\nMedian alpha:", median(results$alpha, na.rm = TRUE))
cat("\nMean alpha:  ", mean(results$alpha, na.rm = TRUE))
cat("\nIQR alpha:   ", IQR(results$alpha, na.rm = TRUE))
cat("\nFraction R2 > 0.95:", mean(results$r2 > 0.95, na.rm = TRUE))
cat("\n\nMean log(eta_j) by j:\n")
print(colMeans(log_eta_matrix, na.rm = TRUE))
cat("\nMean residuals by j:\n")
print(mean_resid)
cat("\n")

# save results table
write.csv(results, "Stats 669/data/hw7_eta_results.csv", row.names = FALSE)
cat("Results saved to Stats 669/data/hw7_eta_results.csv\n")

