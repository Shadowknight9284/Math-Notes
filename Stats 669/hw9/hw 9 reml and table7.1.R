setwd("C:\\Users\\prana\\OneDrive\\Desktop\\MathNotes\\Stats 669")
library(terra)
library(progress)

OUTPUT_FILE <- "hw9/pranav_irf2_reml_results_50x50.rds"
TIF_PATH    <- "hw9/USGS_13_n37w118_20260112.tif"
dir.create("hw9", showWarnings = FALSE)

job_start <- Sys.time()
cat("=== IRF-2 REML + Table 7.1 + Averaging Model ===\n")
cat("Started:", format(job_start), "\n\n")

# -------------------------------------------------------------------
# 1. Load raster
# -------------------------------------------------------------------
cat("[1/6] Loading data...\n")
t0 <- Sys.time()

dem <- rast(TIF_PATH)
Z_full <- matrix(values(dem), nrow = 2000, ncol = 2000, byrow = TRUE)
res_km <- res(dem)[1]

cat(sprintf("done in %.1fs\n\n", as.numeric(Sys.time() - t0, units = "secs")))

# -------------------------------------------------------------------
# 2. Shared geometry for 50x50 IRF-2 REML
# -------------------------------------------------------------------
cat("[2/6] Precomputing shared geometry (50x50 IRF-2)...\n")
t0 <- Sys.time()

tile_size <- 50L
n_side <- 40L
sub_idx <- 1:50
n_obs <- length(sub_idx)^2

row_coord <- rep((sub_idx - 1) * res_km, times = length(sub_idx))
col_coord <- rep((sub_idx - 1) * res_km, each  = length(sub_idx))
coords <- cbind(x = col_coord, y = row_coord)
coords <- scale(coords, center = TRUE, scale = FALSE)

D <- as.matrix(dist(coords))
D2 <- D^2
logD <- log(D)
logD[!is.finite(logD)] <- 0
diag(logD) <- 0

X <- cbind(1, coords[,1], coords[,2], coords[,1]^2, coords[,1] * coords[,2], coords[,2]^2)
q <- ncol(X)
nu <- n_obs - q
Q <- qr.Q(qr(X), complete = TRUE)[, (q + 1):n_obs, drop = FALSE]

regions <- expand.grid(br = 1:n_side, bc = 1:n_side)
n_regions <- nrow(regions)

cat(sprintf("done in %.1fs\n\n", as.numeric(Sys.time() - t0, units = "secs")))

# -------------------------------------------------------------------
# 3. Extract projected contrasts for all 1600 regions
# -------------------------------------------------------------------
cat("[3/6] Extracting projected contrasts for 1600 regions...\n")
t0 <- Sys.time()

Y <- matrix(NA_real_, nrow = nu, ncol = n_regions)
valid <- rep(FALSE, n_regions)

for (a in seq_len(n_regions)) {
  br <- regions$br[a]
  bc <- regions$bc[a]

  r_s <- (br - 1) * tile_size + 1
  r_e <- br * tile_size
  c_s <- (bc - 1) * tile_size + 1
  c_e <- bc * tile_size

  z <- as.vector(Z_full[r_s:r_e, c_s:c_e])

  if (anyNA(z) || length(unique(z)) < 10) next

  Y[, a] <- as.numeric(crossprod(Q, z))
  valid[a] <- TRUE
}

valid_cols <- which(valid)
Yv <- Y[, valid_cols, drop = FALSE]

cat(sprintf("done in %.1fs (%d valid regions)\n\n",
            as.numeric(Sys.time() - t0, units = "secs"),
            length(valid_cols)))

# -------------------------------------------------------------------
# 4. Standard REML over kappa grid
# -------------------------------------------------------------------
cat("[4/6] Standard REML over kappa grid...\n")
t0 <- Sys.time()

kappa_grid <- seq(0.01, 2.99, by = 0.01)
n_kappa <- length(kappa_grid)

build_K <- function(kappa, D2_mat, logD_mat) {
  if (abs(kappa - round(kappa)) < 1e-12) {
    k <- as.integer(round(kappa))
    (2 * (-1)^(k + 1) / factorial(k)) * (D2_mat^k) * logD_mat
  } else {
    gamma(-kappa) * (D2_mat^kappa)
  }
}

nll_grid <- matrix(Inf, nrow = n_kappa, ncol = n_regions)
phi_grid <- matrix(NA_real_, nrow = n_kappa, ncol = n_regions)

pb <- progress_bar$new(
  format = " standard [:bar] :percent ETA: :eta",
  total = n_kappa,
  width = 60
)

for (g in seq_along(kappa_grid)) {
  pb$tick()

  K <- build_K(kappa_grid[g], D2, logD)
  Sigma <- crossprod(Q, K %*% Q)
  Sigma <- (Sigma + t(Sigma)) / 2

  Rchol <- try(chol(Sigma), silent = TRUE)
  if (inherits(Rchol, "try-error")) next

  log_det <- 2 * sum(log(diag(Rchol)))
  V <- forwardsolve(t(Rchol), Yv)
  quad <- colSums(V^2)

  bad <- !is.finite(quad) | quad <= 0
  nll <- log_det + nu * log(quad / nu)
  nll[bad] <- Inf

  nll_grid[g, valid_cols] <- nll
  phi_grid[g, valid_cols] <- quad / nu
}

min_nll <- apply(nll_grid, 2, min)
has_fit <- is.finite(min_nll)
fit_cols <- which(has_fit)

best_idx <- rep(NA_integer_, n_regions)
best_idx[has_fit] <- max.col(-t(nll_grid[, has_fit, drop = FALSE]), ties.method = "first")

kappa_hat <- rep(NA_real_, n_regions)
phi_hat   <- rep(NA_real_, n_regions)
nll_hat   <- rep(NA_real_, n_regions)

kappa_hat[fit_cols] <- kappa_grid[best_idx[fit_cols]]
phi_hat[fit_cols]   <- phi_grid[cbind(best_idx[fit_cols], fit_cols)]
nll_hat[fit_cols]   <- nll_grid[cbind(best_idx[fit_cols], fit_cols)]

out <- data.frame(
  br    = regions$br,
  bc    = regions$bc,
  kappa = kappa_hat,
  phi   = phi_hat,
  nll   = nll_hat
)

saveRDS(out, OUTPUT_FILE)
write.csv(out, "hw9/pranav_irf2_reml_results_50x50.csv", row.names = FALSE)

cat(sprintf("\ndone in %.1fs\n", as.numeric(Sys.time() - t0, units = "secs")))
cat(sprintf("Q1 summary: NAs = %d / %d, kappa min = %.2f, median = %.2f, max = %.2f\n\n",
            sum(is.na(out$kappa)), nrow(out),
            min(out$kappa, na.rm = TRUE),
            median(out$kappa, na.rm = TRUE),
            max(out$kappa, na.rm = TRUE)))

png("hw9/pranav_kappa_histogram_irf2_reml.png", width = 700, height = 600, res = 96)
hist(out$kappa,
     breaks = 30,
     col = "steelblue",
     border = "white",
     main = "Histogram of Kappa Estimates (IRF-2 REML)",
     xlab = "Smoothness parameter (kappa)")
dev.off()

# -------------------------------------------------------------------
# 5. Table 7.1 for first 100 regions
# -------------------------------------------------------------------
cat("[5/6] Table 7.1 on first 100 regions...\n")
t_tbl <- Sys.time()

b100 <- regions[1:100, ]

# moment1 / moment2
cat("  - moment estimators...\n")
moment_df <- do.call(rbind, lapply(seq_len(100), function(i) {
  br <- b100$br[i]
  bc <- b100$bc[i]

  m <- Z_full[((br - 1) * 50 + 1):(br * 50), ((bc - 1) * 50 + 1):(bc * 50)]

  eta <- function(k) {
    mean(c(
      as.vector((m[, 1:(50 - k)] - m[, (1 + k):50])^2),
      as.vector((m[1:(50 - k), ] - m[(1 + k):50, ])^2)
    ))
  }

  e1 <- eta(1)
  e2 <- eta(2)
  e4 <- eta(4)

  data.frame(
    br = br,
    bc = bc,
    moment1 = log(e2 / e1) / (2 * log(2)),
    moment2 = log(e4 / e2) / (2 * log(2))
  )
}))

# reml from Q1 output
reml_col <- data.frame(
  br = b100$br,
  bc = b100$bc,
  reml = out$kappa[1:100]
)

# reml2: odd rows / odd cols -> 25x25 IRF-2
cat("  - reml2 (25x25 IRF-2)...\n")
sub2 <- seq(1, 50, by = 2)
n2 <- length(sub2)^2

row2 <- rep((sub2 - 1) * res_km, times = length(sub2))
col2 <- rep((sub2 - 1) * res_km, each = length(sub2))
co2 <- scale(cbind(x = col2, y = row2), center = TRUE, scale = FALSE)

D2_r2 <- as.matrix(dist(co2))
D2sq_r2 <- D2_r2^2
logD2_r2 <- log(D2_r2)
logD2_r2[!is.finite(logD2_r2)] <- 0
diag(logD2_r2) <- 0

X2 <- cbind(1, co2[,1], co2[,2], co2[,1]^2, co2[,1] * co2[,2], co2[,2]^2)
q2 <- 6L
nu2 <- n2 - q2
Q2 <- qr.Q(qr(X2), complete = TRUE)[, (q2 + 1L):n2, drop = FALSE]

build_K2 <- function(kappa) {
  if (abs(kappa - round(kappa)) < 1e-12) {
    k <- as.integer(round(kappa))
    (2 * (-1)^(k + 1) / factorial(k)) * (D2sq_r2^k) * logD2_r2
  } else {
    gamma(-kappa) * (D2sq_r2^kappa)
  }
}

Y2 <- matrix(NA_real_, nrow = nu2, ncol = 100L)

for (i in 1:100) {
  br <- b100$br[i]
  bc <- b100$bc[i]
  m  <- Z_full[((br - 1) * 50 + 1):(br * 50), ((bc - 1) * 50 + 1):(bc * 50)]
  z  <- as.vector(m[sub2, sub2])
  if (!anyNA(z)) Y2[, i] <- crossprod(Q2, z)
}

nll2 <- matrix(Inf, nrow = n_kappa, ncol = 100L)

for (g in seq_along(kappa_grid)) {
  K2 <- build_K2(kappa_grid[g])
  Sig2 <- crossprod(Q2, K2 %*% Q2)
  Sig2 <- (Sig2 + t(Sig2)) / 2

  Rc2 <- try(chol(Sig2), silent = TRUE)
  if (inherits(Rc2, "try-error")) next

  ld2 <- 2 * sum(log(diag(Rc2)))
  V2 <- forwardsolve(t(Rc2), Y2)
  qf2 <- colSums(V2^2)
  qf2[!is.finite(qf2) | qf2 <= 0] <- NA
  v <- which(!is.na(qf2))

  if (length(v) > 0) {
    nll2[g, v] <- ld2 + nu2 * log(qf2[v] / nu2)
  }
}

kappa_r2 <- rep(NA_real_, 100L)
for (j in 1:100) {
  if (all(!is.finite(nll2[, j]))) next
  kappa_r2[j] <- kappa_grid[which.min(nll2[, j])]
}

reml2_df <- data.frame(
  br = b100$br,
  bc = b100$bc,
  reml2 = kappa_r2
)

# blocks5: split each 50x50 into 5x5 blocks, IRF-1
cat("  - blocks5 (100 subblocks per region, IRF-1)...\n")
n5 <- 25L
row5 <- rep((0:4) * res_km, times = 5)
col5 <- rep((0:4) * res_km, each = 5)
co5 <- scale(cbind(x = col5, y = row5), center = TRUE, scale = FALSE)

D5 <- as.matrix(dist(co5))
D5sq <- D5^2
logD5 <- log(D5)
logD5[!is.finite(logD5)] <- 0
diag(logD5) <- 0

X5 <- cbind(1, co5[,1], co5[,2])   # IRF-1
q5 <- 3L
nu5 <- n5 - q5
Q5 <- qr.Q(qr(X5), complete = TRUE)[, (q5 + 1L):n5, drop = FALSE]

build_K5 <- function(kappa) {
  if (abs(kappa - round(kappa)) < 1e-12) {
    k <- as.integer(round(kappa))
    (2 * (-1)^(k + 1) / factorial(k)) * (D5sq^k) * logD5
  } else {
    gamma(-kappa) * (D5sq^kappa)
  }
}

Y5 <- matrix(NA_real_, nrow = nu5, ncol = 100L * 100L)
idx5 <- 0L

for (i in 1:100) {
  br <- b100$br[i]
  bc <- b100$bc[i]
  big <- Z_full[((br - 1) * 50 + 1):(br * 50), ((bc - 1) * 50 + 1):(bc * 50)]

  for (sr in 1:10) {
    for (sc in 1:10) {
      idx5 <- idx5 + 1L
      z5 <- as.vector(big[((sr - 1) * 5 + 1):(sr * 5), ((sc - 1) * 5 + 1):(sc * 5)])
      if (!anyNA(z5)) Y5[, idx5] <- crossprod(Q5, z5)
    }
  }
}

nll5 <- matrix(Inf, nrow = n_kappa, ncol = 100L * 100L)

for (g in seq_along(kappa_grid)) {
  K5 <- build_K5(kappa_grid[g])
  Sig5 <- crossprod(Q5, K5 %*% Q5)
  Sig5 <- (Sig5 + t(Sig5)) / 2

  Rc5 <- try(chol(Sig5), silent = TRUE)
  if (inherits(Rc5, "try-error")) next

  ld5 <- 2 * sum(log(diag(Rc5)))
  V5 <- forwardsolve(t(Rc5), Y5)
  qf5 <- colSums(V5^2)
  qf5[!is.finite(qf5) | qf5 <= 0] <- NA
  v <- which(!is.na(qf5))

  if (length(v) > 0) {
    nll5[g, v] <- ld5 + nu5 * log(qf5[v] / nu5)
  }
}

kappa5_all <- rep(NA_real_, 100L * 100L)
for (j in seq_len(100L * 100L)) {
  if (all(!is.finite(nll5[, j]))) next
  kappa5_all[j] <- kappa_grid[which.min(nll5[, j])]
}

kappa5_mat <- matrix(kappa5_all, nrow = 100L, ncol = 100L)
blocks5_df <- data.frame(
  br = b100$br,
  bc = b100$bc,
  blocks5 = colMeans(kappa5_mat, na.rm = TRUE)
)

# merge and summarize
all_est <- Reduce(
  function(a, b) merge(a, b, by = c("br", "bc")),
  list(moment_df, reml_col, blocks5_df, reml2_df)
)

cols <- c("moment1", "moment2", "reml", "blocks5", "reml2")

tbl71 <- data.frame(
  Estimate = cols,
  mean = round(sapply(cols, function(v) mean(all_est[[v]], na.rm = TRUE)), 3),
  st.dev = round(sapply(cols, function(v) sd(all_est[[v]], na.rm = TRUE)), 3),
  row.names = NULL
)

write.csv(tbl71,   "hw9/table71.csv", row.names = FALSE)
write.csv(all_est, "hw9/all_estimates.csv", row.names = FALSE)

cat(sprintf("Table 7.1 done in %.1f min\n\n",
            as.numeric(Sys.time() - t_tbl, units = "mins")))
print(tbl71, row.names = FALSE)
cat("\n")

# -------------------------------------------------------------------
# 6. Question 3: averaging model on first 100 regions
# -------------------------------------------------------------------
cat("[6/6] Averaging model comparison on first 100 regions...\n")
t_avg <- Sys.time()

cx_diff <- outer(coords[,1], coords[,1], "-")
cy_diff <- outer(coords[,2], coords[,2], "-")

build_K_avg <- function(kappa) {
  h <- 0.5 * res_km
  deltas <- list(
    c(0, 0, 4),
    c(0, h, 2), c(0, -h, 2),
    c(h, 0, 2), c(-h, 0, 2),
    c(h, h, 1), c(h, -h, 1), c(-h, h, 1), c(-h, -h, 1)
  )

  K_sum <- matrix(0, n_obs, n_obs)

  for (d in deltas) {
    dx <- d[1]
    dy <- d[2]
    w  <- d[3]

    D2d <- pmax((cx_diff + dx)^2 + (cy_diff + dy)^2, 0)
    lDd <- 0.5 * log(D2d)
    lDd[!is.finite(lDd)] <- 0
    diag(lDd) <- 0

    if (abs(kappa - round(kappa)) < 1e-12) {
      k <- as.integer(round(kappa))
      Kd <- (2 * (-1)^(k + 1) / factorial(k)) * (D2d^k) * lDd
    } else {
      Kd <- gamma(-kappa) * (D2d^kappa)
    }

    K_sum <- K_sum + w * Kd
  }

  K_sum / 16
}

Y100 <- Y[, 1:100, drop = FALSE]
nll_a <- matrix(Inf, nrow = n_kappa, ncol = 100L)
phi_a <- matrix(NA_real_, nrow = n_kappa, ncol = 100L)

pb3 <- progress_bar$new(
  format = " averaging [:bar] :percent ETA: :eta",
  total = n_kappa,
  width = 60
)

for (g in seq_along(kappa_grid)) {
  pb3$tick()

  Ka <- build_K_avg(kappa_grid[g])
  Sa <- crossprod(Q, Ka %*% Q)
  Sa <- (Sa + t(Sa)) / 2

  Rca <- try(chol(Sa), silent = TRUE)
  if (inherits(Rca, "try-error")) next

  lda <- 2 * sum(log(diag(Rca)))
  Va <- forwardsolve(t(Rca), Y100)
  qfa <- colSums(Va^2)
  qfa[!is.finite(qfa) | qfa <= 0] <- NA
  va <- which(!is.na(qfa))

  if (length(va) > 0) {
    nll_a[g, va] <- lda + nu * log(qfa[va] / nu)
    phi_a[g, va] <- qfa[va] / nu
  }
}

best_a <- rep(NA_integer_, 100L)
for (j in 1:100) {
  if (all(!is.finite(nll_a[, j]))) next
  best_a[j] <- which.min(nll_a[, j])
}

kappa_avg <- rep(NA_real_, 100L)
nll_best_a <- rep(NA_real_, 100L)

good_avg <- which(!is.na(best_a))
kappa_avg[good_avg] <- kappa_grid[best_a[good_avg]]
nll_best_a[good_avg] <- nll_a[cbind(best_a[good_avg], good_avg)]

# IMPORTANT FIX:
# use already-saved best standard-model NLLs instead of re-indexing rA$nll_grid
stopifnot(all(out$br[1:100] == b100$br), all(out$bc[1:100] == b100$bc))
nll_best_s <- out$nll[1:100]

diff_nll <- nll_best_a - nll_best_s

avg_df <- data.frame(
  br = b100$br,
  bc = b100$bc,
  kappa_std = out$kappa[1:100],
  kappa_avg = kappa_avg,
  nll_std = nll_best_s,
  nll_avg = nll_best_a,
  delta_nll = diff_nll
)

write.csv(avg_df, "hw9/part3_averaging_comparison.csv", row.names = FALSE)

cat(sprintf("\nPart 3 done in %.1f min\n",
            as.numeric(Sys.time() - t_avg, units = "mins")))
cat(sprintf("kappa_avg: min = %.2f, median = %.2f, max = %.2f\n",
            min(kappa_avg, na.rm = TRUE),
            median(kappa_avg, na.rm = TRUE),
            max(kappa_avg, na.rm = TRUE)))
cat(sprintf("delta NLL mean = %.3f; regions where averaging improves fit = %d / 100\n\n",
            mean(diff_nll, na.rm = TRUE),
            sum(diff_nll < 0, na.rm = TRUE)))

png("hw9/part3_comparison.png", width = 1000, height = 450, res = 96)
par(mfrow = c(1, 3))

hist(out$kappa[1:100],
     breaks = 20, col = "steelblue", border = "white",
     main = "kappa: standard", xlab = "kappa", xlim = c(0, 3))

hist(kappa_avg,
     breaks = 20, col = "tomato", border = "white",
     main = "kappa: averaging", xlab = "kappa", xlim = c(0, 3))

hist(diff_nll,
     breaks = 20, col = "gold", border = "white",
     main = "NLL(avg) - NLL(std)\nnegative = avg better",
     xlab = "delta NLL")
abline(v = 0, col = "red", lwd = 2)

par(mfrow = c(1, 1))
dev.off()

cat("Outputs written to hw9/\n")
cat(sprintf("\n=== ALL DONE: %.1f min ===\n",
            as.numeric(Sys.time() - job_start, units = "mins")))