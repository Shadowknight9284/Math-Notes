## ============================================================
## Stats 669 – IRF-2 REML fit on 50x50 regions
## Necessary changes only:
##   1. projected metric coordinates
##   2. IRF-2 generalized covariance K_kappa
##   3. proper polynomial filtering via Q
##   4. profile scale parameter phi in filtered REML
##   5. estimate kappa (not old lambda)
## No major vectorization / batching optimizations included.
## ============================================================

library(terra)

## ------------------------------------------------------------
## 0. Paths
## ------------------------------------------------------------

outdir <- "hw9/output"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

tif_path <- "data/USGS_13_n37w118_20260112.tif"

## ------------------------------------------------------------
## 1. Load raster, crop around center, project to metric CRS,
##    resample to 2000 x 2000 over a 20 km x 20 km window
## ------------------------------------------------------------

dem <- rast(tif_path)
e <- ext(dem)

center_x <- mean(c(e$xmin, e$xmax))
center_y <- mean(c(e$ymin, e$ymax))

# Rough crop first, then project
rough_crop <- crop(
  dem,
  ext(center_x - 0.15, center_x + 0.15,
      center_y - 0.15, center_y + 0.15)
)

dem_meters_rough <- project(rough_crop, "EPSG:5070")

mid_x <- mean(c(xmin(dem_meters_rough), xmax(dem_meters_rough)))
mid_y <- mean(c(ymin(dem_meters_rough), ymax(dem_meters_rough)))

template <- rast(
  xmin = mid_x - 10000, xmax = mid_x + 10000,
  ymin = mid_y - 10000, ymax = mid_y + 10000,
  nrows = 2000, ncols = 2000,
  crs = "EPSG:5070"
)

dem_final <- project(dem_meters_rough, template, method = "bilinear")

# Normalize extent to 0-20 km in each direction
ext(dem_final) <- c(0, 20, 0, 20)

Z_full <- matrix(values(dem_final), nrow = 2000, ncol = 2000, byrow = TRUE)
res_km <- res(dem_final)[1]

if (!all(dim(Z_full) == c(2000, 2000))) {
  stop("Expected 2000 x 2000 raster after reprojection/resampling.")
}

## ------------------------------------------------------------
## 2. Shared geometry for one thinned 25x25 block
##    (this is still conceptually necessary because Q depends
##     on the coordinates; we are not batching across all blocks)
## ------------------------------------------------------------

sub_idx <- seq(1, 50, by = 2)  # every other row/col
n_sub <- length(sub_idx)       # 25
n_obs <- n_sub^2               # 625

row_coord <- rep((sub_idx - 1) * res_km, times = n_sub)
col_coord <- rep((sub_idx - 1) * res_km, each = n_sub)

coords <- cbind(x = col_coord, y = row_coord)
coords <- scale(coords, center = TRUE, scale = FALSE)

D <- as.matrix(dist(coords))
D2 <- D^2
logD <- log(D)
logD[!is.finite(logD)] <- 0
diag(logD) <- 0

# IRF-2 polynomial basis: 1, x, y, x^2, xy, y^2
X <- cbind(
  1,
  coords[, 1],
  coords[, 2],
  coords[, 1]^2,
  coords[, 1] * coords[, 2],
  coords[, 2]^2
)

q <- ncol(X)
nu <- n_obs - q

# Orthogonal complement for filtering out quadratic polynomials
Q <- qr.Q(qr(X), complete = TRUE)[, (q + 1):n_obs, drop = FALSE]

## ------------------------------------------------------------
## 3. Generalized covariance builder for IRF power-law model
## ------------------------------------------------------------

build_K <- function(kappa, D2, logD) {
  if (abs(kappa - round(kappa)) < 1e-12) {
    k <- as.integer(round(kappa))
    (2 * (-1)^(k + 1) / factorial(k)) * (D2^k) * logD
  } else {
    gamma(-kappa) * (D2^kappa)
  }
}

## ------------------------------------------------------------
## 4. Profile REML objective for one block
## ------------------------------------------------------------

fit_one_block_irf2 <- function(z_tile,
                               sub_idx,
                               Q,
                               D2,
                               logD,
                               kappa_grid = seq(0.01, 2.99, by = 0.01)) {
  z <- as.vector(z_tile[sub_idx, sub_idx])

  if (anyNA(z) || length(unique(z)) < 10) {
    return(data.frame(
      kappa = NA_real_,
      phi = NA_real_,
      nll = NA_real_,
      status = "invalid_block"
    ))
  }

  # Filter out polynomials of degree <= 2
  y <- as.numeric(crossprod(Q, z))

  best_kappa <- NA_real_
  best_phi <- NA_real_
  best_nll <- Inf

  for (kappa in kappa_grid) {
    K <- build_K(kappa, D2 = D2, logD = logD)

    Sigma <- crossprod(Q, K %*% Q)
    Sigma <- (Sigma + t(Sigma)) / 2

    Rchol <- try(chol(Sigma), silent = TRUE)
    if (inherits(Rchol, "try-error")) next

    log_det <- 2 * sum(log(diag(Rchol)))

    v <- forwardsolve(t(Rchol), y)
    quad <- sum(v^2)

    if (!is.finite(quad) || quad <= 0) next

    # Profiled scale parameter
    phi_hat <- quad / length(y)

    # Profile negative REML log-likelihood up to additive constant
    nll <- log_det + length(y) * log(phi_hat)

    if (is.finite(nll) && nll < best_nll) {
      best_nll <- nll
      best_kappa <- kappa
      best_phi <- phi_hat
    }
  }

  data.frame(
    kappa = best_kappa,
    phi = best_phi,
    nll = best_nll,
    status = ifelse(is.finite(best_nll), "ok", "no_fit")
  )
}

## ------------------------------------------------------------
## 5. Loop over 50x50 regions
## ------------------------------------------------------------

tile_size <- 50
n_side <- 40

regions <- expand.grid(
  br = 1:n_side,
  bc = 1:n_side
)

# For now, you can restrict to a subset if desired:
regions <- regions[1:10, , drop = FALSE]

results_list <- vector("list", nrow(regions))

cat("Starting IRF-2 REML fits...\n")

t0 <- proc.time()

for (a in seq_len(nrow(regions))) {
  br <- regions$br[a]
  bc <- regions$bc[a]

  r_s <- (br - 1) * tile_size + 1
  r_e <- br * tile_size
  c_s <- (bc - 1) * tile_size + 1
  c_e <- bc * tile_size

  z_tile <- Z_full[r_s:r_e, c_s:c_e]

  fit <- fit_one_block_irf2(
    z_tile = z_tile,
    sub_idx = sub_idx,
    Q = Q,
    D2 = D2,
    logD = logD,
    kappa_grid = seq(0.01, 2.99, by = 0.01)
  )

  results_list[[a]] <- data.frame(
    br = br,
    bc = bc,
    kappa = fit$kappa,
    phi = fit$phi,
    nll = fit$nll,
    status = fit$status
  )

  if (a %% 25 == 0) {
    cat("Finished", a, "of", nrow(regions), "regions\n")
    gc(verbose = FALSE)
  }
}

elapsed <- proc.time() - t0
cat("Elapsed time (seconds):", elapsed["elapsed"], "\n")

results_irf2 <- do.call(rbind, results_list)

write.csv(
  results_irf2,
  file.path(outdir, "hw9_irf2_reml_results.csv"),
  row.names = FALSE
)

cat("Saved results to", file.path(outdir, "hw9_irf2_reml_results.csv"), "\n")
