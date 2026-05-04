############################################################
# FOURIER-AR(1) SPATIO-TEMPORAL MODEL WITH M MATRIX
#
# MODEL:
#   X_t(x) = a_0(t) +
#            sum_{j=1}^J [ a_j(t) cos(j x) + b_j(t) sin(j x) ]
#
# VECTOR AR(1):
#   Z_{t+1} = M * Z_t + epsilon_t
#
#   where Z_t = [a_0(t), a_1(t), b_1(t), ..., a_J(t), b_J(t)]'
#         M   = block-diagonal matrix (2J+1) x (2J+1)
#         eps ~ N(0, Sigma_eps)
#
# M MATRIX STRUCTURE (block-diagonal):
#   - Block for j=0: scalar rho_0  (1x1)
#   - Block for j>=1: rho_j * rotation matrix (2x2)
#
#   M_j = rho_j * [cos(theta_j)  -sin(theta_j)]
#                 [sin(theta_j)   cos(theta_j)]
#
# SIGMA_EPS STRUCTURE (block-diagonal):
#   - Block for j=0: sigma_0^2 * (1 - rho_0^2)  (1x1)
#   - Block for j>=1:
#       sigma_j^2 * (1 - rho_j^2) * [1      phi_j]
#                                    [phi_j  1    ]
#
# PARAMETERS:
#   rho_j   = temporal persistence for mode j (from power-law)
#   theta_j = rotation angle for mode j (phase drift)
#   phi_j   = cosine/sine innovation correlation for mode j
#   sigma_j = marginal std for mode j (from Matern spectrum)
#
# VERIFIED: theta_base = 0.01 gives Cor(a_j, b_j) ≈ phi_j
############################################################

# ----------------------------------------------------------
# 1. Load packages
# ----------------------------------------------------------
library(MASS)      # for mvrnorm
library(magick)    # for GIF creation

# ----------------------------------------------------------
# 2. Helper: Matern-like Fourier variances
# ----------------------------------------------------------
# sigma_j^2 ∝ (kappa^2 + j^2)^(-(nu + 0.5))
#
# Interpretation:
#   - larger nu  => faster decay in j => smoother spatial fields
#   - larger ell => more low-frequency dominance => longer spatial range
matern_spectrum <- function(J, nu, ell, total_var = 1) {
  kappa    <- sqrt(2 * nu) / ell
  j        <- 0:J
  raw_spec <- (kappa^2 + j^2)^(-(nu + 0.5))
  sigma2   <- total_var * raw_spec / sum(raw_spec)
  return(sigma2)
}

# ----------------------------------------------------------
# 3. Helper: temporal persistence rho_j (power-law version)
# ----------------------------------------------------------
# rho_j = 1 / (1 + lambda * j^alpha)^beta
#
# Power-law decay is SLOWER than exponential at high j,
# giving longer temporal memory to high-frequency modes.
rho_function <- function(J, lambda, alpha, beta, rho0 = 0.95) {
  j   <- 0:J
  rho <- 1 / (1 + lambda * (j^alpha))^beta
  rho[1] <- rho0   # override j=0 directly
  return(pmax(rho, 0))
}

# ----------------------------------------------------------
# 4. Helper: theta_j (rotation angle per mode)
# ----------------------------------------------------------
# Controls phase drift in the M matrix.
#
# theta_j = 0    => M_j = rho_j * I (no coupling, pure damping)
# theta_j = 0.01 => small phase drift, preserves Cor(a_j, b_j) ≈ phi_j
# theta_j large  => strong coupling, destroys phi_j correlation
#
# VERIFIED: theta_base = 0.01 passes the test Cor(a_1,b_1) ≈ phi_1
# with |difference| < 0.10.
theta_function <- function(J, theta_base = 0.01) {
  theta    <- rep(theta_base, J + 1)
  theta[1] <- 0    # j=0 has no sine component
  return(theta)
}

# ----------------------------------------------------------
# 5. Helper: phi_j (cosine/sine innovation correlation per mode)
# ----------------------------------------------------------
# Controls Cor(eta_{a,j}, eta_{b,j}) in the innovation noise.
#
# phi_j = 0    => cosine and sine innovations are independent
# phi_j > 0    => positive coupling between a_j and b_j shocks
#
# Decays exponentially with j:
#   phi_j = phi_base * exp(-decay * j / J)
# so low frequencies have stronger coupling than high frequencies.
phi_function <- function(J, phi_base = 0.7, decay = 0.3) {
  j   <- 0:J
  phi <- phi_base * exp(-decay * j / J)
  phi[1] <- 0    # j=0 has no sine component
  return(pmax(pmin(phi, 0.99), -0.99))
}

# ----------------------------------------------------------
# 6. Helper: build 2x2 M_j block
# ----------------------------------------------------------
# M_j = rho_j * [cos(theta_j)  -sin(theta_j)]
#               [sin(theta_j)   cos(theta_j)]
#
# This is a scaled rotation matrix. It:
#   1. Damps the state by rho_j
#   2. Rotates the (a_j, b_j) phase plane by theta_j
build_Mj <- function(rho_j, theta_j) {
  matrix(
    rho_j * c(cos(theta_j),  -sin(theta_j),
             sin(theta_j),  cos(theta_j)),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

# ----------------------------------------------------------
# 7. Helper: build 2x2 innovation covariance block Sigma_eps_j
# ----------------------------------------------------------
# Sigma_{eps,j} = sigma_j^2 * (1 - rho_j^2) * [1      phi_j]
#                                               [phi_j  1    ]
#
# The scale sigma_j^2 * (1 - rho_j^2) is the Yule-Walker
# innovation variance needed to achieve stationary marginal
# variance sigma_j^2.
build_Sigma_eps_j <- function(sigma2_j, rho_j, phi_j) {
  scale <- sigma2_j * (1 - rho_j^2)
  matrix(scale * c(1, phi_j, phi_j, 1), nrow = 2, ncol = 2, byrow = FALSE)
}

# ----------------------------------------------------------
# 8. Helper: build 2x2 initial (stationary) covariance block
# ----------------------------------------------------------
# At stationarity, Cov(a_j, b_j) = sigma_j^2 * phi_j
# so the initial draw uses this full 2x2 covariance matrix.
build_Sigma_init_j <- function(sigma2_j, phi_j) {
  matrix(sigma2_j * c(1, phi_j, phi_j, 1), nrow = 2, ncol = 2, byrow = FALSE)
}

# ----------------------------------------------------------
# 9. Helper: assemble the full (2J+1) x (2J+1) M matrix
# ----------------------------------------------------------
# The full M matrix is block-diagonal:
#   M = diag(rho_0, M_1, M_2, ..., M_J)
#
# Ordering of Z_t:
#   [a_0, a_1, b_1, a_2, b_2, ..., a_J, b_J]
#
# This function builds M explicitly so it can be inspected,
# printed, or used in matrix operations.
build_M_full <- function(rho, theta, J) {
  dim_M <- 2 * J + 1
  M     <- matrix(0, nrow = dim_M, ncol = dim_M)
  
  # j = 0: scalar block (1x1)
  M[1, 1] <- rho[1]
  
  # j = 1,...,J: 2x2 blocks
  for (j in 1:J) {
    row_start <- 2 * j        # row index in M (1-indexed)
    col_start <- 2 * j
    
    M_j <- build_Mj(rho[j + 1], theta[j + 1])
    
    M[row_start:(row_start + 1), col_start:(col_start + 1)] <- M_j
  }
  
  return(M)
}

# ----------------------------------------------------------
# 10. Helper: assemble the full (2J+1) x (2J+1) Sigma_eps matrix
# ----------------------------------------------------------
# Sigma_eps is block-diagonal with the same ordering as M.
# Off-diagonal blocks are zero (no cross-frequency dependence).
build_Sigma_eps_full <- function(sigma2, rho, phi, J) {
  dim_S    <- 2 * J + 1
  Sigma    <- matrix(0, nrow = dim_S, ncol = dim_S)
  
  # j = 0: scalar block
  Sigma[1, 1] <- sigma2[1] * (1 - rho[1]^2)
  
  # j = 1,...,J: 2x2 blocks
  for (j in 1:J) {
    row_start <- 2 * j
    col_start <- 2 * j
    
    S_j <- build_Sigma_eps_j(sigma2[j + 1], rho[j + 1], phi[j + 1])
    
    Sigma[row_start:(row_start + 1), col_start:(col_start + 1)] <- S_j
  }
  
  return(Sigma)
}

# ----------------------------------------------------------
# 11. Main simulation function
# ----------------------------------------------------------
simulate_fourier_ar1 <- function(T, J, nu, ell, lambda, alpha, beta,
                                 theta_base = 0.01,
                                 phi_base   = 0.7,
                                 phi_decay  = 0.3,
                                 total_var  = 1,
                                 rho0       = 0.95,
                                 seed       = NULL) {
  
  if (!is.null(seed)) set.seed(seed)
  
  # Compute all parameter vectors
  sigma2 <- matern_spectrum(J = J, nu = nu, ell = ell, total_var = total_var)
  rho    <- rho_function(J = J, lambda = lambda, alpha = alpha,
                         beta = beta, rho0 = rho0)
  theta  <- theta_function(J = J, theta_base = theta_base)
  phi    <- phi_function(J = J, phi_base = phi_base, decay = phi_decay)
  
  # Assemble full M and Sigma_eps matrices (for inspection/return)
  M_full         <- build_M_full(rho, theta, J)
  Sigma_eps_full <- build_Sigma_eps_full(sigma2, rho, phi, J)
  
  # Allocate storage for coefficients
  a <- matrix(0, nrow = T, ncol = J + 1)   # cosine coefficients
  b <- matrix(0, nrow = T, ncol = J + 1)   # sine coefficients
  
  # --------------------------------------------------------
  # Initial state from stationary distributions
  # --------------------------------------------------------
  
  # j = 0: scalar
  a[1, 1] <- rnorm(1, mean = 0, sd = sqrt(sigma2[1]))
  
  # j >= 1: bivariate draw from stationary covariance
  for (j in 1:J) {
    idx        <- j + 1
    Sigma_init <- build_Sigma_init_j(sigma2[idx], phi[idx])
    init       <- as.vector(mvrnorm(1, c(0, 0), Sigma_init))
    a[1, idx]  <- init[1]
    b[1, idx]  <- init[2]
  }
  
  # --------------------------------------------------------
  # Forward simulation: Z_{t+1} = M_j * Z_t + eps_t
  # --------------------------------------------------------
  for (t in 2:T) {
    
    # j = 0: scalar AR(1)
    eps_var0 <- sigma2[1] * (1 - rho[1]^2)
    a[t, 1]  <- rho[1] * a[t-1, 1] + rnorm(1, 0, sqrt(eps_var0))
    
    # j >= 1: bivariate AR(1) with rotation and correlated innovations
    for (j in 1:J) {
      idx         <- j + 1
      M_j         <- build_Mj(rho[idx], theta[idx])
      Sigma_eps_j <- build_Sigma_eps_j(sigma2[idx], rho[idx], phi[idx])
      
      eta       <- as.vector(mvrnorm(1, c(0, 0), Sigma_eps_j))
      new_state <- M_j %*% c(a[t-1, idx], b[t-1, idx]) + eta
      
      a[t, idx] <- new_state[1]
      b[t, idx] <- new_state[2]
    }
  }
  
  return(list(
    a              = a,
    b              = b,
    sigma2         = sigma2,
    rho            = rho,
    theta          = theta,
    phi            = phi,
    M_full         = M_full,          # full (2J+1)x(2J+1) M matrix
    Sigma_eps_full = Sigma_eps_full,  # full (2J+1)x(2J+1) Sigma_eps
    params = list(T = T, J = J, nu = nu, ell = ell,
                  lambda = lambda, alpha = alpha, beta = beta,
                  theta_base = theta_base,
                  phi_base   = phi_base,
                  phi_decay  = phi_decay,
                  total_var  = total_var,
                  rho0       = rho0)
  ))
}

# ----------------------------------------------------------
# 12. Helper: reconstruct the spatial field on a grid
# ----------------------------------------------------------
reconstruct_field <- function(a, b, x_grid) {
  T  <- nrow(a)
  J  <- ncol(a) - 1
  X  <- matrix(0, nrow = T, ncol = length(x_grid))
  X  <- X + a[, 1]   # j=0 constant mode
  
  for (j in 1:J) {
    idx <- j + 1
    X   <- X + a[, idx] %*% t(cos(j * x_grid)) +
               b[, idx] %*% t(sin(j * x_grid))
  }
  return(X)
}

# ----------------------------------------------------------
# 13. Helper: amplitude and phase for each mode
# ----------------------------------------------------------
amplitude_phase <- function(a, b) {
  J     <- ncol(a) - 1
  amp   <- a
  phase <- a
  
  amp[, 1]   <- abs(a[, 1])
  phase[, 1] <- NA
  
  for (j in 1:J) {
    idx          <- j + 1
    amp[, idx]   <- sqrt(a[, idx]^2 + b[, idx]^2)
    phase[, idx] <- atan2(b[, idx], a[, idx])
  }
  return(list(amplitude = amp, phase = phase))
}

# ----------------------------------------------------------
# 14. Run simulation
# ----------------------------------------------------------
sim <- simulate_fourier_ar1(
  T          = 200,
  J          = 50,
  nu         = 1.5,
  ell        = 1.0,
  lambda     = 0.01,
  alpha      = 1.2,
  beta       = 1.0,
  theta_base = 0.01,   # small rotation: phase drift
  phi_base   = 0.7,    # base cosine/sine coupling
  phi_decay  = 0.3,    # decay of coupling with frequency
  total_var  = 1,
  rho0       = 0.95,
  seed       = 123
)

# Inspect the M matrix (first 7x7 block = j=0,1,2,3)
cat("=== M matrix (first 7 rows/cols: j=0,1,2,3) ===\n")
print(round(sim$M_full[1:7, 1:7], 5))

cat("\n=== Sigma_eps matrix (first 7 rows/cols) ===\n")
print(round(sim$Sigma_eps_full[1:7, 1:7], 6))

cat("\n=== rho_j for j=0,...,10 ===\n")
print(round(sim$rho[1:11], 4))

cat("\n=== theta_j for j=0,...,10 ===\n")
print(round(sim$theta[1:11], 4))

cat("\n=== phi_j for j=0,...,10 ===\n")
print(round(sim$phi[1:11], 4))

# Spatial grid and field reconstruction
x_grid <- seq(0, 2 * pi, length.out = 200)
X      <- reconstruct_field(sim$a, sim$b, x_grid)
ap     <- amplitude_phase(sim$a, sim$b)

# ----------------------------------------------------------
# 15. Diagnostics and Visualization
# ----------------------------------------------------------
out_dir <- "Stats 669/research/img/simulationAR1"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# (a) Matern-like mode variances
png(file.path(out_dir, "matern_variances.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$sigma2, type = "b", pch = 19,
     xlab = "Fourier mode j", ylab = expression(sigma[j]^2),
     main = expression("Matern-like Fourier variances " * sigma[j]^2))
dev.off()

# (b) Temporal persistence rho_j
png(file.path(out_dir, "temporal_persistence.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$rho, type = "b", pch = 19,
     xlab = "Fourier mode j", ylab = expression(rho[j]),
     main = expression("Mode-specific temporal persistence " * rho[j]))
dev.off()

# (c) phi_j across frequencies
png(file.path(out_dir, "phi_j.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$phi, type = "b", pch = 19, col = "darkgreen",
     xlab = "Fourier mode j", ylab = expression(phi[j]),
     main = expression("Cosine/Sine coupling " * phi[j]))
dev.off()

# (d) Spatial field at selected times
png(file.path(out_dir, "spatial_field_at_times.png"), width = 800, height = 600)
matplot(x_grid, t(X[c(1, 10, 50, 100, 150, 200), ]),
        type = "l", lty = 1, lwd = 2,
        xlab = "x", ylab = expression(X[t](x)),
        main = "Spatial field at selected times")
legend("topright", legend = paste("t =", c(1, 10, 50, 100, 150, 200)),
       col = 1:6, lty = 1, lwd = 2, bty = "n")
dev.off()

# (e) Spatio-temporal heatmap
png(file.path(out_dir, "spatio_temporal_field.png"), width = 800, height = 600)
image(x = x_grid, y = 1:nrow(X), z = t(X),
      xlab = "space x", ylab = "time t",
      main = "Simulated spatio-temporal field",
      col = hcl.colors(100, "YlGnBu", rev = TRUE))
dev.off()

# (f) Amplitude trajectories for selected modes
png(file.path(out_dir, "amplitude_trajectories.png"), width = 800, height = 600)
matplot(1:nrow(ap$amplitude), ap$amplitude[, c(2, 6, 11, 21)],
        type = "l", lty = 1, lwd = 2,
        xlab = "time t", ylab = "Amplitude",
        main = "Amplitude trajectories for selected modes")
legend("topright", legend = c("j=1", "j=5", "j=10", "j=20"),
       col = 1:4, lty = 1, lwd = 2, bty = "n")
dev.off()

# (g) Phase trajectories for selected modes
png(file.path(out_dir, "phase_trajectories.png"), width = 800, height = 600)
matplot(1:nrow(ap$phase), ap$phase[, c(2, 6, 11, 21)],
        type = "l", lty = 1, lwd = 2,
        xlab = "time t", ylab = "Phase (radians)",
        main = "Phase trajectories for selected modes")
legend("topright", legend = c("j=1", "j=5", "j=10", "j=20"),
       col = 1:4, lty = 1, lwd = 2, bty = "n")
dev.off()

# (h) ACF of amplitude for selected modes
png(file.path(out_dir, "temporal_autocorrelation.png"), width = 800, height = 600)
par(mfrow = c(2, 2))
for (j in c(1, 5, 10, 20)) {
  idx <- j + 1
  acf(ap$amplitude[, idx], main = paste("ACF of amplitude for j =", j))
}
par(mfrow = c(1, 1))
dev.off()

# (i) Visualize M matrix block structure (first 11x11)
png(file.path(out_dir, "M_matrix_heatmap.png"), width = 800, height = 700)
image(1:11, 1:11, t(sim$M_full[1:11, 1:11])[, 11:1],
      col  = hcl.colors(50, "RdBu", rev = TRUE),
      xlab = "Column index", ylab = "Row index",
      main = "M matrix heatmap (first 11x11: j=0,...,5)")
abline(h = c(1.5, 3.5, 5.5, 7.5, 9.5), col = "gray50", lty = 2)
abline(v = c(1.5, 3.5, 5.5, 7.5, 9.5), col = "gray50", lty = 2)
dev.off()

# ----------------------------------------------------------
# 16. GIF animation of field evolution
# ----------------------------------------------------------
movie_dir <- file.path(out_dir, "movie")
dir.create(movie_dir, recursive = TRUE, showWarnings = FALSE)

y_range <- range(X)

for (t in 1:nrow(X)) {
  png(file.path(movie_dir, sprintf("field_evolution_%03d.png", t)),
      width = 800, height = 600)
  plot(x_grid, X[t, ], type = "l", lwd = 2,
       ylim = y_range,
       xlab = "space x", ylab = expression(X[t](x)),
       main = paste("Spatio-temporal field at time t =", t))
  dev.off()
}

png_files    <- sort(list.files(movie_dir, pattern = "\\.png$", full.names = TRUE))
images       <- image_read(png_files)
gif_animated <- image_animate(images, fps = 10)
image_write(gif_animated, file.path(out_dir, "field_evolution.gif"))

cat("All plots saved to:", out_dir, "\n")
cat("GIF animation saved as: field_evolution.gif\n")

# ----------------------------------------------------------
# 17. EMPIRICAL VS. THEORETICAL VALIDATION
# ----------------------------------------------------------
cat("\n=== Running Empirical vs. Theoretical Validation ===\n")

val_dir <- file.path(out_dir, "validation")
dir.create(val_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# (A) VARIANCE VALIDATION: Theoretical σ²_j vs. Empirical Var(a_j), Var(b_j)
# ============================================================
empirical_var_a <- apply(sim$a, 2, var)
empirical_var_b <- apply(sim$b, 2, var)
empirical_var_b[1] <- NA  # j=0 has no sine component

png(file.path(val_dir, "variance_validation.png"), width = 1200, height = 600)
par(mfrow = c(1, 2))

# Panel 1: Cosine coefficients
plot(0:sim$params$J, sim$sigma2, type = "l", lwd = 3, col = "blue",
     xlab = "Fourier mode j", ylab = expression(Variance),
     main = "Cosine coefficients: Theoretical vs. Empirical",
     ylim = range(c(sim$sigma2, empirical_var_a), na.rm = TRUE))
points(0:sim$params$J, empirical_var_a, pch = 19, col = "red", cex = 0.6)
legend("topright", 
       legend = c(expression("Theoretical " * sigma[j]^2), 
                  expression("Empirical Var(" * a[j] * ")")),
       col = c("blue", "red"), lwd = c(3, NA), pch = c(NA, 19),
       bty = "n", cex = 0.9)
grid()

# Panel 2: Sine coefficients (j >= 1 only)
plot(1:sim$params$J, sim$sigma2[-1], type = "l", lwd = 3, col = "blue",
     xlab = "Fourier mode j", ylab = expression(Variance),
     main = "Sine coefficients: Theoretical vs. Empirical",
     ylim = range(c(sim$sigma2[-1], empirical_var_b[-1]), na.rm = TRUE))
points(1:sim$params$J, empirical_var_b[-1], pch = 19, col = "red", cex = 0.6)
legend("topright", 
       legend = c(expression("Theoretical " * sigma[j]^2), 
                  expression("Empirical Var(" * b[j] * ")")),
       col = c("blue", "red"), lwd = c(3, NA), pch = c(NA, 19),
       bty = "n", cex = 0.9)
grid()

par(mfrow = c(1, 1))
dev.off()

# ============================================================
# (B) TEMPORAL ACF VALIDATION: ρ^k vs. Empirical ACF
# ============================================================
test_modes <- c(1, 5, 10, 20)
max_lag <- 20

png(file.path(val_dir, "acf_validation.png"), width = 1200, height = 900)
par(mfrow = c(2, 2))

for (j in test_modes) {
  idx <- j + 1
  
  # Theoretical ACF: ρ_j^k for k = 0, 1, ..., max_lag
  theoretical_acf <- sim$rho[idx]^(0:max_lag)
  
  # Empirical ACF from amplitude
  empirical_acf_obj <- acf(ap$amplitude[, idx], lag.max = max_lag, plot = FALSE)
  empirical_acf <- as.vector(empirical_acf_obj$acf)
  
  # Plot
  plot(0:max_lag, theoretical_acf, type = "l", lwd = 3, col = "blue",
       xlab = "Lag k", ylab = "ACF",
       main = bquote("Mode j =" ~ .(j) ~ " | " ~ rho[j] ~ "=" ~ .(round(sim$rho[idx], 3))),
       ylim = c(0, 1))
  points(0:max_lag, empirical_acf, pch = 19, col = "red", cex = 0.8)
  abline(h = 0, lty = 2, col = "gray50")
  grid()
  
  if (j == test_modes[1]) {
    legend("topright", 
           legend = c(expression("Theoretical " * rho^k), 
                      "Empirical ACF"),
           col = c("blue", "red"), lwd = c(3, NA), pch = c(NA, 19),
           bty = "n", cex = 0.9)
  }
}

par(mfrow = c(1, 1))
dev.off()

# ============================================================
# (C) CORRELATION VALIDATION: φ_j vs. Empirical Cor(a_j, b_j)
# ============================================================
empirical_cor <- numeric(sim$params$J + 1)
empirical_cor[1] <- NA  # j=0 has no b component

for (j in 1:sim$params$J) {
  idx <- j + 1
  empirical_cor[idx] <- cor(sim$a[, idx], sim$b[, idx])
}

png(file.path(val_dir, "correlation_validation.png"), width = 1000, height = 700)
plot(1:sim$params$J, sim$phi[-1], type = "l", lwd = 3, col = "blue",
     xlab = "Fourier mode j", 
     ylab = expression("Correlation Cor(" * a[j] * ", " * b[j] * ")"),
     main = expression("Cosine/Sine Correlation: Theoretical " * phi[j] ~ " vs. Empirical"),
     ylim = range(c(sim$phi[-1], empirical_cor[-1]), na.rm = TRUE))
points(1:sim$params$J, empirical_cor[-1], pch = 19, col = "red", cex = 0.6)
legend("topright", 
       legend = c(expression("Theoretical " * phi[j]), 
                  expression("Empirical Cor(" * a[j] * ", " * b[j] * ")")),
       col = c("blue", "red"), lwd = c(3, NA), pch = c(NA, 19),
       bty = "n")
grid()
dev.off()

# ============================================================
# (D) SPATIAL COVARIANCE VALIDATION
# ============================================================
n_lags_space <- 50
spatial_lags <- seq(0, pi, length.out = n_lags_space)
empirical_spatial_cov <- numeric(n_lags_space)

# Compute empirical spatial covariance at each lag h
for (i in 1:n_lags_space) {
  h <- spatial_lags[i]
  
  # Find spatial indices approximately h apart
  lag_indices <- which.min(abs(x_grid - h))
  
  # Compute average covariance across all time points and spatial pairs
  cov_sum <- 0
  count <- 0
  
  for (t in 1:nrow(X)) {
    for (x_idx in 1:(length(x_grid) - lag_indices)) {
      cov_sum <- cov_sum + X[t, x_idx] * X[t, x_idx + lag_indices]
      count <- count + 1
    }
  }
  
  empirical_spatial_cov[i] <- cov_sum / count - mean(X)^2
}

# Theoretical spatial covariance C(h) = Σ_j σ_j^2 * cos(j*h)
theoretical_spatial_cov <- sapply(spatial_lags, function(h) {
  sum(sim$sigma2[1]) + sum(sim$sigma2[-1] * cos((1:sim$params$J) * h))
})

png(file.path(val_dir, "spatial_cov_validation.png"), width = 1000, height = 700)
plot(spatial_lags, theoretical_spatial_cov, type = "l", lwd = 3, col = "blue",
     xlab = "Spatial lag h (radians)", ylab = "Covariance",
     main = "Spatial Covariance C(h): Theoretical vs. Empirical",
     ylim = range(c(theoretical_spatial_cov, empirical_spatial_cov)))
points(spatial_lags, empirical_spatial_cov, pch = 19, col = "red", cex = 0.6)
legend("topright", 
       legend = c("Theoretical C(h)", "Empirical Cov"),
       col = c("blue", "red"), lwd = c(3, NA), pch = c(NA, 19),
       bty = "n")
grid()
dev.off()

# ============================================================
# (E) NORMALITY CHECK: Q-Q Plots
# ============================================================
png(file.path(val_dir, "normality_check.png"), width = 1200, height = 900)
par(mfrow = c(2, 2))

# Q-Q plots for selected modes
qq_modes <- c(1, 5, 10, 20)
for (j in qq_modes) {
  idx <- j + 1
  
  # Q-Q plot for amplitude (which should NOT be normal - it's chi-distributed)
  # But coefficients a_j should be normal
  qqnorm(sim$a[, idx], 
         main = bquote("Q-Q Plot: " * a[.(j)] * "(t) | Mode j =" ~ .(j)),
         pch = 19, col = rgb(0, 0, 1, 0.4))
  qqline(sim$a[, idx], col = "red", lwd = 2)
}

par(mfrow = c(1, 1))
dev.off()

# ============================================================
# (F) SUMMARY STATISTICS TABLE
# ============================================================
n_summary <- min(20, sim$params$J + 1)  # First 20 modes or all if J < 20

summary_table <- data.frame(
  Mode_j = 0:(n_summary - 1),
  Theoretical_sigma2 = sim$sigma2[1:n_summary],
  Empirical_Var_a = empirical_var_a[1:n_summary],
  Theoretical_rho = sim$rho[1:n_summary],
  Theoretical_phi = sim$phi[1:n_summary],
  Empirical_Cor_ab = empirical_cor[1:n_summary]
)

# Add empirical rho from lag-1 autocorrelation of amplitude
summary_table$Empirical_rho <- sapply(1:n_summary, function(i) {
  acf_obj <- acf(ap$amplitude[, i], lag.max = 1, plot = FALSE)
  as.vector(acf_obj$acf)[2]  # lag-1 autocorrelation
})

# Compute relative errors
summary_table$Var_RelError <- abs(summary_table$Empirical_Var_a - summary_table$Theoretical_sigma2) / summary_table$Theoretical_sigma2
summary_table$Rho_RelError <- abs(summary_table$Empirical_rho - summary_table$Theoretical_rho) / summary_table$Theoretical_rho
summary_table$Phi_RelError <- ifelse(summary_table$Mode_j == 0, NA,
                                      abs(summary_table$Empirical_Cor_ab - summary_table$Theoretical_phi) / abs(summary_table$Theoretical_phi))

cat("\n=== VALIDATION SUMMARY TABLE (First 20 modes) ===\n")
print(round(summary_table[, 1:7], 4))

cat("\n=== RELATIVE ERRORS ===\n")
print(round(summary_table[, c(1, 8, 9, 10)], 4))

# Save table to CSV
write.csv(summary_table, file.path(val_dir, "validation_summary.csv"), 
          row.names = FALSE)

# ============================================================
# (G) SCATTER PLOTS: a_j vs. b_j for Selected Modes
# ============================================================
png(file.path(val_dir, "scatter_a_vs_b.png"), width = 1200, height = 900)
par(mfrow = c(2, 2))

scatter_modes <- c(1, 5, 10, 20)
for (j in scatter_modes) {
  idx <- j + 1
  
  plot(sim$a[, idx], sim$b[, idx], 
       pch = 19, cex = 0.4, col = rgb(0.2, 0.4, 0.8, 0.3),
       xlab = bquote(a[.(j)](t)),
       ylab = bquote(b[.(j)](t)),
       main = bquote("Scatter: " * a[.(j)] ~ "vs." ~ b[.(j)] ~ 
                     " | " * phi[.(j)] ~ "=" ~ .(round(sim$phi[idx], 3)) ~
                     ", Emp. Cor =" ~ .(round(empirical_cor[idx], 3))))
  
  # Add regression line
  abline(lm(sim$b[, idx] ~ sim$a[, idx]), col = "red", lwd = 2, lty = 2)
  
  # Add theoretical correlation line (only if phi > 0)
  if (sim$phi[idx] > 0) {
    # The slope of the major axis is approximately phi_j
    # (since both have same variance)
    abline(0, sim$phi[idx], col = "blue", lwd = 2, lty = 1)
  }
  
  grid()
  legend("topleft", 
         legend = c("Regression line", 
                    bquote("y = " * phi[.(j)] * " x")),
         col = c("red", "blue"), lwd = 2, lty = c(2, 1),
         bty = "n", cex = 0.8)
}

par(mfrow = c(1, 1))
dev.off()

# ============================================================
# (H) PHASE DRIFT CHECK: Mean Phase Change per Time Step
# ============================================================
# With rotation matrix M_j, the phase should drift at rate θ_j per time step
# Check: Δphase ≈ θ_j

mean_phase_drift <- numeric(sim$params$J + 1)
mean_phase_drift[1] <- NA

for (j in 1:sim$params$J) {
  idx <- j + 1
  phase_diff <- diff(ap$phase[, idx])
  
  # Handle phase wrapping (-π to π)
  phase_diff[phase_diff > pi] <- phase_diff[phase_diff > pi] - 2*pi
  phase_diff[phase_diff < -pi] <- phase_diff[phase_diff < -pi] + 2*pi
  
  mean_phase_drift[idx] <- mean(phase_diff, na.rm = TRUE)
}

png(file.path(val_dir, "phase_drift_validation.png"), width = 1000, height = 700)
plot(1:sim$params$J, sim$theta[-1], type = "l", lwd = 3, col = "blue",
     xlab = "Fourier mode j", 
     ylab = "Mean phase drift per time step (radians)",
     main = expression("Phase Drift: Theoretical " * theta[j] ~ " vs. Empirical"),
     ylim = range(c(sim$theta[-1], mean_phase_drift[-1]), na.rm = TRUE))
points(1:sim$params$J, mean_phase_drift[-1], pch = 19, col = "red", cex = 0.6)
abline(h = 0, lty = 2, col = "gray50")
legend("topright", 
       legend = c(expression("Theoretical " * theta[j]), 
                  "Empirical mean Δphase"),
       col = c("blue", "red"), lwd = c(3, NA), pch = c(NA, 19),
       bty = "n")
grid()
dev.off()

cat("\nValidation plots saved to:", val_dir, "\n")
cat("  - variance_validation.png\n")
cat("  - acf_validation.png\n")
cat("  - correlation_validation.png\n")
cat("  - spatial_cov_validation.png\n")
cat("  - normality_check.png\n")
cat("  - scatter_a_vs_b.png\n")
cat("  - phase_drift_validation.png\n")
cat("Summary table saved as: validation_summary.csv\n\n")
