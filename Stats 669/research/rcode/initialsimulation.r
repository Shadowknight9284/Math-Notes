############################################################
# SIMPLE FOURIER-AR(1) SPATIO-TEMPORAL MODEL ON A PERIODIC DOMAIN
#
# MODEL:
#   X_t(x) = a_0(t) +
#            sum_{j=1}^J [ a_j(t) cos(j x) + b_j(t) sin(j x) ]
#
# TEMPORAL DYNAMICS:
#   For each frequency j, the pair (a_j(t), b_j(t)) follows
#   a simple bivariate AR(1) with persistence rho_j.
#
#   Here we start with the simplest block-diagonal model:
#   - no drift
#   - no cross-frequency dependence
#   - no cosine/sine mixing within a frequency
#
#   So for j >= 1:
#     a_j(t+1) = rho_j * a_j(t) + eta_{a,j,t}
#     b_j(t+1) = rho_j * b_j(t) + eta_{b,j,t}
#
#   and for j = 0:
#     a_0(t+1) = rho_0 * a_0(t) + eta_{0,t}
#
# SPATIAL SPECTRUM:
#   sigma_j^2 ∝ (kappa^2 + j^2)^(-(nu + 1/2))
#
# where:
#   nu     = Matérn smoothness parameter
#   ell    = spatial range parameter
#   kappa  = sqrt(2*nu) / ell
#
# TEMPORAL PERSISTENCE:
#   rho_j = exp(-lambda * j^alpha)
#
# NOTE:
#   This is a good first simulation model because it is simple,
#   stable, and easy to understand.
############################################################

# ----------------------------------------------------------
# 1. Load package
# ----------------------------------------------------------
# MASS::mvrnorm is a standard function for simulating
# multivariate normal random vectors.[web:76][web:84]
library(MASS)

# ----------------------------------------------------------
# 2. Helper: Matérn-like Fourier variances
# ----------------------------------------------------------
# This function creates the mode variances sigma_j^2 for j=0,...,J.
#
# Input:
#   J         = highest Fourier mode
#   nu        = Matérn smoothness
#   ell       = range parameter
#   total_var = target total marginal variance scale
#
# Output:
#   vector of length J+1 containing sigma_j^2 for j=0,...,J
#
# Interpretation:
#   - larger nu  => faster decay in j => smoother spatial fields
#   - larger ell => more low-frequency dominance => longer spatial range
matern_spectrum <- function(J, nu, ell, total_var = 1) {
  kappa <- sqrt(2 * nu) / ell
  j <- 0:J
  
  raw_spec <- (kappa^2 + j^2)^(-(nu + 0.5))
  
  # Normalize so the variances sum to total_var.
  # This is a convenient finite-J normalization for simulation.
  sigma2 <- total_var * raw_spec / sum(raw_spec)
  
  return(sigma2)
}

# ----------------------------------------------------------
# 3. Helper: temporal persistence rho_j -- POWER-LAW VERSION
# ----------------------------------------------------------
# rho_j = 1 / (1 + lambda * j^alpha)^beta
#
# This gives ALGEBRAIC (power-law) decay:
#   rho_j ~ j^(-alpha * beta) for large j
#
# Compared to exponential rho_j = exp(-lambda * j^alpha):
#   - power-law decays SLOWER at high j
#   - gives longer temporal memory to high frequencies
#   - more realistic for many physical systems
rho_function <- function(J, lambda, alpha, beta, rho0 = 0.95) {
  j <- 0:J
  rho <- 1 / (1 + lambda * (j^alpha))^beta
  rho[1] <- rho0   # R indexing: first entry is j=0
  return(pmax(rho, 0))  # ensure non-negative for safety
}

# ----------------------------------------------------------
# 4. Helper: build stationary innovation variances
# ----------------------------------------------------------
# For a scalar AR(1):
#   Z_{t+1} = rho Z_t + eps_t
# with stationary variance Var(Z_t) = sigma^2,
# the innovation variance must be
#   Var(eps_t) = sigma^2 * (1 - rho^2).
#
# For j >= 1, both cosine and sine coefficients get the same variance.
innovation_variances <- function(sigma2, rho) {
  eps_var <- sigma2 * (1 - rho^2)
  return(eps_var)
}

# ----------------------------------------------------------
# 5. Helper: simulate Fourier coefficients through time (ALGEBRAIC VERSION)
# ----------------------------------------------------------
simulate_fourier_ar1 <- function(T, J, nu, ell, lambda, alpha, beta,
                                 total_var = 1,
                                 rho0 = 0.95,
                                 seed = NULL) {
  
  if (!is.null(seed)) set.seed(seed)
  
  # Get mode variances sigma_j^2
  sigma2 <- matern_spectrum(J = J, nu = nu, ell = ell, total_var = total_var)
  
  # Get mode-specific AR coefficients -- POWER-LAW VERSION
  rho <- rho_function(J = J, lambda = lambda, alpha = alpha, beta = beta, rho0 = rho0)
  
  # Innovation variances chosen so that each mode has stationary variance sigma_j^2
  eps_var <- innovation_variances(sigma2 = sigma2, rho = rho)
  
  # Allocate storage
  a <- matrix(0, nrow = T, ncol = J + 1)  # cosine coefficients
  b <- matrix(0, nrow = T, ncol = J + 1)  # sine coefficients
  
  # Initial state from stationary distributions
  a[1, 1] <- rnorm(1, mean = 0, sd = sqrt(sigma2[1]))  # j = 0
  
  for (j in 1:J) {
    idx <- j + 1
    a[1, idx] <- rnorm(1, mean = 0, sd = sqrt(sigma2[idx]))
    b[1, idx] <- rnorm(1, mean = 0, sd = sqrt(sigma2[idx]))
  }
  
  # Forward simulation
  for (t in 2:T) {
    # Mode j = 0
    a[t, 1] <- rho[1] * a[t - 1, 1] + rnorm(1, 0, sqrt(eps_var[1]))
    
    # Modes j = 1,...,J
    for (j in 1:J) {
      idx <- j + 1
      a[t, idx] <- rho[idx] * a[t - 1, idx] + rnorm(1, 0, sqrt(eps_var[idx]))
      b[t, idx] <- rho[idx] * b[t - 1, idx] + rnorm(1, 0, sqrt(eps_var[idx]))
    }
  }
  
  return(list(
    a = a, b = b, sigma2 = sigma2, rho = rho, eps_var = eps_var,
    params = list(T = T, J = J, nu = nu, ell = ell,
                  lambda = lambda, alpha = alpha, beta = beta,
                  total_var = total_var, rho0 = rho0)
  ))
}

# ----------------------------------------------------------
# 6. Helper: reconstruct the spatial field on a grid
# ----------------------------------------------------------
# Given coefficients at all times, reconstruct X_t(x)
# on a chosen spatial grid x_grid.
#
# Output:
#   matrix X of size T x length(x_grid)
#   each row = one time point
reconstruct_field <- function(a, b, x_grid) {
  T <- nrow(a)
  J <- ncol(a) - 1
  nx <- length(x_grid)
  
  X <- matrix(0, nrow = T, ncol = nx)
  
  # Start with j = 0 constant mode
  X <- X + a[, 1]
  
  # Add cosine and sine modes
  for (j in 1:J) {
    idx <- j + 1
    X <- X + a[, idx] %*% t(cos(j * x_grid)) +
             b[, idx] %*% t(sin(j * x_grid))
  }
  
  return(X)
}

# ----------------------------------------------------------
# 7. Helper: amplitude and phase for each mode
# ----------------------------------------------------------
# For j >= 1, each pair (a_j, b_j) can be rewritten as
#   R_j cos(jx - phi_j)
#
# where:
#   R_j   = sqrt(a_j^2 + b_j^2)
#   phi_j = atan2(b_j, a_j)
#
# This is often useful for interpretation.
amplitude_phase <- function(a, b) {
  J <- ncol(a) - 1
  
  amp <- a
  phase <- a
  
  # j = 0: amplitude is just |a_0|, phase is not meaningful
  amp[, 1] <- abs(a[, 1])
  phase[, 1] <- NA
  
  for (j in 1:J) {
    idx <- j + 1
    amp[, idx] <- sqrt(a[, idx]^2 + b[, idx]^2)
    phase[, idx] <- atan2(b[, idx], a[, idx])
  }
  
  return(list(amplitude = amp, phase = phase))
}

# ----------------------------------------------------------
# 8. Example: simulate one dataset
# ----------------------------------------------------------
# Suggested first values:
#   nu      = 0.5, 1.5, 2.5 are all worth trying
#   ell     = larger => smoother / longer-range
#   lambda  = larger => faster temporal decorrelation
#   alpha   = larger => stronger frequency dependence in time
sim <- simulate_fourier_ar1(
  T = 200,         # number of time points
  J = 50,          # truncation level
  nu = 1.5,        # Matérn smoothness
  ell = 1.0,       # spatial range
  lambda = 0.01,   # temporal decay strength
  alpha = 1.2,     # frequency dependence exponent
  beta = 1.0,      # power-law exponent
  total_var = 1,
  rho0 = 0.95,
  seed = 123
)

# Spatial grid on [0, 2*pi]
x_grid <- seq(0, 2 * pi, length.out = 200)

# Reconstruct the field
X <- reconstruct_field(sim$a, sim$b, x_grid)

# Amplitude / phase representation
ap <- amplitude_phase(sim$a, sim$b)

# ----------------------------------------------------------
# 9. Quick diagnostics
# ----------------------------------------------------------

# Specify output directory for plots
out_dir <- "Stats 669/research/img/power_rho"  
dir.create(out_dir, showWarnings = FALSE)

# (a) Plot the Matérn-like mode variances sigma_j^2
png(file.path(out_dir, "matern_variances.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$sigma2, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(sigma[j]^2),
     main = expression("Matérn-like Fourier variances " * sigma[j]^2))
dev.off()

# (b) Plot rho_j across frequency
png(file.path(out_dir, "temporal_persistence.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$rho, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(rho[j]),
     main = expression("Mode-specific temporal persistence " * rho[j]))
dev.off()

# (c) Plot the field at a few time points
png(file.path(out_dir, "spatial_field_at_times.png"), width = 800, height = 600)
matplot(x_grid, t(X[c(1, 10, 50, 100, 150, 200), ]),
        type = "l", lty = 1, lwd = 2,
        xlab = "x", ylab = expression(X[t](x)),
        main = "Spatial field at selected times")

legend("topright",
       legend = paste("t =", c(1, 10, 50, 100, 150, 200)),
       col = 1:6, lty = 1, lwd = 2, bty = "n")
dev.off()

# (d) Show an image plot of the full spatio-temporal field
png(file.path(out_dir, "spatio_temporal_field.png"), width = 800, height = 600)
image(x = x_grid, y = 1:nrow(X), z = X,
      xlab = "space x", ylab = "time t",
      main = "Simulated spatio-temporal field",
      col = hcl.colors(100, "YlGnBu", rev = TRUE))
dev.off()

# (e) Look at amplitude over time for a few modes
png(file.path(out_dir, "amplitude_trajectories.png"), width = 800, height = 600)
matplot(1:nrow(ap$amplitude), ap$amplitude[, c(2, 6, 11, 21)],
        type = "l", lty = 1, lwd = 2,
        xlab = "time t", ylab = "Amplitude",
        main = "Amplitude trajectories for selected modes")

legend("topright",
       legend = c("j=1", "j=5", "j=10", "j=20"),
       col = 1:4, lty = 1, lwd = 2, bty = "n")
dev.off()


# (f) Look at phase over time for the same modes
png(file.path(out_dir, "phase_trajectories.png"), width = 800,
    height = 600)
matplot(1:nrow(ap$phase), ap$phase[, c(2, 6, 11, 21)],
        type = "l", lty = 1, lwd = 2,
        xlab = "time t", ylab = "Phase (radians)",
        main = "Phase trajectories for selected modes")
legend("topright",
        legend = c("j=1", "j=5", "j=10", "j=20"),
        col = 1:4, lty = 1, lwd = 2, bty = "n")
dev.off()

# (g) Look at the temporal autocorrelation of a few modes
png(file.path(out_dir, "temporal_autocorrelation.png"), width = 800,
    height = 600)
par(mfrow = c(2, 2))
for (j in c(1, 5, 10, 20)) {
  idx <- j + 1
  acf(ap$amplitude[, idx], main = paste("ACF of amplitude for j =", j))
}
dev.off()

# (h) movie of the spatial field evolving through time
# This is a bit more involved, but can be done using the 'animation' package
# install.packages("animation")
library(animation)
dir.create(file.path(out_dir, "movie"), showWarnings = FALSE)
png(file.path(out_dir, "movie", "field_evolution_%03d.png"), width = 800, height = 600)
for (t in 1:nrow(X)) {
  plot(x_grid, X[t, ], type = "l", lwd = 2,
       xlab = "space x", ylab = expression(X[t](x)),
       main = paste("Spatio-temporal field at time t =", t))
}
dev.off()


library(magick)

# Read all PNG files in a directory
png_files <- list.files(file.path(out_dir, "movie"), pattern = "\\.png$", full.names = TRUE)

# Read and combine images into single magick object
images <- image_read(png_files)

# Animate with delay between frames (fps = frames per second)
gif_animated <- image_animate(images, fps = 10)

# Save as GIF
image_write(gif_animated, file.path(out_dir, "field_evolution.gif"))

# ------------ Debug --------------------
