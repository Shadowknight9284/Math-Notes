############################################################
# FOURIER-AR(1) SPATIO-TEMPORAL MODEL ON A PERIODIC DOMAIN
# WITH PHASE ROTATION AND CORRELATED INNOVATIONS
#
# --------------------------------------------------------
# MODEL:
#   X_t(x) = a_0(t)
#            + sum_{j=1}^J [ a_j(t) cos(j x) + b_j(t) sin(j x) ]
#
# --------------------------------------------------------
# TEMPORAL DYNAMICS (block-diagonal AR(1)):
#
#   For j = 0 (constant / mean mode):
#     a_0(t+1) = rho_0 * a_0(t) + eta_{0,t}
#
#   For j >= 1 (each frequency evolves as a 2x2 AR(1)):
#
#     [a_j(t+1)]   = M_j * [a_j(t)] + [eta_{a,j,t}]
#     [b_j(t+1)]           [b_j(t)]   [eta_{b,j,t}]
#
#   where the TRANSITION MATRIX is a rotation-scaling:
#
#     M_j = rho_j * [ cos(theta_j)  -sin(theta_j) ]
#                   [ sin(theta_j)   cos(theta_j) ]
#
#     theta_j != 0  => a_j(t+1) depends on BOTH a_j(t) AND b_j(t)
#                   => phase rotation: patterns propagate around circle
#     theta_j  = 0  => cosine and sine evolve completely independently
#
#   and the INNOVATION COVARIANCE is:
#
#     Cov([eta_{a,j}, eta_{b,j}]) = sigma_j^2 * (1 - rho_j^2) * [1      phi_j]
#                                                                  [phi_j  1    ]
#
#     phi_j != 0  => eta_{a,j,t} and eta_{b,j,t} are correlated
#                 => Cov(a_j(t), b_j(t)) = sigma_j^2 * phi_j at stationarity
#     phi_j  = 0  => innovations are independent (no instantaneous correlation)
#
# --------------------------------------------------------
# SPATIAL SPECTRUM (Matern-like):
#   sigma_j^2 proportional to (kappa^2 + j^2)^(-(nu + 1/2))
#   kappa = sqrt(2*nu) / ell
#
# --------------------------------------------------------
# TEMPORAL PERSISTENCE (algebraic / power-law decay):
#   rho_j = 1 / (1 + lambda * j^alpha)^beta
#
#   - Slower decay than exponential; more realistic for physical systems
#   - For large j: rho_j ~ j^(-alpha * beta)
#   - Smaller beta => longer temporal memory at all frequencies
#
# --------------------------------------------------------
# NONSEPARABILITY:
#   Because rho_j depends on j, high-frequency modes decorrelate
#   faster in time than low-frequency modes. The space-time
#   covariance C(h,k) = sum_j sigma_j^2 * rho_j^|k| * cos(j*h)
#   cannot be written as C_space(h) * C_time(k), so the model
#   is genuinely nonseparable.
#
# --------------------------------------------------------
# KEY IDENTITIES:
#   At stationarity (from the Lyapunov equation):
#     Var(a_j(t))         = sigma_j^2
#     Var(b_j(t))         = sigma_j^2
#     Cov(a_j(t), b_j(t)) = sigma_j^2 * phi_j
#
# --------------------------------------------------------
# PARAMETERS:
#   J          : Fourier truncation (highest mode)
#   T          : number of time steps
#   nu         : Matern smoothness (larger => smoother spatial fields)
#   ell        : spatial range (larger => longer-range correlation)
#   total_var  : total marginal variance of X_t(x)
#   lambda     : overall temporal decay strength
#   alpha      : frequency exponent for persistence decay
#   beta       : power-law exponent for persistence decay
#   rho0       : AR(1) persistence for the j=0 constant mode
#   theta_vec  : vector of rotation angles theta_j, j=1,...,J
#   phi_vec    : vector of innovation correlations phi_j, j=1,...,J
############################################################

library(MASS)   # for mvrnorm(): multivariate normal sampling

# ============================================================
# SECTION 1: HELPER FUNCTIONS
# ============================================================

# ------------------------------------------------------------
# 1a. matern_spectrum()
#
# Computes the Matern-like mode variances sigma_j^2 for j=0,...,J.
#
# The spectral density of a 1D Matern covariance with smoothness nu
# and range ell is proportional to (kappa^2 + omega^2)^(-(nu + 1/2)),
# where kappa = sqrt(2*nu) / ell. We evaluate this at integer
# frequencies j = 0, 1, ..., J and normalize so the sum equals
# total_var (a convenient finite-J normalization for simulation).
#
# Effect of parameters on realized fields:
#   - larger nu  => faster spectral decay => smoother spatial fields
#   - larger ell => kappa smaller => more mass at low j => longer range
# ------------------------------------------------------------
matern_spectrum <- function(J, nu, ell, total_var = 1) {
  kappa    <- sqrt(2 * nu) / ell
  j        <- 0:J
  raw_spec <- (kappa^2 + j^2)^(-(nu + 0.5))
  sigma2   <- total_var * raw_spec / sum(raw_spec)
  return(sigma2)
}

# ------------------------------------------------------------
# 1b. rho_function()
#
# Computes the mode-specific AR(1) persistence rho_j for j=0,...,J.
#
# We use algebraic (power-law) decay:
#   rho_j = 1 / (1 + lambda * j^alpha)^beta    for j >= 1
#   rho_0 = rho0                                (set by hand)
#
# For large j: rho_j ~ j^(-alpha * beta)
#
# Why power-law instead of exponential exp(-lambda * j^alpha)?
#   Power-law decays more slowly at high j, giving longer temporal
#   memory to fine-scale spatial features. This is common in physical
#   systems (turbulence, climate, geophysics).
#
# Parameter effects:
#   - lambda larger => faster overall decay => less temporal persistence
#   - alpha larger  => stronger frequency sensitivity of decay
#   - beta larger   => steeper power-law => less memory at high j
# ------------------------------------------------------------
rho_function <- function(J, lambda, alpha, beta, rho0 = 0.95) {
  j   <- 0:J
  rho <- 1 / (1 + lambda * j^alpha)^beta
  rho[1] <- rho0      # override j=0; formula gives rho=1 which is nonstationary
  return(pmax(rho, 0))
}

# ------------------------------------------------------------
# 1c. build_theta_vec() and build_phi_vec()
#
# Convenience functions to build the rotation angles theta_j and
# the innovation correlation parameters phi_j for j=1,...,J.
#
# Default parameterizations:
#   theta_j = theta0 / j    (larger rotation for lower frequencies)
#   phi_j   = phi0 / (1+j)  (stronger correlation for lower frequencies)
#
# You can replace these with any frequency-dependent function you like.
# Setting theta0 = 0 or phi0 = 0 recovers the simpler version.
# ------------------------------------------------------------
build_theta_vec <- function(J, theta0 = 0.3) {
  # theta_j = theta0 / j for j = 1,...,J
  # Larger theta0 => more phase rotation per time step
  j <- 1:J
  return(theta0 / j)
}

build_phi_vec <- function(J, phi0 = 0.5) {
  # phi_j = phi0 / (1 + j) for j = 1,...,J
  # |phi_j| < 1 is required for positive-definite innovation covariance
  # Larger phi0 => stronger instantaneous correlation between a_j and b_j
  j <- 1:J
  phi_vec <- phi0 / (1 + j)
  if (any(abs(phi_vec) >= 1)) stop("phi_j must satisfy |phi_j| < 1 for all j")
  return(phi_vec)
}

# ------------------------------------------------------------
# 1d. build_Mj()
#
# Builds the 2x2 transition matrix M_j for mode j:
#
#   M_j = rho_j * [ cos(theta_j)  -sin(theta_j) ]
#                 [ sin(theta_j)   cos(theta_j) ]
#
# This is rho_j times a rotation matrix. Its eigenvalues are
# rho_j * exp(+/- i * theta_j), with modulus rho_j < 1, so
# the process is stationary regardless of theta_j.
#
# When theta_j = 0:  M_j = rho_j * I_2  (independent evolution)
# When theta_j != 0: a_j(t+1) mixes with b_j(t) and vice versa
# ------------------------------------------------------------
build_Mj <- function(rho_j, theta_j) {
  matrix(
    rho_j * c(cos(theta_j), sin(theta_j), -sin(theta_j), cos(theta_j)),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

# ------------------------------------------------------------
# 1e. build_Sigma_eps_j()
#
# Builds the 2x2 innovation covariance Sigma_{eps,j} for mode j:
#
#   Sigma_{eps,j} = sigma_j^2 * (1 - rho_j^2) * [1      phi_j]
#                                                  [phi_j  1    ]
#
# This is chosen so that the stationary covariance of (a_j, b_j)
# satisfies the Lyapunov equation  Sigma_j = M_j Sigma_j M_j' + Sigma_{eps,j},
# with stationary solution:
#
#   Sigma_j = sigma_j^2 * [1      phi_j]
#                          [phi_j  1    ]
#
# Proof:
#   M_j * Sigma_j * M_j' = rho_j^2 * sigma_j^2 * [1 phi_j; phi_j 1]
#   Adding Sigma_{eps,j} = sigma_j^2*(1-rho_j^2)*[1 phi_j; phi_j 1]
#   gives sigma_j^2 * [1 phi_j; phi_j 1] = Sigma_j  ✓
#
# Note: the factor (1-rho_j^2) is the same as the scalar AR(1) case.
# Note: phi_j must satisfy |phi_j| < 1 for Sigma_{eps,j} to be PD.
# ------------------------------------------------------------
build_Sigma_eps_j <- function(sigma2_j, rho_j, phi_j) {
  scale <- sigma2_j * (1 - rho_j^2)
  matrix(
    scale * c(1, phi_j, phi_j, 1),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

# ------------------------------------------------------------
# 1f. build_Sigma_init_j()
#
# Builds the 2x2 stationary (initial) covariance for (a_j(1), b_j(1)):
#
#   Sigma_{init,j} = sigma_j^2 * [1      phi_j]
#                                  [phi_j  1    ]
#
# We initialize the Fourier coefficients from the stationary distribution
# so that the simulated process is exactly stationary from t=1.
# ------------------------------------------------------------
build_Sigma_init_j <- function(sigma2_j, phi_j) {
  matrix(
    sigma2_j * c(1, phi_j, phi_j, 1),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}


# ============================================================
# SECTION 2: MAIN SIMULATION FUNCTION
# ============================================================

# ------------------------------------------------------------
# simulate_fourier_ar1()
#
# Simulates the full spatio-temporal process X_t(x) by:
#   (1) computing sigma_j^2 (Matern-like spectrum)
#   (2) computing rho_j (power-law persistence)
#   (3) building M_j (rotation-scaling matrices)
#   (4) building Sigma_{eps,j} (correlated innovation covariances)
#   (5) initializing from the stationary distribution
#   (6) iterating the AR(1) forward in time
#
# Arguments:
#   T         : number of time steps
#   J         : Fourier truncation level
#   nu        : Matern smoothness
#   ell       : spatial range
#   lambda    : temporal decay strength
#   alpha     : frequency exponent for persistence
#   beta      : power-law exponent for persistence
#   total_var : total marginal variance
#   rho0      : AR(1) persistence for j=0 mode
#   theta_vec : length-J vector of rotation angles theta_j for j=1,...,J
#               Default: theta0/j parameterization via build_theta_vec()
#   phi_vec   : length-J vector of innovation correlations phi_j for j=1,...,J
#               Default: phi0/(1+j) parameterization via build_phi_vec()
#   seed      : optional random seed for reproducibility
#
# Returns a list with:
#   a         : T x (J+1) matrix of cosine coefficients
#   b         : T x (J+1) matrix of sine coefficients (b[,1] unused for j=0)
#   sigma2    : length-(J+1) vector of mode variances
#   rho       : length-(J+1) vector of AR(1) persistences
#   theta_vec : length-J vector of rotation angles
#   phi_vec   : length-J vector of innovation correlations
#   params    : list of all input parameters (for bookkeeping)
# ------------------------------------------------------------
simulate_fourier_ar1 <- function(T, J, nu, ell, lambda, alpha, beta,
                                 total_var = 1,
                                 rho0      = 0.95,
                                 theta_vec = build_theta_vec(J, theta0 = 0.3),
                                 phi_vec   = build_phi_vec(J, phi0 = 0.5),
                                 seed      = NULL) {

  if (!is.null(seed)) set.seed(seed)

  # Sanity checks
  stopifnot(length(theta_vec) == J)
  stopifnot(length(phi_vec) == J)
  stopifnot(all(abs(phi_vec) < 1))
  stopifnot(rho0 > 0 && rho0 < 1)

  # --------------------------------------------------
  # Step 1: Matern-like mode variances sigma_j^2
  # --------------------------------------------------
  sigma2 <- matern_spectrum(J = J, nu = nu, ell = ell, total_var = total_var)

  # --------------------------------------------------
  # Step 2: power-law persistence rho_j
  # --------------------------------------------------
  rho <- rho_function(J = J, lambda = lambda, alpha = alpha, beta = beta, rho0 = rho0)

  # --------------------------------------------------
  # Step 3: allocate storage
  # a[t, j+1] = cosine coefficient for mode j at time t
  # b[t, j+1] = sine coefficient for mode j at time t
  # (b[, 1] is never used since there is no sin(0*x) term)
  # --------------------------------------------------
  a <- matrix(0, nrow = T, ncol = J + 1)
  b <- matrix(0, nrow = T, ncol = J + 1)

  # --------------------------------------------------
  # Step 4: initialize at t=1 from the stationary distribution
  # --------------------------------------------------

  # j = 0: scalar, stationary variance = sigma2[1]
  a[1, 1] <- rnorm(1, mean = 0, sd = sqrt(sigma2[1]))

  # j = 1,...,J: bivariate, stationary covariance = Sigma_{init,j}
  for (j in 1:J) {
    idx         <- j + 1
    Sigma_init  <- build_Sigma_init_j(sigma2[idx], phi_vec[j])
    draw        <- mvrnorm(1, mu = c(0, 0), Sigma = Sigma_init)
    a[1, idx]   <- draw[1]
    b[1, idx]   <- draw[2]
  }

  # --------------------------------------------------
  # Step 5: forward simulation t = 2, ..., T
  # --------------------------------------------------
  for (t in 2:T) {

    # --- j = 0: scalar AR(1), no rotation, no phi ---
    # a_0(t+1) = rho_0 * a_0(t) + eta_{0,t}
    # eta_{0,t} ~ N(0, sigma_0^2 * (1 - rho_0^2))
    eps_var_0 <- sigma2[1] * (1 - rho[1]^2)
    a[t, 1]   <- rho[1] * a[t - 1, 1] + rnorm(1, 0, sqrt(eps_var_0))

    # --- j = 1,...,J: 2x2 AR(1) with rotation and correlated innovations ---
    for (j in 1:J) {
      idx <- j + 1

      # Build transition matrix M_j = rho_j * R(theta_j)
      Mj <- build_Mj(rho[idx], theta_vec[j])

      # Build innovation covariance Sigma_{eps,j}
      Sigma_eps <- build_Sigma_eps_j(sigma2[idx], rho[idx], phi_vec[j])

      # Draw correlated innovation vector
      eta <- mvrnorm(1, mu = c(0, 0), Sigma = Sigma_eps)

      # Update: [a_j; b_j](t) = M_j * [a_j; b_j](t-1) + eta
      new_state  <- Mj %*% c(a[t - 1, idx], b[t - 1, idx]) + eta
      a[t, idx]  <- new_state[1]
      b[t, idx]  <- new_state[2]
    }
  }

  return(list(
    a         = a,
    b         = b,
    sigma2    = sigma2,
    rho       = rho,
    theta_vec = theta_vec,
    phi_vec   = phi_vec,
    params    = list(
      T = T, J = J, nu = nu, ell = ell,
      lambda = lambda, alpha = alpha, beta = beta,
      total_var = total_var, rho0 = rho0
    )
  ))
}


# ============================================================
# SECTION 3: POST-PROCESSING HELPERS
# ============================================================

# ------------------------------------------------------------
# reconstruct_field()
#
# Reconstructs the spatial field X_t(x) on a grid from the
# Fourier coefficient arrays a and b.
#
# X_t(x) = a_0(t) + sum_{j=1}^J [a_j(t) cos(jx) + b_j(t) sin(jx)]
#
# Uses matrix operations for speed:
#   a[, idx] is a T-vector; cos(j * x_grid) is an N-vector.
#   a[, idx] %*% t(cos(j * x_grid)) gives a T x N outer product.
#
# Returns:
#   X : T x N matrix, X[t, n] = X_t(x_n)
# ------------------------------------------------------------
reconstruct_field <- function(a, b, x_grid) {
  T  <- nrow(a)
  J  <- ncol(a) - 1
  X  <- matrix(a[, 1], nrow = T, ncol = length(x_grid))   # j=0 constant mode

  for (j in 1:J) {
    idx <- j + 1
    X   <- X +
           a[, idx] %*% t(cos(j * x_grid)) +
           b[, idx] %*% t(sin(j * x_grid))
  }
  return(X)
}

# ------------------------------------------------------------
# amplitude_phase()
#
# For each mode j >= 1, the Fourier pair (a_j, b_j) can be written as
#   a_j cos(jx) + b_j sin(jx) = R_j cos(jx - psi_j)
#
# where:
#   R_j   = sqrt(a_j^2 + b_j^2)   -- amplitude (always >= 0)
#   psi_j = atan2(b_j, a_j)       -- phase angle (in [-pi, pi])
#
# The amplitude tells us how strongly frequency j is contributing
# to the field at time t.
# The phase tells us where the crest of frequency j is located.
#
# With theta_j != 0, the phase drifts over time (phase rotation).
#
# Returns:
#   amplitude : T x (J+1) matrix, amplitude[t, j+1] = R_j(t)
#   phase     : T x (J+1) matrix, phase[t, j+1]     = psi_j(t)
#               (first column is NA since j=0 has no phase)
# ------------------------------------------------------------
amplitude_phase <- function(a, b) {
  J   <- ncol(a) - 1
  amp <- matrix(NA, nrow = nrow(a), ncol = J + 1)
  phs <- matrix(NA, nrow = nrow(a), ncol = J + 1)

  amp[, 1] <- abs(a[, 1])   # j=0: amplitude is |a_0|, phase is undefined

  for (j in 1:J) {
    idx        <- j + 1
    amp[, idx] <- sqrt(a[, idx]^2 + b[, idx]^2)
    phs[, idx] <- atan2(b[, idx], a[, idx])
  }

  return(list(amplitude = amp, phase = phs))
}

# ------------------------------------------------------------
# empirical_cov_aj_bj()
#
# Computes the empirical (sample) covariance between a_j(t) and b_j(t)
# across all time points for each mode j = 1,...,J.
#
# Under the model, the theoretical value is:
#   Cov(a_j(t), b_j(t)) = sigma_j^2 * phi_j
#
# This is a diagnostic function: comparing the empirical covariance
# to sigma2[j] * phi_j[j] checks that the correlated innovations
# are working correctly.
#
# Returns:
#   cov_empirical : length-J vector of empirical covariances for j=1,...,J
#   cov_theory    : length-J vector of theoretical covariances sigma_j^2 * phi_j
# ------------------------------------------------------------
empirical_cov_aj_bj <- function(a, b, sigma2, phi_vec) {
  J             <- length(phi_vec)
  cov_empirical <- numeric(J)
  cov_theory    <- numeric(J)

  for (j in 1:J) {
    idx               <- j + 1
    cov_empirical[j]  <- cov(a[, idx], b[, idx])
    cov_theory[j]     <- sigma2[idx] * phi_vec[j]
  }

  return(data.frame(
    j             = 1:J,
    cov_empirical = cov_empirical,
    cov_theory    = cov_theory
  ))
}


# ============================================================
# SECTION 4: RUN THE SIMULATION
# ============================================================

# --------------------------------------------------
# Parameter choices:
#   J = 50   : truncation at 50 Fourier modes
#   T = 200  : 200 time steps
#   nu = 1.5 : Matern smoothness (fields are once mean-square differentiable)
#   ell = 1.0: spatial range
#   lambda = 0.01, alpha = 1.2, beta = 1.0 : power-law persistence decay
#   rho0 = 0.95  : mean mode persistence
#   theta0 = 0.3 : rotation strength (theta_j = 0.3 / j)
#   phi0 = 0.5   : innovation correlation strength (phi_j = 0.5 / (1 + j))
# --------------------------------------------------
sim <- simulate_fourier_ar1(
  T         = 200,
  J         = 50,
  nu        = 1.5,
  ell       = 1.0,
  lambda    = 0.01,
  alpha     = 1.2,
  beta      = 1.0,
  total_var = 1,
  rho0      = 0.95,
  theta_vec = build_theta_vec(J = 50, theta0 = 0.3),
  phi_vec   = build_phi_vec(J = 50, phi0 = 0.5),
  seed      = 123
)

# Spatial grid on [0, 2*pi]
x_grid <- seq(0, 2 * pi, length.out = 200)

# Reconstruct T x N spatial field matrix
X <- reconstruct_field(sim$a, sim$b, x_grid)

# Amplitude and phase for each mode over time
ap <- amplitude_phase(sim$a, sim$b)

# Empirical vs theoretical covariance check
cov_check <- empirical_cov_aj_bj(sim$a, sim$b, sim$sigma2, sim$phi_vec)
cat("Covariance check (first 10 modes):\n")
print(head(cov_check, 10))


# ============================================================
# SECTION 5: DIAGNOSTIC PLOTS
# ============================================================

out_dir <- "Stats 669/research/img/crosscorr"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------
# (a) Matern-like mode variances sigma_j^2
#   Shows how variance is distributed across spatial frequencies.
#   We expect rapid decay: most variance in low-j (large-scale) modes.
# --------------------------------------------------
png(file.path(out_dir, "matern_variances.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$sigma2, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(sigma[j]^2),
     main = expression("Matern-like Fourier variances " * sigma[j]^2))
dev.off()

# --------------------------------------------------
# (b) Mode-specific temporal persistence rho_j
#   Shows how quickly each spatial frequency forgets its past.
#   High-j modes have smaller rho_j => decorrelate faster in time.
#   This is the key source of nonseparability.
# --------------------------------------------------
png(file.path(out_dir, "temporal_persistence.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$rho, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(rho[j]),
     main = expression("Mode-specific temporal persistence " * rho[j]))
dev.off()

# --------------------------------------------------
# (c) Rotation angles theta_j and innovation correlations phi_j
#   New diagnostic for the two new model features.
# --------------------------------------------------
png(file.path(out_dir, "theta_phi_profiles.png"), width = 800, height = 600)
par(mfrow = c(1, 2))

plot(1:sim$params$J, sim$theta_vec, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(theta[j]),
     main = expression("Rotation angles " * theta[j]))
abline(h = 0, lty = 2, col = "gray")

plot(1:sim$params$J, sim$phi_vec, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(phi[j]),
     main = expression("Innovation correlations " * phi[j]))
abline(h = 0, lty = 2, col = "gray")

par(mfrow = c(1, 1))
dev.off()

# --------------------------------------------------
# (d) Empirical vs theoretical Cov(a_j, b_j)
#   This directly verifies that phi_j is working as expected.
#   Empirical values should match sigma_j^2 * phi_j closely for large T.
# --------------------------------------------------
png(file.path(out_dir, "cov_aj_bj_check.png"), width = 800, height = 600)
plot(cov_check$j, cov_check$cov_theory, type = "l", lwd = 2,
     col = "steelblue",
     xlab = "Fourier mode j",
     ylab = expression(Cov(a[j], b[j])),
     main = expression("Empirical vs theoretical " * Cov(a[j](t), b[j](t))))
lines(cov_check$j, cov_check$cov_empirical, lwd = 2, col = "tomato", lty = 2)
legend("topright",
       legend = c("Theoretical: sigma_j^2 * phi_j", "Empirical"),
       col = c("steelblue", "tomato"), lwd = 2, lty = c(1, 2), bty = "n")
dev.off()

# --------------------------------------------------
# (e) Spatial field at selected time points
#   Each line is a spatial snapshot X_t(x) across x in [0, 2*pi].
#   With theta_j != 0, patterns shift position over time (phase rotation).
# --------------------------------------------------
png(file.path(out_dir, "spatial_field_at_times.png"), width = 800, height = 600)
matplot(x_grid, t(X[c(1, 10, 50, 100, 150, 200), ]),
        type = "l", lty = 1, lwd = 2,
        xlab = "x", ylab = expression(X[t](x)),
        main = "Spatial field at selected time points")
legend("topright",
       legend = paste("t =", c(1, 10, 50, 100, 150, 200)),
       col = 1:6, lty = 1, lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# (f) Full spatio-temporal field as an image
#   x-axis: spatial location; y-axis: time.
#   Reveals space-time patterns: do features drift spatially over time?
#   With theta_j != 0 you should see diagonal streaks (traveling waves).
# --------------------------------------------------
png(file.path(out_dir, "spatio_temporal_field.png"), width = 800, height = 600)
image(x = x_grid, y = 1:nrow(X), z = t(X),
      xlab = "space x", ylab = "time t",
      main = "Simulated spatio-temporal field",
      col = hcl.colors(100, "YlGnBu", rev = TRUE))
dev.off()

# --------------------------------------------------
# (g) Amplitude trajectories for selected modes
#   R_j(t) = sqrt(a_j(t)^2 + b_j(t)^2) over time.
#   Low-j modes vary slowly (large rho_j); high-j modes vary quickly.
# --------------------------------------------------
png(file.path(out_dir, "amplitude_trajectories.png"), width = 800, height = 600)
matplot(1:nrow(ap$amplitude), ap$amplitude[, c(2, 6, 11, 21)],
        type = "l", lty = 1, lwd = 2,
        xlab = "time t", ylab = "Amplitude",
        main = "Amplitude trajectories for selected modes")
legend("topright",
       legend = c("j=1", "j=5", "j=10", "j=20"),
       col = 1:4, lty = 1, lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# (h) Phase trajectories for selected modes
#   psi_j(t) = atan2(b_j(t), a_j(t)) over time.
#   With theta_j != 0, phases should drift systematically -- this is
#   the phase rotation effect. Compare to theta_j = 0 case where
#   phases should look like a random walk / flat.
# --------------------------------------------------
png(file.path(out_dir, "phase_trajectories.png"), width = 800, height = 600)
matplot(1:nrow(ap$phase), ap$phase[, c(2, 6, 11, 21)],
        type = "l", lty = 1, lwd = 2,
        xlab = "time t", ylab = "Phase (radians)",
        main = "Phase trajectories (shows rotation if theta_j != 0)")
abline(h = c(-pi, 0, pi), lty = 2, col = "gray")
legend("topright",
       legend = c("j=1", "j=5", "j=10", "j=20"),
       col = 1:4, lty = 1, lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# (i) Temporal autocorrelation of amplitudes for selected modes
#   ACF of R_j(t) should decay faster for larger j (smaller rho_j).
#   This is the empirical check of nonseparability.
# --------------------------------------------------
png(file.path(out_dir, "temporal_autocorrelation.png"), width = 800, height = 600)
par(mfrow = c(2, 2))
for (j in c(1, 5, 10, 20)) {
  idx <- j + 1
  acf(ap$amplitude[, idx],
      main = paste("ACF of amplitude | j =", j,
                   "| rho_j =", round(sim$rho[idx], 3)))
}
par(mfrow = c(1, 1))
dev.off()

# --------------------------------------------------
# (j) GIF: field evolving through time
#   Requires the magick package. Saves individual frames and combines.
#   With theta_j != 0, you should see features traveling (not just
#   fading in and out), confirming the phase rotation is active.
# --------------------------------------------------
library(magick)
dir.create(file.path(out_dir, "movie"), showWarnings = FALSE)

png(file.path(out_dir, "movie", "field_evolution_%03d.png"), width = 800, height = 600)
ylim_range <- range(X)
for (t in 1:nrow(X)) {
  plot(x_grid, X[t, ], type = "l", lwd = 2,
       ylim = ylim_range,
       xlab = "space x", ylab = expression(X[t](x)),
       main = paste("Spatio-temporal field at time t =", t))
}
dev.off()

png_files    <- list.files(file.path(out_dir, "movie"), pattern = "\\.png$", full.names = TRUE)
images       <- image_read(png_files)
gif_animated <- image_animate(images, fps = 10)
image_write(gif_animated, file.path(out_dir, "field_evolution.gif"))

cat("\nAll plots saved to:", out_dir, "\n")
