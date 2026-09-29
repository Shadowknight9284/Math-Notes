############################################################
# FOURIER-AR(1) SPATIO-TEMPORAL MODEL ON A PERIODIC DOMAIN
# WITH PHASE ROTATION, CORRELATED INNOVATIONS, AND FFT FIELD EVALUATION
#
# --------------------------------------------------------
# MODEL:
# X_t(x) = a_0(t)
#         + sum_{j=1}^J [ a_j(t) cos(j x) + b_j(t) sin(j x) ]
#
# --------------------------------------------------------
# TEMPORAL DYNAMICS (block-diagonal AR(1)):
#
# For j = 0 (constant / mean mode):
#   a_0(t+1) = rho_0 * a_0(t) + eta_{0,t}
#
# For j >= 1 (each frequency evolves as a 2x2 AR(1)):
#
#   [a_j(t+1)] = M_j * [a_j(t)] + [eta_{a,j,t}]
#   [b_j(t+1)]          [b_j(t)]   [eta_{b,j,t}]
#
#   M_j = rho_j * [ cos(theta_j)  -sin(theta_j) ]
#                 [ sin(theta_j)   cos(theta_j) ]
#
#   Cov([eta_{a,j}, eta_{b,j}]) = sigma_j^2 * (1 - rho_j^2) * [1     phi_j]
#                                                               [phi_j 1    ]
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
# --------------------------------------------------------
# SPATIAL FIELD EVALUATION:
#   Two methods are provided:
#
#   (1) reconstruct_field_direct() -- direct summation, O(N*J)
#       Evaluates X_t(x) on any grid (regular or irregular).
#       Simple but slow for large N or J.
#
#   (2) reconstruct_field_fft() -- Inverse FFT, O(N log N)
#       Evaluates X_t(x) on a DENSE REGULAR grid of size N_fft.
#       Requires N_fft >= 4*J (Nyquist) and N_fft a power of 2.
#       Much faster for large N_fft. Can then interpolate to any
#       desired observation points from the dense grid.
#
#   The FFT method packs the Fourier coefficients into a complex
#   vector and uses R's built-in fft() to evaluate all N_fft spatial
#   points simultaneously. The two methods are mathematically
#   equivalent (identical output up to machine precision ~1e-13).
#
# --------------------------------------------------------
# PARAMETERS:
#   J         : Fourier truncation (highest mode)
#   T         : number of time steps
#   nu        : Matern smoothness
#   ell       : spatial range
#   total_var : total marginal variance of X_t(x)
#   lambda    : overall temporal decay strength
#   alpha     : frequency exponent for persistence decay
#   beta      : power-law exponent for persistence decay
#   rho0      : AR(1) persistence for the j=0 constant mode
#   theta_vec : length-J vector of rotation angles theta_j, j=1,...,J
#   phi_vec   : length-J vector of innovation correlations phi_j, j=1,...,J
#   N_fft     : FFT grid size (power of 2; recommended N_fft >= 4*J)
############################################################

library(MASS)  # for mvrnorm(): multivariate normal sampling

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
# total_var.
#
# - larger nu => faster spectral decay => smoother spatial fields
# - larger ell => kappa smaller => more mass at low j => longer range
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
# Algebraic (power-law) decay:
#   rho_j = 1 / (1 + lambda * j^alpha)^beta  for j >= 1
#   rho_0 = rho0  (set by hand; formula gives rho=1, nonstationary)
#
# - lambda larger  => faster overall decay
# - alpha larger   => stronger frequency sensitivity
# - beta larger    => steeper power-law, less high-j memory
# ------------------------------------------------------------
rho_function <- function(J, lambda, alpha, beta, rho0 = 0.95) {
  j   <- 0:J
  rho <- 1 / (1 + lambda * j^alpha)^beta
  rho[1] <- rho0
  return(pmax(rho, 0))
}

# ------------------------------------------------------------
# 1c. build_theta_vec() and build_phi_vec()
#
# Default parameterizations for rotation angles and innovation
# correlations across frequencies:
#   theta_j = theta0 / j       (more rotation at low frequencies)
#   phi_j   = phi0 / (1 + j)   (more coupling at low frequencies)
#
# Setting theta0 = 0 or phi0 = 0 recovers simpler models.
# ------------------------------------------------------------
build_theta_vec <- function(J, theta0 = 0.3) {
  j <- 1:J
  return(theta0 / j)
}

build_phi_vec <- function(J, phi0 = 0.5) {
  j       <- 1:J
  phi_vec <- phi0 / (1 + j)
  if (any(abs(phi_vec) >= 1)) stop("phi_j must satisfy |phi_j| < 1 for all j")
  return(phi_vec)
}

# ------------------------------------------------------------
# 1d. build_Mj()
#
# 2x2 transition (rotation-scaling) matrix for mode j:
#
#   M_j = rho_j * [ cos(theta_j)  -sin(theta_j) ]
#                 [ sin(theta_j)   cos(theta_j) ]
#
# Eigenvalues: rho_j * exp(+/- i * theta_j)
# Modulus = rho_j < 1 => stationarity regardless of theta_j.
# theta_j = 0 => M_j = rho_j * I_2 (independent cosine/sine evolution)
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
# 2x2 innovation covariance for mode j, derived from the
# discrete-time Lyapunov equation:
#
#   Sigma_j = M_j Sigma_j M_j' + Sigma_{eps,j}
#
# Since M_j M_j' = rho_j^2 * I_2, the solution is:
#
#   Sigma_{eps,j} = (1 - rho_j^2) * Sigma_j
#                 = sigma_j^2 * (1 - rho_j^2) * [1     phi_j]
#                                                [phi_j 1    ]
#
# This is independent of theta_j: rotation affects temporal dynamics
# but not the stationary covariance structure.
# Requires |phi_j| < 1 for positive definiteness.
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
# Stationary (initial) covariance for (a_j(1), b_j(1)):
#
#   Sigma_{init,j} = sigma_j^2 * [1     phi_j]
#                                 [phi_j 1    ]
#
# Initializing from this distribution makes the process exactly
# stationary from t = 1.
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
# Simulates the Fourier coefficient arrays a[t, j+1] and b[t, j+1]
# for t = 1,...,T and j = 0,...,J. Does NOT evaluate the spatial
# field; use reconstruct_field_direct() or reconstruct_field_fft()
# for that.
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
#   theta_vec : length-J vector of rotation angles for j=1,...,J
#   phi_vec   : length-J vector of innovation correlations for j=1,...,J
#   seed      : optional RNG seed
#
# Returns a list with:
#   a         : T x (J+1) matrix of cosine coefficients
#   b         : T x (J+1) matrix of sine coefficients (b[,1] unused)
#   sigma2    : length-(J+1) vector of mode variances
#   rho       : length-(J+1) vector of AR(1) persistences
#   theta_vec : length-J rotation angles
#   phi_vec   : length-J innovation correlations
#   params    : all input parameters (bookkeeping)
# ------------------------------------------------------------
simulate_fourier_ar1 <- function(T, J, nu, ell, lambda, alpha, beta,
                                 total_var = 1,
                                 rho0      = 0.95,
                                 theta_vec = build_theta_vec(J, theta0 = 0.3),
                                 phi_vec   = build_phi_vec(J, phi0 = 0.5),
                                 seed      = NULL) {

  if (!is.null(seed)) set.seed(seed)

  stopifnot(length(theta_vec) == J)
  stopifnot(length(phi_vec)   == J)
  stopifnot(all(abs(phi_vec)  <  1))
  stopifnot(rho0 > 0 && rho0  <  1)

  # Step 1: Matern-like mode variances
  sigma2 <- matern_spectrum(J = J, nu = nu, ell = ell, total_var = total_var)

  # Step 2: Power-law persistence
  rho <- rho_function(J = J, lambda = lambda, alpha = alpha, beta = beta, rho0 = rho0)

  # Step 3: Allocate storage
  # a[t, j+1] = cosine coeff for mode j at time t
  # b[t, j+1] = sine coeff   for mode j at time t  (b[,1] unused: no sin(0*x))
  a <- matrix(0, nrow = T, ncol = J + 1)
  b <- matrix(0, nrow = T, ncol = J + 1)

  # Step 4: Initialize at t=1 from stationary distribution
  a[1, 1] <- rnorm(1, mean = 0, sd = sqrt(sigma2[1]))  # j=0: scalar

  for (j in 1:J) {
    idx          <- j + 1
    Sigma_init   <- build_Sigma_init_j(sigma2[idx], phi_vec[j])
    draw         <- mvrnorm(1, mu = c(0, 0), Sigma = Sigma_init)
    a[1, idx]    <- draw[1]
    b[1, idx]    <- draw[2]
  }

  # Step 5: Forward simulation
  for (t in 2:T) {

    # j = 0: scalar AR(1)
    eps_var_0 <- sigma2[1] * (1 - rho[1]^2)
    a[t, 1]   <- rho[1] * a[t - 1, 1] + rnorm(1, 0, sqrt(eps_var_0))

    # j = 1,...,J: 2x2 AR(1) with rotation and correlated innovations
    for (j in 1:J) {
      idx       <- j + 1
      Mj        <- build_Mj(rho[idx], theta_vec[j])
      Sigma_eps <- build_Sigma_eps_j(sigma2[idx], rho[idx], phi_vec[j])
      eta       <- mvrnorm(1, mu = c(0, 0), Sigma = Sigma_eps)
      new_state <- Mj %*% c(a[t - 1, idx], b[t - 1, idx]) + eta
      a[t, idx] <- new_state[1]
      b[t, idx] <- new_state[2]
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
# SECTION 3: SPATIAL FIELD EVALUATION
# ============================================================

# ------------------------------------------------------------
# 3a. reconstruct_field_direct()
#
# Evaluates X_t(x) on any spatial grid x_grid by direct summation:
#
#   X_t(x) = a_0(t) + sum_{j=1}^J [a_j(t) cos(jx) + b_j(t) sin(jx)]
#
# Cost: O(N * J) per time step where N = length(x_grid).
#
# Use this when:
#   - x_grid is irregular (not equally spaced)
#   - N is small
#   - J is small
#   - you want a simple reference implementation
#
# Returns:
#   X : T x N matrix, X[t, n] = X_t(x_grid[n])
# ------------------------------------------------------------
reconstruct_field_direct <- function(a, b, x_grid) {
  T  <- nrow(a)
  J  <- ncol(a) - 1
  N  <- length(x_grid)
  X  <- matrix(a[, 1], nrow = T, ncol = N)  # j=0 constant mode

  for (j in 1:J) {
    idx <- j + 1
    X   <- X +
      a[, idx] %*% t(cos(j * x_grid)) +
      b[, idx] %*% t(sin(j * x_grid))
  }
  return(X)
}

# ------------------------------------------------------------
# 3b. reconstruct_field_fft()
#
# Evaluates X_t(x) on a DENSE REGULAR grid using the Inverse FFT.
#
# The Fourier series evaluated at equally-spaced grid points
#   x_k = 2*pi*k / N_fft,  k = 0, 1, ..., N_fft - 1
# is exactly a discrete Fourier transform (DFT). R's fft() computes
# the DFT in O(N_fft * log(N_fft)) instead of O(N_fft * J).
#
# HOW THE PACKING WORKS:
#   The real Fourier series  a_j cos(jx) + b_j sin(jx)
#   equals the real part of  (a_j - i b_j) e^{ijx}.
#   We therefore form a complex coefficient vector c of length N_fft:
#
#     c[1]          = a_0           (DC / j=0 component)
#     c[j+1]        = (a_j - i*b_j) / 2   for j = 1,...,J  (positive freqs)
#     c[N_fft-j+1]  = (a_j + i*b_j) / 2   for j = 1,...,J  (negative freqs,
#                                                             conjugate symmetry
#                                                             for real output)
#     c[rest]       = 0              (zero-pad; no energy above freq J)
#
#   Then:  X_t evaluated at x_k = Re[ N_fft * IFFT(c) ][k+1]
#
#   R's fft(c, inverse=TRUE) / N_fft computes the IFFT, so we
#   multiply by N_fft to get the correct scale.
#
# NYQUIST CONSTRAINT:
#   N_fft must satisfy N_fft/2 > J, i.e. N_fft > 2*J.
#   Recommended: N_fft >= 4*J (gives 2x safety margin).
#   N_fft should be a power of 2 for maximum FFT efficiency.
#
# ACCURACY:
#   The FFT and direct methods are mathematically identical.
#   Numerical difference is at machine precision (~1e-13).
#
# Arguments:
#   a, b    : T x (J+1) coefficient matrices from simulate_fourier_ar1()
#   N_fft   : FFT grid size (power of 2; must be > 2*J)
#
# Returns:
#   X_fft   : T x N_fft matrix, X_fft[t, k] = X_t(2*pi*(k-1)/N_fft)
#   x_grid  : length-N_fft vector of corresponding spatial locations
# ------------------------------------------------------------
reconstruct_field_fft <- function(a, b, N_fft) {
  T   <- nrow(a)
  J   <- ncol(a) - 1

  # Enforce Nyquist: highest frequency J must be < N_fft / 2
  if (N_fft <= 2 * J) {
    stop(sprintf(
      "N_fft = %d is too small for J = %d. Need N_fft > 2*J = %d. Recommend N_fft >= 4*J = %d.",
      N_fft, J, 2 * J, 4 * J
    ))
  }
  if (log2(N_fft) != floor(log2(N_fft))) {
    warning("N_fft is not a power of 2; FFT performance may be suboptimal.")
  }

  # Output grid: x_k = 2*pi*k/N_fft for k = 0,...,N_fft-1
  x_grid_fft <- 2 * pi * (0:(N_fft - 1)) / N_fft

  # Allocate output
  X_fft <- matrix(0, nrow = T, ncol = N_fft)

  for (t in 1:T) {
    # Allocate complex coefficient vector (all zeros = zero-padding)
    c_vec <- complex(length.out = N_fft)

    # DC component: frequency j = 0
    c_vec[1] <- a[t, 1]

    # Positive frequencies j = 1,...,J
    # and their conjugate-symmetric negative-frequency partners
    for (j in 1:J) {
      idx            <- j + 1
      aj             <- a[t, idx]
      bj             <- b[t, idx]
      c_vec[j + 1]         <- complex(real =  aj / 2, imaginary = -bj / 2)  # pos freq
      c_vec[N_fft - j + 1] <- complex(real =  aj / 2, imaginary =  bj / 2)  # neg freq (conj)
    }

    # Inverse FFT and scale: X = Re( N_fft * IFFT(c) )
    raw          <- fft(c_vec, inverse = TRUE)
    X_fft[t, ]  <- Re(raw) * N_fft / N_fft  # scale cancels: fft(inverse) divides by N
    # Note: R's fft(x, inverse=TRUE) computes sum_k c[k] exp(2*pi*i*j*k/N),
    # which already equals our desired X_t(x_k) without extra scaling.
    # We keep the explicit factor for clarity. Taking Re() removes ~1e-15 imaginary noise.
  }

  return(list(
    X      = X_fft,
    x_grid = x_grid_fft
  ))
}

# ------------------------------------------------------------
# 3c. interpolate_to_obs()
#
# After evaluating on the dense FFT grid, interpolate to arbitrary
# observation points x_obs (which need not be on the regular grid).
#
# Uses linear interpolation (approx()). For higher accuracy with
# smooth fields, cubic spline interpolation (spline()) can be used.
#
# Arguments:
#   fft_result : output list from reconstruct_field_fft()
#   x_obs      : vector of observation locations in [0, 2*pi]
#
# Returns:
#   X_obs : T x length(x_obs) matrix of interpolated field values
# ------------------------------------------------------------
interpolate_to_obs <- function(fft_result, x_obs) {
  T       <- nrow(fft_result$X)
  x_grid  <- fft_result$x_grid
  X_obs   <- matrix(0, nrow = T, ncol = length(x_obs))

  for (t in 1:T) {
    X_obs[t, ] <- approx(
      x     = x_grid,
      y     = fft_result$X[t, ],
      xout  = x_obs,
      rule  = 2  # extrapolate at boundaries using endpoint values
    )$y
  }

  return(X_obs)
}

# ============================================================
# SECTION 4: POST-PROCESSING HELPERS
# ============================================================

# ------------------------------------------------------------
# 4a. amplitude_phase()
#
# For each mode j >= 1, decompose (a_j, b_j) into amplitude and phase:
#   R_j(t)   = sqrt(a_j(t)^2 + b_j(t)^2)   -- amplitude
#   psi_j(t) = atan2(b_j(t), a_j(t))         -- phase angle in [-pi, pi]
#
# With theta_j != 0, psi_j(t) drifts linearly with slope ~theta_j.
# ------------------------------------------------------------
amplitude_phase <- function(a, b) {
  J   <- ncol(a) - 1
  amp <- matrix(NA, nrow = nrow(a), ncol = J + 1)
  phs <- matrix(NA, nrow = nrow(a), ncol = J + 1)

  amp[, 1] <- abs(a[, 1])  # j=0: amplitude = |a_0|, phase undefined

  for (j in 1:J) {
    idx       <- j + 1
    amp[, idx] <- sqrt(a[, idx]^2 + b[, idx]^2)
    phs[, idx] <- atan2(b[, idx], a[, idx])
  }

  return(list(amplitude = amp, phase = phs))
}

# ------------------------------------------------------------
# 4b. empirical_cov_aj_bj()
#
# Computes empirical Cov(a_j(t), b_j(t)) for each mode j = 1,...,J
# and compares to the theoretical value sigma_j^2 * phi_j.
#
# This is the primary diagnostic for the correlated-innovation
# (phi_j) mechanism. Deviations indicate implementation issues.
# ------------------------------------------------------------
empirical_cov_aj_bj <- function(a, b, sigma2, phi_vec) {
  J             <- length(phi_vec)
  cov_empirical <- numeric(J)
  cov_theory    <- numeric(J)

  for (j in 1:J) {
    idx              <- j + 1
    cov_empirical[j] <- cov(a[, idx], b[, idx])
    cov_theory[j]    <- sigma2[idx] * phi_vec[j]
  }

  return(data.frame(
    j             = 1:J,
    cov_empirical = cov_empirical,
    cov_theory    = cov_theory,
    rel_error     = abs(cov_empirical - cov_theory) / (abs(cov_theory) + 1e-10)
  ))
}

# ------------------------------------------------------------
# 4c. validate_fft_accuracy()
#
# Compares the FFT and direct evaluation methods at a few
# randomly selected time steps and observation points.
# Maximum absolute difference should be < 1e-10.
# ------------------------------------------------------------
validate_fft_accuracy <- function(a, b, N_fft, n_check = 5) {
  T      <- nrow(a)
  t_samp <- sample(1:T, n_check)

  # Direct evaluation on the FFT grid (for comparison)
  x_fft  <- 2 * pi * (0:(N_fft - 1)) / N_fft
  X_dir  <- reconstruct_field_direct(a[t_samp, , drop = FALSE],
                                      b[t_samp, , drop = FALSE],
                                      x_fft)

  # FFT evaluation
  a_sub      <- a[t_samp, , drop = FALSE]
  b_sub      <- b[t_samp, , drop = FALSE]
  fft_result <- reconstruct_field_fft(a_sub, b_sub, N_fft)
  X_fft_val  <- fft_result$X

  max_err <- max(abs(X_dir - X_fft_val))
  cat(sprintf("FFT accuracy check: max |direct - FFT| = %.3e  (should be < 1e-10)\n", max_err))
  if (max_err > 1e-8) warning("FFT accuracy may be poor; check N_fft >= 4*J.")
  return(invisible(max_err))
}

# ============================================================
# SECTION 5: RUN THE SIMULATION
# ============================================================

# --------------------------------------------------
# Parameter choices (default / recommended starting point):
#   J      = 50  : truncation at 50 Fourier modes
#   T      = 200 : 200 time steps
#   nu     = 1.5 : Matern smoothness (once mean-square differentiable)
#   ell    = 1.0 : spatial range
#   lambda = 0.01, alpha = 1.2, beta = 1.0 : power-law persistence
#   rho0   = 0.95
#   theta0 = 0.3 : rotation (theta_j = 0.3 / j)
#   phi0   = 0.5 : cosine-sine coupling (phi_j = 0.5 / (1+j))
#   N_fft  = 256 : FFT grid size; 256 >= 4*50 = 200 and is a power of 2
# --------------------------------------------------

J_sim    <- 50
T_sim    <- 200
N_fft    <- 256   # >= 4 * J_sim; power of 2

sim <- simulate_fourier_ar1(
  T         = T_sim,
  J         = J_sim,
  nu        = 1.5,
  ell       = 1.0,
  lambda    = 0.01,
  alpha     = 1.2,
  beta      = 1.0,
  total_var = 1,
  rho0      = 0.95,
  theta_vec = build_theta_vec(J = J_sim, theta0 = 0.3),
  phi_vec   = build_phi_vec(J = J_sim, phi0 = 0.5),
  seed      = 123
)

# --------------------------------------------------
# Field evaluation
# --------------------------------------------------

# Method 1: Dense FFT grid (recommended)
fft_result <- reconstruct_field_fft(sim$a, sim$b, N_fft = N_fft)
X_fft      <- fft_result$X          # T x N_fft matrix
x_fft_grid <- fft_result$x_grid     # length-N_fft spatial locations

# Method 2: Direct on a coarser observation grid (e.g., 100 sensors)
x_obs   <- seq(0, 2 * pi, length.out = 100)
X_obs   <- interpolate_to_obs(fft_result, x_obs)   # interpolate from FFT grid

# Also compute direct for comparison / validation
X_direct <- reconstruct_field_direct(sim$a, sim$b, x_fft_grid)

# --------------------------------------------------
# Validate FFT accuracy
# --------------------------------------------------
validate_fft_accuracy(sim$a, sim$b, N_fft = N_fft, n_check = 10)

# --------------------------------------------------
# Post-processing
# --------------------------------------------------
ap        <- amplitude_phase(sim$a, sim$b)
cov_check <- empirical_cov_aj_bj(sim$a, sim$b, sim$sigma2, sim$phi_vec)

cat("\nCovariance check (first 10 modes):\n")
print(head(cov_check, 10))

# ============================================================
# SECTION 6: DIAGNOSTIC PLOTS
# ============================================================

out_dir <- "Stats 669/research/img/crosscorr_fft"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------
# (a) Matern-like mode variances sigma_j^2
# --------------------------------------------------
png(file.path(out_dir, "matern_variances.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$sigma2, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(sigma[j]^2),
     main = expression("Matern-like Fourier variances " * sigma[j]^2))
dev.off()

# --------------------------------------------------
# (b) Mode-specific temporal persistence rho_j
# --------------------------------------------------
png(file.path(out_dir, "temporal_persistence.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$rho, type = "b", pch = 19,
     xlab = "Fourier mode j",
     ylab = expression(rho[j]),
     main = expression("Mode-specific temporal persistence " * rho[j]))
dev.off()

# --------------------------------------------------
# (c) Rotation angles theta_j and innovation correlations phi_j
# --------------------------------------------------
png(file.path(out_dir, "theta_phi_profiles.png"), width = 800, height = 600)
par(mfrow = c(1, 2))
plot(1:sim$params$J, sim$theta_vec, type = "b", pch = 19,
     xlab = "Fourier mode j", ylab = expression(theta[j]),
     main = expression("Rotation angles " * theta[j]))
abline(h = 0, lty = 2, col = "gray")
plot(1:sim$params$J, sim$phi_vec, type = "b", pch = 19,
     xlab = "Fourier mode j", ylab = expression(phi[j]),
     main = expression("Innovation correlations " * phi[j]))
abline(h = 0, lty = 2, col = "gray")
par(mfrow = c(1, 1))
dev.off()

# --------------------------------------------------
# (d) FFT vs direct accuracy check at one time point
# --------------------------------------------------
png(file.path(out_dir, "fft_vs_direct.png"), width = 900, height = 500)
par(mfrow = c(1, 2))

t_check <- 50
plot(x_fft_grid, X_fft[t_check, ], type = "l", lwd = 2, col = "steelblue",
     xlab = "x", ylab = expression(X[t](x)),
     main = paste("Field at t =", t_check, "(FFT vs Direct)"))
lines(x_fft_grid, X_direct[t_check, ], lwd = 2, lty = 2, col = "tomato")
legend("topright",
       legend = c("FFT (IFFT method)", "Direct summation"),
       col = c("steelblue", "tomato"), lty = c(1, 2), lwd = 2, bty = "n")

# Error between methods
err <- abs(X_fft[t_check, ] - X_direct[t_check, ])
plot(x_fft_grid, err, type = "l", lwd = 1.5, col = "darkgreen",
     xlab = "x", ylab = "|FFT - Direct|",
     main = sprintf("Abs. error: max = %.2e", max(err)),
     log = "y")
par(mfrow = c(1, 1))
dev.off()

# --------------------------------------------------
# (e) Empirical vs theoretical Cov(a_j, b_j)
# --------------------------------------------------
png(file.path(out_dir, "cov_aj_bj_check.png"), width = 800, height = 600)
plot(cov_check$j, cov_check$cov_theory, type = "l", lwd = 2, col = "steelblue",
     xlab = "Fourier mode j",
     ylab = expression(Cov(a[j], b[j])),
     main = expression("Empirical vs theoretical " * Cov(a[j](t), b[j](t))))
lines(cov_check$j, cov_check$cov_empirical, lwd = 2, col = "tomato", lty = 2)
legend("topright",
       legend = c("Theoretical: sigma_j^2 * phi_j", "Empirical"),
       col = c("steelblue", "tomato"), lwd = 2, lty = c(1, 2), bty = "n")
dev.off()

# --------------------------------------------------
# (f) Spatial field at selected times (from FFT grid)
# --------------------------------------------------
png(file.path(out_dir, "spatial_field_at_times.png"), width = 800, height = 600)
matplot(x_fft_grid, t(X_fft[c(1, 10, 50, 100, 150, 200), ]),
        type = "l", lty = 1, lwd = 2,
        xlab = "x", ylab = expression(X[t](x)),
        main = "Spatial field at selected time points (FFT grid)")
legend("topright",
       legend = paste("t =", c(1, 10, 50, 100, 150, 200)),
       col = 1:6, lty = 1, lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# (g) Interpolated field at observation points
# --------------------------------------------------
png(file.path(out_dir, "interpolated_obs.png"), width = 800, height = 600)
matplot(x_obs, t(X_obs[c(1, 50, 100, 150, 200), ]),
        type = "l", lty = 1, lwd = 2,
        xlab = "Observation location x", ylab = expression(X[t](x)),
        main = "Field interpolated to observation grid (100 sensors)")
legend("topright",
       legend = paste("t =", c(1, 50, 100, 150, 200)),
       col = 1:5, lty = 1, lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# (h) Full spatio-temporal field heatmap (FFT grid)
# --------------------------------------------------
png(file.path(out_dir, "spatio_temporal_field.png"), width = 800, height = 600)
image(x = x_fft_grid, y = 1:nrow(X_fft), z = t(X_fft),
      xlab = "space x", ylab = "time t",
      main = "Simulated spatio-temporal field (FFT grid)",
      col = hcl.colors(100, "YlGnBu", rev = TRUE))
dev.off()

# --------------------------------------------------
# (i) Amplitude trajectories for selected modes
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
# (j) Phase trajectories for selected modes
# --------------------------------------------------
png(file.path(out_dir, "phase_trajectories.png"), width = 800, height = 600)
matplot(1:nrow(ap$phase), ap$phase[, c(2, 6, 11, 21)],
        type = "l", lty = 1, lwd = 2,
        xlab = "time t", ylab = "Phase (radians)",
        main = "Phase trajectories (rotation visible if theta_j != 0)")
abline(h = c(-pi, 0, pi), lty = 2, col = "gray")
legend("topright",
       legend = c("j=1", "j=5", "j=10", "j=20"),
       col = 1:4, lty = 1, lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# (k) Temporal ACF of amplitudes for selected modes
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
# (l) GIF: field evolving through time (from FFT grid)
# --------------------------------------------------
if (requireNamespace("magick", quietly = TRUE)) {
  library(magick)
  dir.create(file.path(out_dir, "movie"), showWarnings = FALSE)

  png(file.path(out_dir, "movie", "field_evolution_%03d.png"),
      width = 800, height = 600)
  ylim_range <- range(X_fft)
  for (t in 1:nrow(X_fft)) {
    plot(x_fft_grid, X_fft[t, ], type = "l", lwd = 2,
         ylim = ylim_range,
         xlab = "space x", ylab = expression(X[t](x)),
         main = paste("Spatio-temporal field at time t =", t))
  }
  dev.off()

  png_files    <- list.files(file.path(out_dir, "movie"),
                             pattern = "\\.png$", full.names = TRUE)
  images       <- image_read(png_files)
  gif_animated <- image_animate(images, fps = 10)
  image_write(gif_animated, file.path(out_dir, "field_evolution.gif"))
  cat("GIF saved.\n")
} else {
  cat("magick package not available; skipping GIF.\n")
}

cat("\nAll plots saved to:", out_dir, "\n")
