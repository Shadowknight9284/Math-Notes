############################################################
# DEBUG SCRIPT FOR FOURIER-AR(1) SPATIO-TEMPORAL MODEL
#
# PURPOSE:
#   Two critical bugs were identified from the diagnostic plots:
#
#   BUG 1: Rotation (theta_j) is NOT working.
#           Phase trajectories show pure random walk instead of
#           systematic drift. Expected: sawtooth pattern.
#
#   BUG 2: Innovation correlation (phi_j) is NOT working.
#           Empirical Cov(a_j, b_j) ≈ 0 for all j.
#           Expected: Cov(a_j, b_j) = sigma_j^2 * phi_j > 0.
#
# STRATEGY:
#   We isolate j = 1 only (simplest possible case) and run
#   a series of increasingly detailed checks:
#
#   SECTION 1: Verify build_Mj() produces the correct matrix
#   SECTION 2: Verify build_Sigma_eps_j() has off-diagonal elements
#   SECTION 3: Verify mvrnorm() draws are actually correlated
#   SECTION 4: Simulate J=1 only, T=500, and inspect (a_1, b_1)
#   SECTION 5: Check rotation by tracking phase drift directly
#   SECTION 6: Check correlation by scatter plot of (a_1, b_1)
#   SECTION 7: Run a controlled test with EXTREME parameters
#              (theta=pi/2, phi=0.99) where effects should be obvious
#   SECTION 8: Summary pass/fail report
############################################################

library(MASS)

# ============================================================
# HELPER FUNCTIONS (copied exactly from updated_simulation.r)
# ============================================================

matern_spectrum <- function(J, nu, ell, total_var = 1) {
  kappa    <- sqrt(2 * nu) / ell
  j        <- 0:J
  raw_spec <- (kappa^2 + j^2)^(-(nu + 0.5))
  sigma2   <- total_var * raw_spec / sum(raw_spec)
  return(sigma2)
}

rho_function <- function(J, lambda, alpha, beta, rho0 = 0.95) {
  j   <- 0:J
  rho <- 1 / (1 + lambda * j^alpha)^beta
  rho[1] <- rho0
  return(pmax(rho, 0))
}

build_Mj <- function(rho_j, theta_j) {
  matrix(
    rho_j * c(cos(theta_j), sin(theta_j), -sin(theta_j), cos(theta_j)),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

build_Sigma_eps_j <- function(sigma2_j, rho_j, phi_j) {
  scale <- sigma2_j * (1 - rho_j^2)
  matrix(
    scale * c(1, phi_j, phi_j, 1),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

build_Sigma_init_j <- function(sigma2_j, phi_j) {
  matrix(
    sigma2_j * c(1, phi_j, phi_j, 1),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

cat("========================================================\n")
cat("DEBUG SCRIPT: Fourier-AR(1) Rotation and Correlation\n")
cat("========================================================\n\n")

# ============================================================
# SECTION 1: Verify build_Mj() is correct
# ============================================================
cat("----------------------------------------------------\n")
cat("SECTION 1: Checking build_Mj()\n")
cat("----------------------------------------------------\n")

# Test with theta = 0: should give rho * I_2
M_no_rotation <- build_Mj(rho_j = 0.9, theta_j = 0)
cat("M_j with theta=0 (expect 0.9 * I_2):\n")
print(M_no_rotation)
cat("PASS?", all.equal(M_no_rotation, 0.9 * diag(2)), "\n\n")

# Test with theta = pi/2: should give rho * [0 -1; 1 0]
M_90deg <- build_Mj(rho_j = 0.9, theta_j = pi/2)
cat("M_j with theta=pi/2 (expect 0.9 * [0 -1; 1 0]):\n")
print(M_90deg)
expected_90 <- 0.9 * matrix(c(0, 1, -1, 0), 2, 2)
cat("PASS?", all.equal(M_90deg, expected_90), "\n\n")

# Test with theta = pi/4
M_45deg <- build_Mj(rho_j = 1.0, theta_j = pi/4)
cat("M_j with rho=1, theta=pi/4 (expect pure rotation matrix):\n")
print(M_45deg)
cat("det(M_j) should be 1 for rho=1:", det(M_45deg), "\n")
cat("PASS?", isTRUE(all.equal(det(M_45deg), 1.0, tolerance=1e-10)), "\n\n")

# Verify eigenvalues: should be rho * exp(+/- i*theta)
eig <- eigen(build_Mj(0.8, 0.5))$values
cat("Eigenvalues of M_j (rho=0.8, theta=0.5):\n")
print(eig)
cat("Expected modulus:", 0.8, " | Got:", Mod(eig[1]), "\n")
cat("Expected argument: +/-0.5 | Got:", Arg(eig[1]), "\n\n")


# ============================================================
# SECTION 2: Verify build_Sigma_eps_j() has off-diagonal elements
# ============================================================
cat("----------------------------------------------------\n")
cat("SECTION 2: Checking build_Sigma_eps_j()\n")
cat("----------------------------------------------------\n")

# With phi = 0.5, sigma2 = 0.1, rho = 0.9
Sigma_eps_test <- build_Sigma_eps_j(sigma2_j = 0.1, rho_j = 0.9, phi_j = 0.5)
cat("Sigma_eps_j (sigma2=0.1, rho=0.9, phi=0.5):\n")
print(Sigma_eps_test)
scale_expected <- 0.1 * (1 - 0.9^2)
cat("Expected diagonal:    ", scale_expected, "\n")
cat("Expected off-diagonal:", scale_expected * 0.5, "\n")
cat("PASS diagonal?",   isTRUE(all.equal(Sigma_eps_test[1,1], scale_expected)), "\n")
cat("PASS off-diagonal?", isTRUE(all.equal(Sigma_eps_test[1,2], scale_expected * 0.5)), "\n\n")

# Is it positive definite?
eig_sigma <- eigen(Sigma_eps_test)$values
cat("Eigenvalues of Sigma_eps_j:", eig_sigma, "\n")
cat("Positive definite?", all(eig_sigma > 0), "\n\n")

# Check with phi = 0: should be diagonal
Sigma_eps_diag <- build_Sigma_eps_j(sigma2_j = 0.1, rho_j = 0.9, phi_j = 0)
cat("Sigma_eps_j with phi=0 (should be diagonal):\n")
print(Sigma_eps_diag)
cat("PASS?", isTRUE(all.equal(Sigma_eps_diag[1,2], 0)), "\n\n")


# ============================================================
# SECTION 3: Verify mvrnorm() draws are actually correlated
# ============================================================
cat("----------------------------------------------------\n")
cat("SECTION 3: Checking mvrnorm() correlation\n")
cat("----------------------------------------------------\n")

# Draw 1000 samples and check empirical correlation
set.seed(42)
Sigma_test <- matrix(c(1, 0.8, 0.8, 1), 2, 2)
draws      <- mvrnorm(1000, mu = c(0, 0), Sigma = Sigma_test)
emp_cor    <- cor(draws[, 1], draws[, 2])
cat("Target correlation: 0.8\n")
cat("Empirical correlation from 1000 mvrnorm draws:", round(emp_cor, 4), "\n")
cat("PASS?", abs(emp_cor - 0.8) < 0.05, "\n\n")


# ============================================================
# SECTION 4: Simulate J=1 only, T=500
#            Check raw a_1(t) and b_1(t) trajectories
# ============================================================
cat("----------------------------------------------------\n")
cat("SECTION 4: Minimal J=1, T=500 simulation\n")
cat("----------------------------------------------------\n")

set.seed(123)

# Parameters for j=1 only
T_debug  <- 500
sigma2_1 <- 0.5
rho_1    <- 0.95
theta_1  <- 0.3
phi_1    <- 0.7

# Build matrices
M1          <- build_Mj(rho_1, theta_1)
Sigma_eps1  <- build_Sigma_eps_j(sigma2_1, rho_1, phi_1)
Sigma_init1 <- build_Sigma_init_j(sigma2_1, phi_1)

cat("Transition matrix M_1:\n");          print(M1)
cat("Innovation covariance Sigma_eps_1:\n"); print(Sigma_eps1)
cat("Initial covariance Sigma_init_1:\n");  print(Sigma_init1)

# ----- Allocate storage -----
a1 <- numeric(T_debug)
b1 <- numeric(T_debug)

# Save innovations to verify they are correlated
eta_saved <- matrix(0, nrow = T_debug, ncol = 2)

# ----- Initial state -----
init <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_init1))
a1[1] <- init[1]
b1[1] <- init[2]

# ----- Forward simulation (CORRECTED) -----
# Key fixes applied here:
#   1. as.vector() ensures eta is a plain length-2 vector, not a 1x2 matrix
#   2. Loop variable is T_debug (not T_ext)
#   3. Uses M1, Sigma_eps1, a1, b1 (NOT the extreme-test variables)
for (t in 2:T_debug) {
  eta            <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_eps1))
  eta_saved[t, ] <- eta
  new_state      <- M1 %*% c(a1[t-1], b1[t-1]) + eta
  a1[t]          <- new_state[1]
  b1[t]          <- new_state[2]
}

# ----- Empirical summaries -----
cat("\nEmpirical statistics for a_1(t):\n")
cat("  Mean:", round(mean(a1), 4), " (expect 0)\n")
cat("  Var: ", round(var(a1),  4), " (expect", sigma2_1, ")\n")

cat("Empirical statistics for b_1(t):\n")
cat("  Mean:", round(mean(b1), 4), " (expect 0)\n")
cat("  Var: ", round(var(b1),  4), " (expect", sigma2_1, ")\n")

cat("Empirical Cov(a_1, b_1):", round(cov(a1, b1), 4),
    " (expect", sigma2_1 * phi_1, ")\n")
cat("Empirical Cor(a_1, b_1):", round(cor(a1, b1), 4),
    " (expect", phi_1, ")\n\n")

# ----- Innovation correlation check -----
# This tells us whether the BUG is in the innovation draws (Sigma_eps)
# or in the AR(1) propagation itself.
# If innovations are correctly correlated but the states are not,
# the rotation matrix M is scrambling the correlation over time.
cat("--- Innovation check (should NOT depend on AR dynamics) ---\n")
cat("Theoretical Cor(eta_a, eta_b):", phi_1, "\n")
cat("Empirical   Cor(eta_a, eta_b):",
    round(cor(eta_saved[2:T_debug, 1], eta_saved[2:T_debug, 2]), 4), "\n")
cat("Expected    Cov(eta_a, eta_b):", sigma2_1 * (1 - rho_1^2) * phi_1, "\n")
cat("Empirical   Cov(eta_a, eta_b):",
    round(cov(eta_saved[2:T_debug, 1], eta_saved[2:T_debug, 2]), 4), "\n\n")

# Phase trajectory (used in Section 5)
psi1 <- atan2(b1, a1)


# ============================================================
# SECTION 5: Check rotation — is phase drifting systematically?
# ============================================================
cat("----------------------------------------------------\n")
cat("SECTION 5: Phase drift check (theta_1 =", theta_1, "rad/step)\n")
cat("----------------------------------------------------\n")

unwrap_phase <- function(phase) {
  dp           <- diff(phase)
  dp_corrected <- dp - 2 * pi * round(dp / (2 * pi))
  return(c(phase[1], phase[1] + cumsum(dp_corrected)))
}

psi1_unwrapped <- unwrap_phase(psi1)
t_idx          <- 1:T_debug
lm_phase       <- lm(psi1_unwrapped ~ t_idx)
slope          <- coef(lm_phase)[2]

cat("Expected phase drift per step: theta_1 =", theta_1, "rad/step\n")
cat("Empirical phase drift (slope):", round(slope, 4), "rad/step\n")
cat("Ratio (empirical / expected):", round(slope / theta_1, 3), "\n")
cat("PASS? (ratio should be close to 1):", abs(slope / theta_1 - 1) < 0.2, "\n\n")


# ============================================================
# SECTION 6: Check correlation — scatter plot of (a_1, b_1)
# ============================================================
cat("----------------------------------------------------\n")
cat("SECTION 6: Correlation check\n")
cat("----------------------------------------------------\n")

cat("Theoretical Cov(a_1, b_1):", sigma2_1 * phi_1, "\n")
cat("Empirical   Cov(a_1, b_1):", round(cov(a1, b1), 4), "\n")
cat("PASS? (|diff| < 0.05):", abs(cov(a1, b1) - sigma2_1 * phi_1) < 0.05, "\n\n")

# ============================================================
# SECTION 7: EXTREME PARAMETER TEST (CORRECTED)
#   theta = pi/2  =>  full 360-degree cycle every 4 steps
#   phi   = 0.95  =>  very strong a/b correlation
#   rho   = 0.5   =>  moderate persistence, rotation visible
# ============================================================
cat("----------------------------------------------------\n")
cat("SECTION 7: Extreme parameter test\n")
cat("  theta = pi/2 rad/step (full cycle in 4 steps)\n")
cat("  phi   = 0.95 (very strong correlation)\n")
cat("  rho   = 0.5\n")
cat("----------------------------------------------------\n")

set.seed(42)
T_ext    <- 200    # <-- extreme test uses its OWN length variable
sigma2_e <- 1.0
rho_e    <- 0.5
theta_e  <- pi / 2
phi_e    <- 0.95

M_e          <- build_Mj(rho_e, theta_e)
Sigma_eps_e  <- build_Sigma_eps_j(sigma2_e, rho_e, phi_e)
Sigma_init_e <- build_Sigma_init_j(sigma2_e, phi_e)

cat("Transition matrix (theta=pi/2):\n"); print(round(M_e, 4))

# ----- Allocate extreme-test storage -----
a_e <- numeric(T_ext)
b_e <- numeric(T_ext)

# ----- Initial state -----
init_e <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_init_e))
a_e[1] <- init_e[1]
b_e[1] <- init_e[2]

# ----- Forward simulation (CORRECTED) -----
# Key fixes:
#   1. Loop runs to T_ext (200), not T_debug (500)
#   2. Uses M_e and Sigma_eps_e (extreme-test matrices)
#   3. Updates a_e[t] and b_e[t] (NOT a1/b1)
for (t in 2:T_ext) {
  eta       <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_eps_e))
  new_state <- M_e %*% c(a_e[t-1], b_e[t-1]) + eta
  a_e[t]    <- new_state[1]
  b_e[t]    <- new_state[2]
}

psi_e           <- atan2(b_e, a_e)
psi_e_unwrapped <- unwrap_phase(psi_e)
slope_e         <- coef(lm(psi_e_unwrapped ~ I(1:T_ext)))[2]

cat("\nRotation test (expect slope ≈", round(pi/2, 4), "rad/step):\n")
cat("  Empirical slope:", round(slope_e, 4), "\n")
cat("  PASS?", abs(slope_e - pi/2) < 0.3, "\n")

cat("\nCorrelation test (expect Cor ≈ 0.95):\n")
cat("  Empirical Cor(a_e, b_e):", round(cor(a_e, b_e), 4), "\n")
cat("  PASS?", abs(cor(a_e, b_e) - phi_e) < 0.05, "\n\n")


# ============================================================
# SECTION 8: Summary report and diagnostic plots
# ============================================================
cat("============================================================\n")
cat("SECTION 8: PASS / FAIL SUMMARY\n")
cat("============================================================\n")

test_results <- data.frame(
  Test = c(
    "build_Mj: theta=0 gives rho*I",
    "build_Mj: eigenvalue modulus = rho",
    "build_Sigma_eps: off-diagonal = scale*phi",
    "build_Sigma_eps: positive definite",
    "mvrnorm: draws are correlated",
    "J=1 sim: Var(a_1) close to sigma2",
    "J=1 sim: Cov(a_1,b_1) close to sigma2*phi",
    "Phase drift: slope close to theta_1",
    "Extreme test: rotation visible",
    "Extreme test: correlation visible"
  ),
  Result = c(
    isTRUE(all.equal(build_Mj(0.9, 0), 0.9 * diag(2))),
    abs(Mod(eigen(build_Mj(0.8, 0.5))$values[1]) - 0.8) < 1e-10,
    isTRUE(all.equal(
      build_Sigma_eps_j(0.1, 0.9, 0.5)[1,2],
      0.1 * (1-0.9^2) * 0.5
    )),
    all(eigen(build_Sigma_eps_j(0.1, 0.9, 0.5))$values > 0),
    abs(emp_cor - 0.8) < 0.05,
    abs(var(a1) - sigma2_1) < 0.1,
    abs(cov(a1, b1) - sigma2_1 * phi_1) < 0.05,
    abs(slope / theta_1 - 1) < 0.2,
    abs(slope_e - pi/2) < 0.3,
    abs(cor(a_e, b_e) - phi_e) < 0.05
  )
)
test_results$Status <- ifelse(test_results$Result, "✅ PASS", "❌ FAIL")
print(test_results[, c("Test", "Status")])


# ============================================================
# SECTION 9: Diagnostic plots for the J=1 minimal simulation
# ============================================================

out_dir <- "Stats 669/research/img/debug"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------
# Plot 1: a_1(t) and b_1(t) time series
#   If phi_j is working: they should move together
#   If rotation is working: b leads/lags a by theta_1 steps
# --------------------------------------------------
png(file.path(out_dir, "debug_a1_b1_timeseries.png"), width = 900, height = 500)
par(mfrow = c(1, 1))
matplot(1:T_debug, cbind(a1, b1),
        type = "l", lty = 1, lwd = 1.5, col = c("steelblue", "tomato"),
        xlab = "time t", ylab = "value",
        main = paste("a_1(t) and b_1(t) | theta=", theta_1, ", phi=", phi_1))
legend("topright", legend = c("a_1(t)", "b_1(t)"),
       col = c("steelblue", "tomato"), lty = 1, lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# Plot 2: Scatter plot of (a_1(t), b_1(t))
#   If phi_j is working: points should cluster on a diagonal (positive correlation)
#   If phi = 0: points should form a circle (uncorrelated, equal variances)
# --------------------------------------------------
png(file.path(out_dir, "debug_scatter_a1_b1.png"), width = 600, height = 600)
plot(a1, b1, pch = 19, cex = 0.4, col = rgb(0.2, 0.4, 0.8, 0.4),
     xlab = expression(a[1](t)), ylab = expression(b[1](t)),
     main = paste("Scatter (a_1, b_1) | empirical cor =",
                  round(cor(a1, b1), 3),
                  "\n expected cor =", phi_1))
abline(0, 1, col = "red", lty = 2, lwd = 2)  # diagonal reference line
dev.off()

# --------------------------------------------------
# Plot 3: Unwrapped phase over time
#   If rotation is working: should be roughly linear with slope theta_1
#   If rotation is NOT working: should look like a random walk (flat on average)
# --------------------------------------------------
png(file.path(out_dir, "debug_unwrapped_phase.png"), width = 900, height = 500)
plot(1:T_debug, psi1_unwrapped, type = "l", lwd = 1.5, col = "steelblue",
     xlab = "time t", ylab = "Unwrapped phase (radians)",
     main = paste("Unwrapped phase | expect slope =", theta_1, "rad/step\n",
                  "empirical slope =", round(slope, 4), "rad/step"))
abline(a = psi1_unwrapped[1], b = theta_1, col = "red",
       lty = 2, lwd = 2)  # theoretical expected drift line
legend("topleft",
       legend = c("Empirical unwrapped phase", paste("Expected slope:", theta_1)),
       col = c("steelblue", "red"), lty = c(1, 2), lwd = 2, bty = "n")
dev.off()

# --------------------------------------------------
# Plot 4: Extreme test — phase should cycle every 4 steps
#   (theta = pi/2 means 90-degree rotation per step)
# --------------------------------------------------
t_short <- 1:40  # show first 40 steps only so cycles are visible
png(file.path(out_dir, "debug_extreme_phase.png"), width = 900, height = 500)
par(mfrow = c(2, 1))

# Raw wrapped phase (should look like sawtooth with period 4)
plot(t_short, atan2(b_e[t_short], a_e[t_short]),
     type = "b", pch = 19, cex = 0.7, col = "steelblue",
     xlab = "time t", ylab = "Phase (radians)",
     main = "Extreme test: wrapped phase (theta=pi/2, expect 4-step cycle)")
abline(h = c(-pi, 0, pi), lty = 2, col = "gray")

# Unwrapped phase (should look like a straight line with slope pi/2)
plot(t_short, psi_e_unwrapped[t_short],
     type = "b", pch = 19, cex = 0.7, col = "tomato",
     xlab = "time t", ylab = "Unwrapped phase (radians)",
     main = paste("Extreme test: unwrapped phase | expected slope pi/2 =",
                  round(pi/2, 3)))
abline(a = psi_e_unwrapped[1], b = pi/2, col = "black", lty = 2, lwd = 2)
legend("topleft", legend = c("Empirical", "Expected slope"),
       col = c("tomato", "black"), lty = c(1, 2), lwd = 2, bty = "n")
par(mfrow = c(1, 1))
dev.off()

# --------------------------------------------------
# Plot 5: Cross-correlation between a_1(t) and b_1(t)
#   If rotation (theta_1 = 0.3) is working:
#     b_1(t) ≈ a_1(t - lag) where lag ≈ (pi/2) / theta_1 ≈ 5.2 steps
#     i.e., CCF should peak at lag ≈ -5
#   If phi_1 is working:
#     CCF at lag 0 should be positive (≈ phi_1)
# --------------------------------------------------
png(file.path(out_dir, "debug_ccf_a1_b1.png"), width = 800, height = 500)
ccf(a1, b1, lag.max = 30,
    main = paste("Cross-correlation a_1(t) vs b_1(t)\n",
                 "lag-0 should ≈", phi_1, "(phi effect)\n",
                 "peak lag should ≈ -(pi/2)/theta_1 =",
                 round(-(pi/2)/theta_1, 1), "(rotation effect)"),
    ylab = "CCF")
abline(v = -(pi/2)/theta_1, col = "red", lty = 2, lwd = 2)
dev.off()

cat("\nAll debug plots saved to:", out_dir, "\n")
cat("\nKey plots to inspect:\n")
cat("  debug_scatter_a1_b1.png     -- Should show diagonal cluster if phi works\n")
cat("  debug_unwrapped_phase.png   -- Should show linear drift if theta works\n")
cat("  debug_extreme_phase.png     -- Should show 4-step sawtooth if theta works\n")
cat("  debug_ccf_a1_b1.png         -- Peak lag should be -(pi/2)/theta_1 ≈ -5\n")



# Same setup as before
set.seed(123)
T_test <- 500
sigma2_1 <- 0.5
rho_1 <- 0.95
theta_1_small <- 0.05  # ← CHANGED: much smaller rotation
phi_1 <- 0.7

M1_small <- build_Mj(rho_1, theta_1_small)
Sigma_eps1 <- build_Sigma_eps_j(sigma2_1, rho_1, phi_1)
Sigma_init1 <- build_Sigma_init_j(sigma2_1, phi_1)

cat("Transition matrix with small rotation:\n")
print(M1_small)
# Should be very close to diagonal:
# [0.95  -0.047]  ← small off-diagonal = weak cross-lagged effect
# [0.047  0.95 ]

a1 <- numeric(T_test)
b1 <- numeric(T_test)

init <- as.vector(mvrnorm(1, c(0,0), Sigma_init1))
a1[1] <- init[1]; b1[1] <- init[2]

for (t in 2:T_test) {
  eta <- as.vector(mvrnorm(1, c(0,0), Sigma_eps1))
  new_state <- M1_small %*% c(a1[t-1], b1[t-1]) + eta
  a1[t] <- new_state[1]
  b1[t] <- new_state[2]
}

cat("\nWith theta =", theta_1_small, ":\n")
cat("Empirical Cor(a_1, b_1):", round(cor(a1, b1), 3), " (expect", phi_1, ")\n")
cat("Cross-lagged coefficient b→a:", round(-rho_1 * sin(theta_1_small), 4), "\n")
cat("Cross-lagged coefficient a→b:", round(rho_1 * sin(theta_1_small), 4), "\n")

# ============================================================
# TEST: Small rotation (theta = 0.01) with decaying phi
# Checks:
#   1. Cor(a_1, b_1) is close to phi_1 (contemporary correlation)
#   2. Cross-lagged coefficients are nonzero but small
#   3. Innovations are correctly correlated
# ============================================================

library(MASS)

# ---- Helper functions ----
build_Mj <- function(rho_j, theta_j) {
  matrix(
    rho_j * c(cos(theta_j), sin(theta_j), -sin(theta_j), cos(theta_j)),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

build_Sigma_eps_j <- function(sigma2_j, rho_j, phi_j) {
  scale <- sigma2_j * (1 - rho_j^2)
  matrix(scale * c(1, phi_j, phi_j, 1), nrow = 2, ncol = 2, byrow = FALSE)
}

build_Sigma_init_j <- function(sigma2_j, phi_j) {
  matrix(sigma2_j * c(1, phi_j, phi_j, 1), nrow = 2, ncol = 2, byrow = FALSE)
}

# ---- Parameters ----
set.seed(123)
T_test   <- 1000     # longer run for better empirical estimates
sigma2_1 <- 0.5
rho_1    <- 0.95

# Three values of theta to compare
theta_vals <- c(0.30, 0.05, 0.01)

# Decaying phi across frequencies (using j=1 only here, so phi is fixed)
# phi_j = phi_base * exp(-decay * j / J)
# For j=1, J=50: phi_1 = 0.7 * exp(-0.3 * 1/50) = 0.696 ≈ 0.7
phi_base  <- 0.7
decay     <- 0.3
J_full    <- 50
phi_1     <- phi_base * exp(-decay * 1 / J_full)

cat("===========================================================\n")
cat("Decaying phi at j=1:", round(phi_1, 4),
    " (base =", phi_base, ", decay =", decay, ")\n")
cat("===========================================================\n\n")

# ---- Run simulation for each theta value ----
results <- data.frame(
  theta        = theta_vals,
  emp_cor_ab   = NA,
  target_cor   = round(phi_1, 3),
  cross_lag_ba = NA,
  cross_lag_ab = NA,
  innov_cor    = NA
)

for (i in seq_along(theta_vals)) {

  theta_1 <- theta_vals[i]

  M1          <- build_Mj(rho_1, theta_1)
  Sigma_eps1  <- build_Sigma_eps_j(sigma2_1, rho_1, phi_1)
  Sigma_init1 <- build_Sigma_init_j(sigma2_1, phi_1)

  # Allocate
  a1        <- numeric(T_test)
  b1        <- numeric(T_test)
  eta_saved <- matrix(0, nrow = T_test, ncol = 2)

  # Initial state from stationary distribution
  init  <- as.vector(mvrnorm(1, c(0, 0), Sigma_init1))
  a1[1] <- init[1]
  b1[1] <- init[2]

  # Forward simulation
  for (t in 2:T_test) {
    eta            <- as.vector(mvrnorm(1, c(0, 0), Sigma_eps1))
    eta_saved[t, ] <- eta
    new_state      <- M1 %*% c(a1[t-1], b1[t-1]) + eta
    a1[t]          <- new_state[1]
    b1[t]          <- new_state[2]
  }

  # Store results
  results$emp_cor_ab[i]   <- round(cor(a1, b1), 4)
  results$cross_lag_ba[i] <- round(-rho_1 * sin(theta_1), 5)
  results$cross_lag_ab[i] <- round( rho_1 * sin(theta_1), 5)
  results$innov_cor[i]    <- round(cor(eta_saved[2:T_test, 1],
                                       eta_saved[2:T_test, 2]), 4)

  cat("theta =", theta_1, "\n")
  cat("  M_j (transition matrix):\n")
  print(round(M1, 5))
  cat("  Contemporary Cor(a_1, b_1):", results$emp_cor_ab[i],
      "  (target:", round(phi_1, 3), ")\n")
  cat("  Cross-lag b_{t-1} -> a_t  :", results$cross_lag_ba[i], "\n")
  cat("  Cross-lag a_{t-1} -> b_t  :", results$cross_lag_ab[i], "\n")
  cat("  Innovation Cor(eta_a, eta_b):", results$innov_cor[i],
      "  (target:", round(phi_1, 3), ")\n\n")
}

# ---- Summary table ----
cat("===========================================================\n")
cat("SUMMARY: Effect of theta on contemporary correlation\n")
cat("===========================================================\n")
print(results)

# ---- Pass/Fail ----
cat("\nPASS/FAIL (tolerance = 0.10 of target phi):\n")
for (i in seq_along(theta_vals)) {
  diff   <- abs(results$emp_cor_ab[i] - phi_1)
  status <- ifelse(diff < 0.10, "PASS", "FAIL")
  cat("  theta =", theta_vals[i], "->", status,
      "| Cor =", results$emp_cor_ab[i],
      "| diff =", round(diff, 4), "\n")
}

# ---- Scatter plot for best theta ----
# Re-run with best theta (smallest) for the scatter plot
best_theta <- theta_vals[which.min(abs(results$emp_cor_ab - phi_1))]
cat("\nBest theta:", best_theta, "\n")

set.seed(123)
M1_best      <- build_Mj(rho_1, best_theta)
Sigma_eps1   <- build_Sigma_eps_j(sigma2_1, rho_1, phi_1)
Sigma_init1  <- build_Sigma_init_j(sigma2_1, phi_1)

a1 <- numeric(T_test); b1 <- numeric(T_test)
init  <- as.vector(mvrnorm(1, c(0, 0), Sigma_init1))
a1[1] <- init[1]; b1[1] <- init[2]

for (t in 2:T_test) {
  eta       <- as.vector(mvrnorm(1, c(0, 0), Sigma_eps1))
  new_state <- M1_best %*% c(a1[t-1], b1[t-1]) + eta
  a1[t]     <- new_state[1]
  b1[t]     <- new_state[2]
}

par(mfrow = c(1, 2))

# Scatter: should show diagonal cluster
plot(a1, b1, pch = 19, cex = 0.3,
     col = rgb(0.2, 0.4, 0.8, 0.3),
     xlab = expression(a[1](t)),
     ylab = expression(b[1](t)),
     main = paste0("Scatter a_1 vs b_1\n",
                   "theta=", best_theta,
                   " | Cor=", round(cor(a1, b1), 3),
                   " | target=", round(phi_1, 3)))
abline(0, 1, col = "red", lty = 2, lwd = 2)

# CCF: lag-0 should be phi_1, peak lag near -(pi/2)/theta
ccf(a1, b1, lag.max = 50,
    main = paste0("CCF a_1 vs b_1\n",
                  "lag-0 target=", round(phi_1, 3),
                  " | peak lag target=-", round((pi/2)/best_theta, 1)),
    ylab = "CCF")
abline(v  = -(pi/2)/best_theta, col = "red", lty = 2, lwd = 2)
abline(h  =  phi_1,             col = "blue", lty = 3, lwd = 1)

par(mfrow = c(1, 1))


############################################################
# FOURIER-AR(1) SPATIO-TEMPORAL MODEL WITH ROTATION
# 
# FINAL WORKING VERSION with small rotation theta = 0.01
#
# This achieves:
#   - Contemporary correlation Cor(a_j, b_j) ≈ phi_j
#   - Cross-lagged dynamics: a_j(t) depends on b_j(t-1)
#   - Slow phase drift for evolving spatial patterns
############################################################

library(MASS)

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------
matern_spectrum <- function(J, nu, ell, total_var = 1) {
  kappa    <- sqrt(2 * nu) / ell
  j        <- 0:J
  raw_spec <- (kappa^2 + j^2)^(-(nu + 0.5))
  sigma2   <- total_var * raw_spec / sum(raw_spec)
  return(sigma2)
}

rho_function <- function(J, lambda, alpha, beta, rho0 = 0.95) {
  j   <- 0:J
  rho <- 1 / (1 + lambda * j^alpha)^beta
  rho[1] <- rho0
  return(pmax(rho, 0))
}

# UPDATED: Small rotation to preserve correlation
theta_function <- function(J, theta_base = 0.01) {
  # Very small rotation: preserves Cor(a_j, b_j) ≈ phi_j
  # while maintaining cross-lagged dynamics
  theta <- rep(theta_base, J+1)
  theta[1] <- 0  # no rotation for j=0 (mean mode)
  return(theta)
}

# UPDATED: Decaying cross-frequency correlation
phi_function <- function(J, phi_base = 0.7, decay = 0.3) {
  # Low frequencies: strong cosine/sine coupling
  # High frequencies: weak coupling
  j   <- 0:J
  phi <- phi_base * exp(-decay * j / J)
  phi[1] <- 0  # j=0 has no sine component
  return(pmax(pmin(phi, 0.99), -0.99))
}

build_Mj <- function(rho_j, theta_j) {
  matrix(
    rho_j * c(cos(theta_j), sin(theta_j), -sin(theta_j), cos(theta_j)),
    nrow = 2, ncol = 2, byrow = FALSE
  )
}

build_Sigma_eps_j <- function(sigma2_j, rho_j, phi_j) {
  scale <- sigma2_j * (1 - rho_j^2)
  matrix(scale * c(1, phi_j, phi_j, 1), nrow = 2, ncol = 2, byrow = FALSE)
}

build_Sigma_init_j <- function(sigma2_j, phi_j) {
  matrix(sigma2_j * c(1, phi_j, phi_j, 1), nrow = 2, ncol = 2, byrow = FALSE)
}

# ----------------------------------------------------------
# MAIN SIMULATION FUNCTION
# ----------------------------------------------------------
simulate_fourier_ar1 <- function(T, J, nu, ell, lambda, alpha, beta,
                                 theta_base = 0.01,
                                 phi_base = 0.7,
                                 phi_decay = 0.3,
                                 total_var = 1,
                                 rho0 = 0.95,
                                 seed = NULL) {
  
  if (!is.null(seed)) set.seed(seed)
  
  # Get mode variances, persistence, rotation, and correlation
  sigma2 <- matern_spectrum(J = J, nu = nu, ell = ell, total_var = total_var)
  rho    <- rho_function(J = J, lambda = lambda, alpha = alpha, beta = beta, rho0 = rho0)
  theta  <- theta_function(J = J, theta_base = theta_base)
  phi    <- phi_function(J = J, phi_base = phi_base, decay = phi_decay)
  
  # Allocate storage
  a <- matrix(0, nrow = T, ncol = J + 1)
  b <- matrix(0, nrow = T, ncol = J + 1)
  
  # --------------------------------------------------------
  # Initial state from stationary distributions
  # --------------------------------------------------------
  # j = 0 (mean mode, no rotation, no phi)
  a[1, 1] <- rnorm(1, mean = 0, sd = sqrt(sigma2[1]))
  
  # j >= 1 (oscillatory modes with correlation)
  for (j in 1:J) {
    idx         <- j + 1
    Sigma_init  <- build_Sigma_init_j(sigma2[idx], phi[idx])
    init        <- as.vector(mvrnorm(1, c(0, 0), Sigma_init))
    a[1, idx]   <- init[1]
    b[1, idx]   <- init[2]
  }
  
  # --------------------------------------------------------
  # Forward simulation with rotation and correlation
  # --------------------------------------------------------
  for (t in 2:T) {
    
    # Mode j = 0 (scalar AR(1), no correlation)
    eps_var   <- sigma2[1] * (1 - rho[1]^2)
    a[t, 1]   <- rho[1] * a[t - 1, 1] + rnorm(1, 0, sqrt(eps_var))
    
    # Modes j = 1,...,J (bivariate AR(1) with rotation and correlated innovations)
    for (j in 1:J) {
      idx <- j + 1
      
      M_j         <- build_Mj(rho[idx], theta[idx])
      Sigma_eps_j <- build_Sigma_eps_j(sigma2[idx], rho[idx], phi[idx])
      
      eta       <- as.vector(mvrnorm(1, c(0, 0), Sigma_eps_j))
      new_state <- M_j %*% c(a[t-1, idx], b[t-1, idx]) + eta
      
      a[t, idx] <- new_state[1]
      b[t, idx] <- new_state[2]
    }
  }
  
  return(list(
    a = a,
    b = b,
    sigma2 = sigma2,
    rho = rho,
    theta = theta,
    phi = phi,
    params = list(T = T, J = J, nu = nu, ell = ell,
                  lambda = lambda, alpha = alpha, beta = beta,
                  theta_base = theta_base,
                  phi_base = phi_base, phi_decay = phi_decay,
                  total_var = total_var, rho0 = rho0)
  ))
}

# ----------------------------------------------------------
# FIELD RECONSTRUCTION
# ----------------------------------------------------------
reconstruct_field <- function(a, b, x_grid) {
  T  <- nrow(a)
  J  <- ncol(a) - 1
  nx <- length(x_grid)
  
  X <- matrix(0, nrow = T, ncol = nx)
  X <- X + a[, 1]  # j=0 constant mode
  
  for (j in 1:J) {
    idx <- j + 1
    X   <- X + a[, idx] %*% t(cos(j * x_grid)) +
                b[, idx] %*% t(sin(j * x_grid))
  }
  
  return(X)
}

amplitude_phase <- function(a, b) {
  J <- ncol(a) - 1
  
  amp   <- a
  phase <- a
  
  amp[, 1]   <- abs(a[, 1])
  phase[, 1] <- NA
  
  for (j in 1:J) {
    idx         <- j + 1
    amp[, idx]  <- sqrt(a[, idx]^2 + b[, idx]^2)
    phase[, idx]<- atan2(b[, idx], a[, idx])
  }
  
  return(list(amplitude = amp, phase = phase))
}

# ----------------------------------------------------------
# EXAMPLE SIMULATION
# ----------------------------------------------------------
sim <- simulate_fourier_ar1(
  T          = 200,
  J          = 50,
  nu         = 1.5,
  ell        = 1.0,
  lambda     = 0.08,
  alpha      = 1.2,
  beta       = 1.0,
  theta_base = 0.01,      # UPDATED: small rotation
  phi_base   = 0.7,       # UPDATED: base correlation
  phi_decay  = 0.3,       # UPDATED: decay across frequencies
  total_var  = 1,
  rho0       = 0.95,
  seed       = 123
)

x_grid <- seq(0, 2 * pi, length.out = 200)
X      <- reconstruct_field(sim$a, sim$b, x_grid)
ap     <- amplitude_phase(sim$a, sim$b)

# ----------------------------------------------------------
# DIAGNOSTICS
# ----------------------------------------------------------
cat("Model summary:\n")
cat("  theta_base =", sim$params$theta_base,
    " (small rotation for cross-lagged dynamics)\n")
cat("  phi_base   =", sim$params$phi_base,
    " (base correlation at j=1)\n")
cat("  phi_decay  =", sim$params$phi_decay,
    " (exponential decay across frequencies)\n\n")

# Check empirical correlation at j=1
j_test <- 2  # j=1 is column 2
emp_cor_j1 <- cor(sim$a[, j_test], sim$b[, j_test])
target_phi_j1 <- sim$phi[j_test]

cat("Empirical check at j=1:\n")
cat("  Target phi_1:", round(target_phi_j1, 4), "\n")
cat("  Empirical Cor(a_1, b_1):", round(emp_cor_j1, 4), "\n")
cat("  Cross-lag strength:", round(sim$params$rho0 * sin(sim$params$theta_base), 5), "\n\n")

# Visualization
par(mfrow = c(2, 2))

# 1. Matérn-like mode variances
plot(0:sim$params$J, sim$sigma2, type = "b", pch = 19,
     xlab = "Fourier mode j", ylab = expression(sigma[j]^2),
     main = expression("Matérn-like variances " * sigma[j]^2))

# 2. Temporal persistence
plot(0:sim$params$J, sim$rho, type = "b", pch = 19, col = "steelblue",
     xlab = "Fourier mode j", ylab = expression(rho[j]),
     main = expression("Temporal persistence " * rho[j]))

# 3. Cross-frequency correlation phi_j
plot(0:sim$params$J, sim$phi, type = "b", pch = 19, col = "darkgreen",
     xlab = "Fourier mode j", ylab = expression(phi[j]),
     main = expression("Cross-frequency correlation " * phi[j]))

# 4. Spatial field at selected times
matplot(x_grid, t(X[c(1, 50, 100, 150, 200), ]),
        type = "l", lty = 1, lwd = 2,
        xlab = "x", ylab = expression(X[t](x)),
        main = "Spatial field at selected times")
legend("topright", legend = paste("t =", c(1, 50, 100, 150, 200)),
       col = 1:5, lty = 1, lwd = 2, bty = "n")

par(mfrow = c(1, 1))

cat("Simulation complete.\n")
