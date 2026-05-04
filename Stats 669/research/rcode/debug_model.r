############################################################
# DEBUG SCRIPT: Fourier-AR(1) Spatio-Temporal Model
#
# GOAL:
#   Isolate and debug two critical mechanisms for a single mode (j = 1):
#     1. Rotation M_j (theta_j): does phase drift at the expected rate?
#     2. Innovation correlation (phi_j): does Cor(a_j, b_j) ≈ phi_j?
#
# STRATEGY:
#   - Unit-test the small building blocks (M_j, Sigma_eps_j, mvrnorm)
#   - Simulate a single bivariate AR(1) (j = 1) under controlled parameters
#   - Check rotation via phase drift
#   - Check correlation via covariance, correlation, and scatter plot
#   - Run an extreme-parameter test (theta = pi/2, phi = 0.95)
#   - Produce a concise PASS/FAIL summary
############################################################

library(MASS)

# ==========================================================
# 0. Core helper functions (single-mode building blocks)
# ==========================================================

matern_spectrum <- function(J, nu, ell, total_var = 1) {
  kappa    <- sqrt(2 * nu) / ell
  j        <- 0:J
  raw_spec <- (kappa^2 + j^2)^(-(nu + 0.5))
  sigma2   <- total_var * raw_spec / sum(raw_spec)
  sigma2
}

rho_function <- function(J, lambda, alpha, beta, rho0 = 0.95) {
  j   <- 0:J
  rho <- 1 / (1 + lambda * j^alpha)^beta
  rho[1] <- rho0
  pmax(rho, 0)
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

unwrap_phase <- function(phase) {
  dp           <- diff(phase)
  dp_corrected <- dp - 2 * pi * round(dp / (2 * pi))
  c(phase[1], phase[1] + cumsum(dp_corrected))
}

cat("========================================================\n")
cat("DEBUG SCRIPT: Fourier-AR(1) Rotation and Correlation\n")
cat("========================================================\n\n")

# ==========================================================
# 1. Check build_Mj()
# ==========================================================

cat("----------------------------------------------------\n")
cat("SECTION 1: Checking build_Mj()\n")
cat("----------------------------------------------------\n")

# theta = 0 => rho * I_2
M_no_rotation <- build_Mj(rho_j = 0.9, theta_j = 0)
cat("M_j with theta = 0 (expect 0.9 * I_2):\n")
print(M_no_rotation)
cat("PASS? ", all.equal(M_no_rotation, 0.9 * diag(2)), "\n\n")

# theta = pi/2 => rho * [0 -1; 1 0]
M_90deg <- build_Mj(rho_j = 0.9, theta_j = pi/2)
cat("M_j with theta = pi/2 (expect 0.9 * [0 -1; 1 0]):\n")
print(M_90deg)
expected_90 <- 0.9 * matrix(c(0, 1, -1, 0), 2, 2)
cat("PASS? ", all.equal(M_90deg, expected_90), "\n\n")

# theta = pi/4, rho = 1 => pure rotation, det = 1
M_45deg <- build_Mj(rho_j = 1.0, theta_j = pi/4)
cat("M_j with rho = 1, theta = pi/4 (pure rotation):\n")
print(M_45deg)
cat("det(M_j) should be 1 for rho = 1:", det(M_45deg), "\n")
cat("PASS? ", isTRUE(all.equal(det(M_45deg), 1.0, tolerance = 1e-10)), "\n\n")

# Check eigenvalues (modulus = rho, argument = ±theta)
eig <- eigen(build_Mj(0.8, 0.5))$values
cat("Eigenvalues (rho = 0.8, theta = 0.5):\n")
print(eig)
cat("Expected modulus 0.8, got:", Mod(eig[1]), "\n")
cat("Expected argument ±0.5, got:", Arg(eig[1]), "\n\n")

# ==========================================================
# 2. Check build_Sigma_eps_j()
# ==========================================================

cat("----------------------------------------------------\n")
cat("SECTION 2: Checking build_Sigma_eps_j()\n")
cat("----------------------------------------------------\n")

Sigma_eps_test <- build_Sigma_eps_j(sigma2_j = 0.1, rho_j = 0.9, phi_j = 0.5)
cat("Sigma_eps_j (sigma2 = 0.1, rho = 0.9, phi = 0.5):\n")
print(Sigma_eps_test)

scale_expected <- 0.1 * (1 - 0.9^2)
cat("Expected diagonal:    ", scale_expected, "\n")
cat("Expected off-diagonal:", scale_expected * 0.5, "\n")
cat("PASS diag? ", isTRUE(all.equal(Sigma_eps_test[1, 1], scale_expected)), "\n")
cat("PASS offd? ", isTRUE(all.equal(Sigma_eps_test[1, 2], scale_expected * 0.5)), "\n\n")

eig_sigma <- eigen(Sigma_eps_test)$values
cat("Eigenvalues of Sigma_eps_j:", eig_sigma, "\n")
cat("Positive definite? ", all(eig_sigma > 0), "\n\n")

Sigma_eps_diag <- build_Sigma_eps_j(sigma2_j = 0.1, rho_j = 0.9, phi_j = 0)
cat("Sigma_eps_j with phi = 0 (should be diagonal):\n")
print(Sigma_eps_diag)
cat("Off-diagonal is zero? ", isTRUE(all.equal(Sigma_eps_diag[1, 2], 0)), "\n\n")

# ==========================================================
# 3. Check mvrnorm() correlation
# ==========================================================

cat("----------------------------------------------------\n")
cat("SECTION 3: Checking mvrnorm() correlation\n")
cat("----------------------------------------------------\n")

set.seed(42)
Sigma_test <- matrix(c(1, 0.8, 0.8, 1), 2, 2)
draws      <- mvrnorm(1000, mu = c(0, 0), Sigma = Sigma_test)
emp_cor    <- cor(draws[, 1], draws[, 2])

cat("Target correlation: 0.8\n")
cat("Empirical correlation from 1000 draws:", round(emp_cor, 4), "\n")
cat("PASS? ", abs(emp_cor - 0.8) < 0.05, "\n\n")

# ==========================================================
# 4. Single-mode simulation (j = 1), moderate theta, phi
# ==========================================================

cat("----------------------------------------------------\n")
cat("SECTION 4: J = 1, T = 500 simulation\n")
cat("----------------------------------------------------\n")

set.seed(123)

T_debug  <- 500
sigma2_1 <- 0.5
rho_1    <- 0.95
theta_1  <- 0.3
phi_1    <- 0.7

M1          <- build_Mj(rho_1, theta_1)
Sigma_eps1  <- build_Sigma_eps_j(sigma2_1, rho_1, phi_1)
Sigma_init1 <- build_Sigma_init_j(sigma2_1, phi_1)

cat("M_1:\n");          print(M1)
cat("Sigma_eps_1:\n"); print(Sigma_eps1)
cat("Sigma_init_1:\n");print(Sigma_init1)

a1 <- numeric(T_debug)
b1 <- numeric(T_debug)
eta_saved <- matrix(0, nrow = T_debug, ncol = 2)

# Initial state
init <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_init1))
a1[1] <- init[1]
b1[1] <- init[2]

# AR(1) recursion
for (t in 2:T_debug) {
  eta            <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_eps1))
  eta_saved[t, ] <- eta
  new_state      <- M1 %*% c(a1[t - 1], b1[t - 1]) + eta
  a1[t]          <- new_state[1]
  b1[t]          <- new_state[2]
}

cat("\nEmpirical stats for a_1(t):\n")
cat("  mean:", round(mean(a1), 4), "  (expect 0)\n")
cat("  var :", round(var(a1),  4), "  (expect", sigma2_1, ")\n")

cat("Empirical stats for b_1(t):\n")
cat("  mean:", round(mean(b1), 4), "  (expect 0)\n")
cat("  var :", round(var(b1),  4), "  (expect", sigma2_1, ")\n")

cat("Empirical Cov(a_1, b_1):", round(cov(a1, b1), 4),
    " (expect", sigma2_1 * phi_1, ")\n")
cat("Empirical Cor(a_1, b_1):", round(cor(a1, b1), 4),
    " (expect", phi_1, ")\n\n")

cat("--- Innovation check (independent of AR dynamics) ---\n")
cat("Theoretical Cor(eta_a, eta_b):", phi_1, "\n")
cat("Empirical   Cor(eta_a, eta_b):",
    round(cor(eta_saved[2:T_debug, 1], eta_saved[2:T_debug, 2]), 4), "\n")
cat("Expected    Cov(eta_a, eta_b):", sigma2_1 * (1 - rho_1^2) * phi_1, "\n")
cat("Empirical   Cov(eta_a, eta_b):",
    round(cov(eta_saved[2:T_debug, 1], eta_saved[2:T_debug, 2]), 4), "\n\n")

psi1          <- atan2(b1, a1)
psi1_unwrapped <- unwrap_phase(psi1)

# ==========================================================
# 5. Phase drift check for moderate theta
# ==========================================================

cat("----------------------------------------------------\n")
cat("SECTION 5: Phase drift check (theta_1 =", theta_1, ")\n")
cat("----------------------------------------------------\n")

t_idx    <- 1:T_debug
lm_phase <- lm(psi1_unwrapped ~ t_idx)
slope    <- coef(lm_phase)[2]

cat("Expected phase drift per step:", theta_1, "\n")
cat("Empirical phase drift (slope):", round(slope, 4), "\n")
cat("Ratio empirical/expected:", round(slope / theta_1, 3), "\n")
cat("PASS? (|ratio - 1| < 0.2):", abs(slope / theta_1 - 1) < 0.2, "\n\n")

# ==========================================================
# 6. Correlation check for moderate theta
# ==========================================================

cat("----------------------------------------------------\n")
cat("SECTION 6: Correlation check (theta_1 =", theta_1, ")\n")
cat("----------------------------------------------------\n")

cov_ab <- cov(a1, b1)
cat("Theoretical Cov(a_1, b_1):", sigma2_1 * phi_1, "\n")
cat("Empirical   Cov(a_1, b_1):", round(cov_ab, 4), "\n")
cat("PASS? (|diff| < 0.05):", abs(cov_ab - sigma2_1 * phi_1) < 0.05, "\n\n")

############################################################
# LYAPUNOV DEBUG CHUNK
#
# Add this after your Section 6 correlation check.
# It computes the stationary covariance implied by
#   z_t = M z_{t-1} + eps_t,
# compares the Lyapunov-theoretical stationary correlation
# with the empirical simulation correlation, and shows that
# phi controls the innovation correlation, not necessarily
# the stationary state correlation when theta != 0.
############################################################

# ----------------------------------------------------------
# 6B. Lyapunov stationary covariance check
# ----------------------------------------------------------

solve_stationary_cov_iter <- function(Mj, Sigma_eps, tol = 1e-12, max_iter = 50000) {
  Sigma <- Sigma_eps
  for (k in 1:max_iter) {
    Sigma_new <- Mj %*% Sigma %*% t(Mj) + Sigma_eps
    if (max(abs(Sigma_new - Sigma)) < tol) {
      return(list(Sigma = Sigma_new, iter = k, converged = TRUE))
    }
    Sigma <- Sigma_new
  }
  list(Sigma = Sigma, iter = max_iter, converged = FALSE)
}

cov_to_cor <- function(Sigma) {
  Sigma[1, 2] / sqrt(Sigma[1, 1] * Sigma[2, 2])
}

lyap_res <- solve_stationary_cov_iter(M1, Sigma_eps1)
Sigma_stat_lyap <- lyap_res$Sigma
cor_stat_lyap   <- cov_to_cor(Sigma_stat_lyap)
emp_cov_ab      <- cov(a1, b1)
emp_cor_ab      <- cor(a1, b1)
innov_cor_ab    <- cor(eta_saved[2:T_debug, 1], eta_saved[2:T_debug, 2])

cat("----------------------------------------------------\n")
cat("SECTION 6B: Lyapunov stationary covariance check\n")
cat("----------------------------------------------------\n")
cat("Converged? ", lyap_res$converged, " after ", lyap_res$iter, " iterations\n", sep = "")
cat("\nStationary covariance from Lyapunov equation:\n")
print(round(Sigma_stat_lyap, 6))
cat("\nInterpretation of correlations:\n")
cat("  phi_1 (input parameter)                 = ", round(phi_1, 4), "\n", sep = "")
cat("  Empirical innovation Cor(eta_a, eta_b) = ", round(innov_cor_ab, 4), "\n", sep = "")
cat("  Lyapunov stationary Cor(a_1, b_1)      = ", round(cor_stat_lyap, 4), "\n", sep = "")
cat("  Empirical state Cor(a_1, b_1)          = ", round(emp_cor_ab, 4), "\n", sep = "")
cat("  Empirical state Cov(a_1, b_1)          = ", round(emp_cov_ab, 4), "\n", sep = "")
cat("\nKey diagnostic:\n")
cat("  If Lyapunov stationary Cor is close to empirical state Cor,\n")
cat("  then the simulation is behaving correctly and phi_1 should be\n")
cat("  interpreted as innovation correlation, not stationary correlation.\n\n")

cat("Differences:\n")
cat("  |Empirical state Cor - Lyapunov Cor| = ", round(abs(emp_cor_ab - cor_stat_lyap), 6), "\n", sep = "")
cat("  |Empirical innovation Cor - phi_1|   = ", round(abs(innov_cor_ab - phi_1), 6), "\n\n", sep = "")

# Optional parameter sweep: how stationary correlation changes with phi
phi_grid <- seq(-0.95, 0.95, length.out = 101)
cor_grid <- numeric(length(phi_grid))
for (i in seq_along(phi_grid)) {
  Sigma_eps_tmp <- build_Sigma_eps_j(sigma2_1, rho_1, phi_grid[i])
  S_tmp         <- solve_stationary_cov_iter(M1, Sigma_eps_tmp)$Sigma
  cor_grid[i]   <- cov_to_cor(S_tmp)
}

out_dir <- "Stats 669/research/img/debug"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

png(file.path(out_dir, "debug_lyapunov_phi_mapping.png"), width = 800, height = 600)
plot(phi_grid, cor_grid, type = "l", lwd = 2, col = "steelblue",
     xlab = expression("Innovation correlation parameter " * phi[1]),
     ylab = expression("Stationary Cor(" * a[1] * "," * b[1] * ")"),
     main = "Lyapunov mapping: innovation correlation -> stationary correlation")
abline(0, 1, col = "gray50", lty = 2)
points(phi_1, cor_stat_lyap, pch = 19, col = "red", cex = 1.2)
text(phi_1, cor_stat_lyap,
     labels = sprintf("  current setting: phi=%.2f\n  stat cor=%.2f", phi_1, cor_stat_lyap),
     pos = 4, cex = 0.9)
legend("topleft",
       legend = c("Lyapunov stationary correlation", "45-degree line", "Current setting"),
       col = c("steelblue", "gray50", "red"), lty = c(1, 2, NA), pch = c(NA, NA, 19),
       lwd = c(2, 1, NA), bty = "n")
dev.off()

cat("Lyapunov debug plot saved to: ", file.path(out_dir, "debug_lyapunov_phi_mapping.png"), "\n", sep = "")

# ==========================================================
# 7. Extreme parameter test: theta = pi/2, phi = 0.95
# ==========================================================

cat("----------------------------------------------------\n")
cat("SECTION 7: Extreme parameter test\n")
cat("  theta = pi/2 (90° per step)\n")
cat("  phi   = 0.95 (very strong corr)\n")
cat("  rho   = 0.5\n")
cat("----------------------------------------------------\n")

set.seed(42)
T_ext    <- 200
sigma2_e <- 1.0
rho_e    <- 0.5
theta_e  <- pi / 2
phi_e    <- 0.95

M_e          <- build_Mj(rho_e, theta_e)
Sigma_eps_e  <- build_Sigma_eps_j(sigma2_e, rho_e, phi_e)
Sigma_init_e <- build_Sigma_init_j(sigma2_e, phi_e)

cat("M_e (theta = pi/2):\n")
print(round(M_e, 4))

a_e <- numeric(T_ext)
b_e <- numeric(T_ext)

init_e  <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_init_e))
a_e[1]  <- init_e[1]
b_e[1]  <- init_e[2]

for (t in 2:T_ext) {
  eta       <- as.vector(mvrnorm(1, mu = c(0, 0), Sigma = Sigma_eps_e))
  new_state <- M_e %*% c(a_e[t - 1], b_e[t - 1]) + eta
  a_e[t]    <- new_state[1]
  b_e[t]    <- new_state[2]
}

psi_e           <- atan2(b_e, a_e)
psi_e_unwrapped <- unwrap_phase(psi_e)
slope_e         <- coef(lm(psi_e_unwrapped ~ I(1:T_ext)))[2]

cat("\nRotation test (expect slope ≈", round(pi/2, 4), "):\n")
cat("  Empirical slope:", round(slope_e, 4), "\n")
cat("  PASS?", abs(slope_e - pi/2) < 0.3, "\n")

cat("\nCorrelation test (expect Cor ≈ 0.95):\n")
cat("  Empirical Cor(a_e, b_e):", round(cor(a_e, b_e), 4), "\n")
cat("  PASS?", abs(cor(a_e, b_e) - phi_e) < 0.05, "\n\n")

# ==========================================================
# 8. PASS / FAIL summary
# ==========================================================

cat("========================================================\n")
cat("SECTION 8: PASS / FAIL SUMMARY\n")
cat("========================================================\n")

test_results <- data.frame(
  Test = c(
    "build_Mj: theta=0 => rho*I",
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
      build_Sigma_eps_j(0.1, 0.9, 0.5)[1, 2],
      0.1 * (1 - 0.9^2) * 0.5
    )),
    all(eigen(build_Sigma_eps_j(0.1, 0.9, 0.5))$values > 0),
    abs(emp_cor - 0.8) < 0.05,
    abs(var(a1) - sigma2_1) < 0.1,
    abs(cov_ab - sigma2_1 * phi_1) < 0.05,
    abs(slope / theta_1 - 1) < 0.2,
    abs(slope_e - pi/2) < 0.3,
    abs(cor(a_e, b_e) - phi_e) < 0.05
  )
)
test_results$Status <- ifelse(test_results$Result, "PASS", "FAIL")
print(test_results[, c("Test", "Status")])

# ==========================================================
# 9. Diagnostic plots for J = 1 simulation
# ==========================================================

out_dir <- "Stats 669/research/img/debug"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Time series of a1 and b1
png(file.path(out_dir, "debug_a1_b1_timeseries.png"),
    width = 900, height = 500)
matplot(
  1:T_debug, cbind(a1, b1),
  type = "l", lty = 1, lwd = 1.5, col = c("steelblue", "tomato"),
  xlab = "time t", ylab = "value",
  main = sprintf("a_1(t) and b_1(t) | theta=%.2f, phi=%.2f", theta_1, phi_1)
)
legend("topright", legend = c("a_1(t)", "b_1(t)"),
       col = c("steelblue", "tomato"), lty = 1, lwd = 2, bty = "n")
dev.off()

# Scatter (a1, b1)
png(file.path(out_dir, "debug_scatter_a1_b1.png"),
    width = 600, height = 600)
plot(
  a1, b1,
  pch = 19, cex = 0.4, col = rgb(0.2, 0.4, 0.8, 0.4),
  xlab = expression(a[1](t)), ylab = expression(b[1](t)),
  main = sprintf("Scatter (a_1, b_1) | cor=%.3f, target=%.3f",
                 cor(a1, b1), phi_1)
)
abline(0, 1, col = "red", lty = 2, lwd = 2)
dev.off()

# Unwrapped phase
png(file.path(out_dir, "debug_unwrapped_phase.png"),
    width = 900, height = 500)
plot(
  1:T_debug, psi1_unwrapped,
  type = "l", lwd = 1.5, col = "steelblue",
  xlab = "time t", ylab = "Unwrapped phase (radians)",
  main = sprintf("Unwrapped phase | expected slope=%.3f, empirical=%.3f",
                 theta_1, slope)
)
abline(a = psi1_unwrapped[1], b = theta_1, col = "red", lty = 2, lwd = 2)
legend("topleft",
       legend = c("Empirical", "Expected slope"),
       col = c("steelblue", "red"), lty = c(1, 2), lwd = 2, bty = "n")
dev.off()

# Extreme test phase
t_short <- 1:40
png(file.path(out_dir, "debug_extreme_phase.png"),
    width = 900, height = 500)
par(mfrow = c(2, 1))

plot(
  t_short, atan2(b_e[t_short], a_e[t_short]),
  type = "b", pch = 19, cex = 0.7, col = "steelblue",
  xlab = "time t", ylab = "Phase (radians)",
  main = "Extreme test: wrapped phase (theta = pi/2, ~4-step cycle)"
)
abline(h = c(-pi, 0, pi), lty = 2, col = "gray")

plot(
  t_short, psi_e_unwrapped[t_short],
  type = "b", pch = 19, cex = 0.7, col = "tomato",
  xlab = "time t", ylab = "Unwrapped phase (radians)",
  main = sprintf("Extreme test: unwrapped phase | expected slope=%.3f",
                 pi / 2)
)
abline(a = psi_e_unwrapped[1], b = pi / 2, col = "black", lty = 2, lwd = 2)
legend("topleft", legend = c("Empirical", "Expected slope"),
       col = c("tomato", "black"), lty = c(1, 2), lwd = 2, bty = "n")
par(mfrow = c(1, 1))
dev.off()

# Cross-correlation a1 vs b1
png(file.path(out_dir, "debug_ccf_a1_b1.png"),
    width = 800, height = 500)
ccf(
  a1, b1, lag.max = 30,
  main = sprintf("CCF a_1 vs b_1\nlag-0 ≈ %.2f (phi); peak lag ≈ %.1f",
                 phi_1, -(pi / 2) / theta_1),
  ylab = "CCF"
)
abline(v = -(pi / 2) / theta_1, col = "red", lty = 2, lwd = 2)
dev.off()

cat("\nAll debug plots saved to:", out_dir, "\n")
cat("Check:\n")
cat("  debug_scatter_a1_b1.png\n")
cat("  debug_unwrapped_phase.png\n")
cat("  debug_extreme_phase.png\n")
cat("  debug_ccf_a1_b1.png\n")

