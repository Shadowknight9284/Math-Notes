############################################################
# VALIDATION SCRIPT: Fourier-AR(1) Spatio-Temporal Model
#
# GOAL:
#   Run the full multi-mode model, reconstruct the field,
#   and save validation/summary images under the validation folder.
#
# OUTPUTS:
#   Saves validation images to: Stats 669/research/img/val
############################################################

library(MASS)

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

theta_function <- function(J, theta_base = 0.01) {
  theta    <- rep(theta_base, J + 1)
  theta[1] <- 0
  theta
}

phi_function <- function(J, phi_base = 0.7, decay = 0.3) {
  j   <- 0:J
  phi <- phi_base * exp(-decay * j / J)
  phi[1] <- 0
  pmax(pmin(phi, 0.99), -0.99)
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

simulate_fourier_ar1 <- function(T, J, nu, ell, lambda, alpha, beta,
                                 theta_base = 0.01,
                                 phi_base   = 0.7,
                                 phi_decay  = 0.3,
                                 total_var  = 1,
                                 rho0       = 0.95,
                                 seed       = NULL) {
  if (!is.null(seed)) set.seed(seed)
  sigma2 <- matern_spectrum(J = J, nu = nu, ell = ell, total_var = total_var)
  rho    <- rho_function(J = J, lambda = lambda, alpha = alpha, beta = beta, rho0 = rho0)
  theta  <- theta_function(J = J, theta_base = theta_base)
  phi    <- phi_function(J = J, phi_base = phi_base, decay = phi_decay)
  a <- matrix(0, nrow = T, ncol = J + 1)
  b <- matrix(0, nrow = T, ncol = J + 1)
  a[1, 1] <- rnorm(1, mean = 0, sd = sqrt(sigma2[1]))
  for (j in 1:J) {
    idx        <- j + 1
    Sigma_init <- build_Sigma_init_j(sigma2[idx], phi[idx])
    init       <- as.vector(mvrnorm(1, c(0, 0), Sigma_init))
    a[1, idx]  <- init[1]
    b[1, idx]  <- init[2]
  }
  for (t in 2:T) {
    eps_var <- sigma2[1] * (1 - rho[1]^2)
    a[t, 1] <- rho[1] * a[t - 1, 1] + rnorm(1, 0, sqrt(eps_var))
    for (j in 1:J) {
      idx         <- j + 1
      M_j         <- build_Mj(rho[idx], theta[idx])
      Sigma_eps_j <- build_Sigma_eps_j(sigma2[idx], rho[idx], phi[idx])
      eta          <- as.vector(mvrnorm(1, c(0, 0), Sigma_eps_j))
      new_state    <- M_j %*% c(a[t - 1, idx], b[t - 1, idx]) + eta
      a[t, idx]    <- new_state[1]
      b[t, idx]    <- new_state[2]
    }
  }
  list(a = a, b = b, sigma2 = sigma2, rho = rho, theta = theta, phi = phi,
       params = list(T = T, J = J, nu = nu, ell = ell, lambda = lambda,
                     alpha = alpha, beta = beta, theta_base = theta_base,
                     phi_base = phi_base, phi_decay = phi_decay,
                     total_var = total_var, rho0 = rho0))
}

reconstruct_field <- function(a, b, x_grid) {
  T  <- nrow(a)
  J  <- ncol(a) - 1
  nx <- length(x_grid)
  X <- matrix(0, nrow = T, ncol = nx)
  X <- X + a[, 1]
  for (j in 1:J) {
    idx <- j + 1
    X   <- X + a[, idx] %*% t(cos(j * x_grid)) + b[, idx] %*% t(sin(j * x_grid))
  }
  X
}

sim <- simulate_fourier_ar1(
  T = 200, J = 50, nu = 1.5, ell = 1.0,
  lambda = 0.08, alpha = 1.2, beta = 1.0,
  theta_base = 0.01, phi_base = 0.7, phi_decay = 0.3,
  total_var = 1, rho0 = 0.95, seed = 123
)

x_grid <- seq(0, 2 * pi, length.out = 200)
X      <- reconstruct_field(sim$a, sim$b, x_grid)

out_dir <- "Stats 669/research/img/debug/val"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

j_test <- 2
emp_cor_j1 <- cor(sim$a[, j_test], sim$b[, j_test])
target_phi_j1 <- sim$phi[j_test]

cat("Validation summary:\n")
cat("Target phi_1:", round(target_phi_j1, 4), "\n")
cat("Empirical Cor(a_1, b_1):", round(emp_cor_j1, 4), "\n")

png(file.path(out_dir, "val_matern_variances.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$sigma2, type = "b", pch = 19,
     xlab = "mode j", ylab = expression(sigma[j]^2),
     main = expression("Matérn-like variances " * sigma[j]^2))
dev.off()

png(file.path(out_dir, "val_temporal_persistence.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$rho, type = "b", pch = 19, col = "steelblue",
     xlab = "mode j", ylab = expression(rho[j]),
     main = expression("Temporal persistence " * rho[j]))
dev.off()

png(file.path(out_dir, "val_phi_by_mode.png"), width = 800, height = 600)
plot(0:sim$params$J, sim$phi, type = "b", pch = 19, col = "darkgreen",
     xlab = "mode j", ylab = expression(phi[j]),
     main = expression("Cosine/sine correlation " * phi[j]))
dev.off()

png(file.path(out_dir, "val_spatial_field_selected_times.png"), width = 900, height = 600)
matplot(x_grid, t(X[c(1, 50, 100, 150, 200), ]),
        type = "l", lty = 1, lwd = 2,
        xlab = "x", ylab = expression(X[t](x)),
        main = "Spatial field at selected times")
legend("topright", legend = paste("t =", c(1, 50, 100, 150, 200)),
       col = 1:5, lty = 1, lwd = 2, bty = "n")
dev.off()

png(file.path(out_dir, "val_j1_scatter.png"), width = 650, height = 650)
plot(sim$a[, j_test], sim$b[, j_test], pch = 19, cex = 0.5,
     col = rgb(0.2, 0.4, 0.8, 0.35),
     xlab = expression(a[1](t)), ylab = expression(b[1](t)),
     main = sprintf("j=1 scatter | cor=%.3f target=%.3f", emp_cor_j1, target_phi_j1))
abline(0, 1, col = "red", lty = 2, lwd = 2)
dev.off()

png(file.path(out_dir, "val_spacetime_heatmap.png"), width = 900, height = 600)
image(x = x_grid, y = 1:nrow(X), z = t(X),
      xlab = "space x", ylab = "time t",
      main = "Simulated spatio-temporal field",
      col = hcl.colors(100, "YlGnBu", rev = TRUE))
dev.off()

cat("All validation plots saved to:", out_dir, "\n")
