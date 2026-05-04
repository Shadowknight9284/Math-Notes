library(MASS)
library(ggplot2)

set.seed(123)

## build covariance K(r) = (phi1/phi2) * exp(-phi2 * r)
exp_cov_phi <- function(D, phi1, phi2) {
  (phi1 / phi2) * exp(-phi2 * D)
}

## REML log-likelihood for given phi2, given fixed distance matrix D
reml_loglik_phi2 <- function(phi2, y, D) {
  if (phi2 <= 0) return(-Inf)
  m <- length(y)

  # correlation-type matrix R(phi2) = exp(-phi2 * D)
  R <- exp(-phi2 * D)

  # Cholesky for speed and stability
  L <- tryCatch(chol(R), error = function(e) NULL)
  if (is.null(L)) return(-Inf)

  # log|R| from Cholesky
  logdetR <- 2 * sum(log(diag(L)))

  # solve R^{-1} * v via triangular solves
  solve_R <- function(v) {
    # solves R x = v
    backsolve(L, forwardsolve(t(L), v))
  }

  one   <- rep(1, m)
  Rinvy <- solve_R(y)
  Rinv1 <- solve_R(one)

  yRinvy   <- sum(y * Rinvy)
  oneRinv1 <- sum(one * Rinv1)
  yRinv1   <- sum(y * Rinv1)

  beta_hat <- yRinv1 / oneRinv1
  resid    <- y - beta_hat * one

  Rinv_resid       <- solve_R(resid)
  resid_Rinv_resid <- sum(resid * Rinv_resid)

  sigma2_hat <- resid_Rinv_resid / m
  if (sigma2_hat <= 0) return(-Inf)

  # restricted log-likelihood up to constant
  ll <- -0.5 * ((m - 1) * log(sigma2_hat) + logdetR + log(oneRinv1))
  ll
}

fit_reml_phi2 <- function(y, D, phi2_max = 10) {
  opt <- optimize(
    f        = reml_loglik_phi2,
    interval = c(1e-6, phi2_max),
    maximum  = TRUE,
    y = y,
    D = D
  )
  list(phi2_hat = opt$maximum, ll_max = opt$objective)
}

## ---- simulation setup ----

m_vals    <- c(5, 10, 50, 100, 200)
phi2_true <- c(0.2, 0.5, 1, 2, 5)
phi1_true <- 1
beta_true <- 0

n_rep <- 200  # reduce from 500 for speed; increase later if needed

results <- expand.grid(
  m    = m_vals,
  phi2 = phi2_true
)
results$prob_hat_zero <- NA_real_

## ---- main loop ----

for (i in seq_len(nrow(results))) {
  m    <- results$m[i]
  phi2 <- results$phi2[i]

  count_zero <- 0

  # locations fixed within this (m, phi2) cell: share D across reps
  coords <- cbind(runif(m), runif(m))
  D      <- as.matrix(dist(coords))

  for (rep in 1:n_rep) {
    Sigma <- exp_cov_phi(D, phi1 = phi1_true, phi2 = phi2)
    y     <- as.numeric(mvrnorm(1, mu = rep(beta_true, m), Sigma = Sigma))

    fit <- fit_reml_phi2(y, D, phi2_max = 10)
    if (fit$phi2_hat < 1e-4) count_zero <- count_zero + 1
  }

  results$prob_hat_zero[i] <- count_zero / n_rep
  cat("Done m =", m, ", phi2 =", phi2,
      ", prob_hat_zero =", results$prob_hat_zero[i], "\n")
}

## ---- plot ----

ggplot(results, aes(x = m, y = prob_hat_zero,
                    color = factor(phi2), group = phi2)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(trans = "log10", breaks = m_vals) +
  labs(x = "m (log scale)",
       y = "Estimated P(phi2_hat^R = 0)",
       color = "True phi2",
       title = "Probability that REML estimate of inverse range hits 0") +
  theme_minimal()
