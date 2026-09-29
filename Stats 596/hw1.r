# Stats 596 - Homework 1

library(MASS)
set.seed(0)

n <- 200
p <- 4
mu <- rep(0, p)
msig <- toeplitz(.8^(1:p-1))
sig <- .5

x <- mvrnorm(n, mu, msig)
y <- x[,1] + pmax(0, apply(x[,2:p], 1, sum))^2 / 3 +
  rnorm(n, 0, sig)

# First three rows requested in 1.8(a)
first_three_rows <- cbind(y, x)[1:3, ]
print(first_three_rows)

ols_stats <- function(x, y) {
  X <- cbind(1, x)
  n <- nrow(X)
  k <- ncol(X)
  XtX_inv <- solve(crossprod(X))
  beta_hat <- drop(XtX_inv %*% crossprod(X, y))
  residual <- drop(y - X %*% beta_hat)

  V_model <- sum(residual^2) / (n - k) * XtX_inv
  V_robust <- XtX_inv %*%
    crossprod(X, X * residual^2) %*% XtX_inv

  list(
    beta = beta_hat,
    residual = residual,
    V_model = V_model,
    V_robust = V_robust,
    X = X
  )
}

ror_stats <- function(X, y) {
  n <- nrow(X)
  k <- ncol(X)
  full <- ols_stats(X[, -1, drop = FALSE], y)
  out <- matrix(NA_real_, k, 3)

  for (j in seq_len(k)) {
    Z <- X[, j]
    W <- X[, -j, drop = FALSE]
    RZ <- drop(Z - W %*%
      solve(crossprod(W), crossprod(W, Z)))

    out[j, 1] <- sum(RZ * y) / sum(RZ^2)
    out[j, 2] <- sqrt(
      sum(full$residual^2) / (n - k) / sum(RZ^2)
    )
    out[j, 3] <- sqrt(
      sum(RZ^2 * full$residual^2) / sum(RZ^2)^2
    )
  }

  colnames(out) <- c("estimate", "model_se", "robust_se")
  rownames(out) <- colnames(X)
  out
}

fit <- ols_stats(x, y)
standard <- cbind(
  estimate = fit$beta,
  model_se = sqrt(diag(fit$V_model)),
  robust_se = sqrt(diag(fit$V_robust))
)
rownames(standard) <- colnames(fit$X)

ror <- ror_stats(fit$X, y)

cat("\nExercise 1.8(a): OLS coefficients and SEs\n")
print(standard)
cat("\nExercise 1.8(b): Residual-on-residual results\n")
print(ror)
cat("\nMax absolute difference between direct and RoR calculations:\n")
print(max(abs(standard - ror)))

# Exercise 1.8(c): random-design population targets
coef_names <- c("(Intercept)", paste0("x", 1:4))
a <- c(0, 1, 1, 1)
vS <- drop(t(a) %*% msig %*% a)
lambda <- sqrt(vS) * sqrt(2 / pi) / 3
target_random <- c(vS / 6, 1, rep(lambda, 3))
names(target_random) <- coef_names

cat("\nExercise 1.8(c): Random-design population targets\n")
print(target_random)

summarize_mc <- function(B, V_model, V_robust, target) {
  z90 <- qnorm(.95)
  z95 <- qnorm(.975)
  coverage <- function(V, z) {
    colMeans(abs(sweep(B, 2, target, "-")) <= z * sqrt(V))
  }

  data.frame(
    target = target,
    mean = colMeans(B),
    bias = colMeans(B) - target,
    mc_sd = apply(B, 2, sd),
    model_root_mean_var = sqrt(colMeans(V_model)),
    model_cov90 = coverage(V_model, z90),
    model_cov95 = coverage(V_model, z95),
    robust_root_mean_var = sqrt(colMeans(V_robust)),
    robust_cov90 = coverage(V_robust, z90),
    robust_cov95 = coverage(V_robust, z95)
  )
}

# Exercise 1.9: random-design simulation
set.seed(0)
m <- 2000
B <- V_model <- V_robust <- matrix(NA_real_, m, 5)
for (s in seq_len(m)) {
  x_s <- mvrnorm(n, mu, msig)
  y_s <- x_s[, 1] +
    pmax(0, rowSums(x_s[, 2:4, drop = FALSE]))^2 / 3 +
    rnorm(n, 0, sig)

  fit_s <- ols_stats(x_s, y_s)
  B[s, ] <- fit_s$beta
  V_model[s, ] <- diag(fit_s$V_model)
  V_robust[s, ] <- diag(fit_s$V_robust)
}

random_results <- summarize_mc(B, V_model, V_robust, target_random)
rownames(random_results) <- coef_names

cat("\nExercise 1.9: Random-design Monte Carlo results\n")
print(random_results)

# Exercise 1.10: fixed-design simulation
set.seed(0)
x_fixed <- mvrnorm(n, mu, msig)
X_fixed <- cbind(1, x_fixed)
colnames(X_fixed) <- coef_names
mean_y_fixed <- x_fixed[, 1] +
  pmax(0, rowSums(x_fixed[, 2:4, drop = FALSE]))^2 / 3

target_fixed <- drop(
  solve(crossprod(X_fixed), crossprod(X_fixed, mean_y_fixed))
)
names(target_fixed) <- coef_names

m <- 2000
B <- V_model <- V_robust <- matrix(NA_real_, m, 5)
for (s in seq_len(m)) {
  y_s <- mean_y_fixed + rnorm(n, 0, sig)
  fit_s <- ols_stats(x_fixed, y_s)

  B[s, ] <- fit_s$beta
  V_model[s, ] <- diag(fit_s$V_model)
  V_robust[s, ] <- diag(fit_s$V_robust)
}

fixed_results <- summarize_mc(B, V_model, V_robust, target_fixed)
rownames(fixed_results) <- coef_names

cat("\nExercise 1.10(a): Fixed-design target\n")
print(target_fixed)
cat("\nExercise 1.10(b)-(d): Fixed-design Monte Carlo results\n")
print(fixed_results)
