# Stats 596 HW2 - Pranav Tikkawar

library(MASS)
library(glmnet)

out <- "hw2_results"
dir.create(out, showWarnings = FALSE)

# write a data frame as a booktabs table that hw2.tex can \input
tex_table <- function(d, file, digits = 4) {
  d[] <- lapply(d, function(v) {
    if (is.numeric(v)) formatC(v, digits = digits, format = "f") else v
  })
  rows <- paste(apply(d, 1, paste, collapse = " & "), "\\\\")
  cat("\\begin{center}\\small\\setlength{\\tabcolsep}{4pt}",
      paste0("\\begin{tabular}{l", strrep("r", ncol(d) - 1), "}"),
      "\\toprule",
      paste(paste(names(d), collapse = " & "), "\\\\"),
      "\\midrule", rows, "\\bottomrule",
      "\\end{tabular}\\end{center}",
      file = file, sep = "\n")
}


#### Exercise 1.11 ####

ridge_mse <- function(mu, lam) (1 + lam^2 * mu^2) / (1 + lam)^2

lasso_mse <- function(mu, lam) {
  a <- -lam - mu
  b <- lam - mu
  (1 + lam^2) * (pnorm(a) + pnorm(b, lower.tail = FALSE)) -
    (a + 2 * lam) * dnorm(a) + (b - 2 * lam) * dnorm(b) +
    mu^2 * (pnorm(b) - pnorm(a))
}

# sanity check of the closed form against numerical integration
lasso_mse_int <- function(mu, lam) {
  a <- -lam - mu
  b <- lam - mu
  integrate(function(u) (u + lam)^2 * dnorm(u), -Inf, a)$value +
    mu^2 * (pnorm(b) - pnorm(a)) +
    integrate(function(u) (u - lam)^2 * dnorm(u), b, Inf)$value
}
for (lam in c(0.1, 1, 5, 10)) {
  for (mu in c(-3, 0, 2)) {
    stopifnot(abs(lasso_mse(mu, lam) - lasso_mse_int(mu, lam)) < 1e-6)
  }
}

lams <- c(0.1, 1, 5, 10)
cols <- c("black", "blue", "red", "darkgreen")
pdf(file.path(out, "ex111_mse.pdf"), width = 11, height = 9)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
for (zoom in c(FALSE, TRUE)) {
  mu <- if (zoom) seq(-3, 3, length.out = 601) else seq(-15, 15, length.out = 1201)
  for (method in c("Ridge", "Lasso")) {
    f <- if (method == "Ridge") ridge_mse else lasso_mse
    risk <- sapply(lams, function(l) f(mu, l))
    matplot(mu, risk, type = "l", lty = 1, lwd = 2, col = cols,
            ylim = if (zoom) c(0, 3) else range(risk),
            xlab = expression(mu), ylab = "MSE",
            main = paste0(method, if (zoom) ": near zero" else ": full range"))
    abline(h = 1, lty = 2, col = "gray50")
    legend("top", paste("lambda =", lams), col = cols, lwd = 2, bty = "n")
  }
}
dev.off()


#### Exercises 1.17 and 1.18 ####

# simulation code from the book, with p as an argument
make_data <- function(p) {
  set.seed(0)
  n <- 200
  mu <- rep(0, p)
  msig <- toeplitz(.8^(1:p - 1))
  sig <- .5
  x <- mvrnorm(n, mu, msig)
  y <- x[, 1] + pmax(0, apply(x[, 2:4], 1, sum))^2 / 3 + rnorm(n, 0, sig)
  ntest <- 100
  xtest <- mvrnorm(ntest, mu, msig)
  ytest <- xtest[, 1] + pmax(0, apply(xtest[, 2:4], 1, sum))^2 / 3 + rnorm(ntest, 0, sig)
  list(x = x, y = y, xtest = xtest, ytest = ytest)
}

# Ridge: minimize RSS/(2n) + (lambda/2)*||b||^2 with an unpenalized intercept.
# Center x and y (Ex 1.12), then use the SVD of the centered x:
#   b = V diag(d / (d^2 + n*lambda)) U'y,   df = 1 + sum d^2 / (d^2 + n*lambda)
# lambda = 0 gives the minimum-norm least squares fit.
ridge_fit <- function(x, y, lambda) {
  n <- nrow(x)
  xbar <- colMeans(x)
  ybar <- mean(y)
  s <- svd(sweep(x, 2, xbar))
  keep <- s$d > 1e-10 * s$d[1]   # drop zero singular values (p >= n case)
  d <- s$d[keep]
  uy <- drop(crossprod(s$u[, keep], y - ybar))
  beta <- sapply(lambda, function(l) s$v[, keep] %*% (d / (d^2 + n * l) * uy))
  beta <- matrix(beta, ncol = length(lambda))
  df <- sapply(lambda, function(l) 1 + sum(d^2 / (d^2 + n * l)))
  list(beta = beta, a0 = ybar - drop(xbar %*% beta), df = df)
}

# Lasso: glmnet solves RSS/(2n) + lambda*||b||_1. No standardization.
# thresh is tightened so the KKT check below passes at the small lambdas.
lasso_fit <- function(x, y, lambda) {
  fit <- glmnet(x, y, alpha = 1, lambda = lambda, standardize = FALSE,
                intercept = TRUE, thresh = 1e-14, maxit = 1e8)
  stopifnot(length(fit$lambda) == length(lambda))
  list(beta = as.matrix(fit$beta), a0 = fit$a0)
}

fit_path <- function(x, y, lambda, method) {
  if (method == "ridge") ridge_fit(x, y, lambda) else lasso_fit(x, y, lambda)
}

predict_path <- function(fit, x) sweep(x %*% fit$beta, 2, fit$a0, "+")
mse <- function(y, pred) colMeans((pred - y)^2)

# Largest violation of the Lasso optimality conditions (Ex 1.15) along the path:
#   g_j + lambda*sign(b_j) = 0 if b_j != 0,  |g_j| <= lambda if b_j = 0,  mean residual = 0
kkt_violation <- function(x, y, fit, lambda) {
  r <- predict_path(fit, x) - y
  g <- crossprod(x, r) / nrow(x)
  sapply(seq_along(lambda), function(k) {
    b <- fit$beta[, k]
    on <- abs(b) > 1e-8
    max(abs(g[on, k] + lambda[k] * sign(b[on])),
        pmax(abs(g[!on, k]) - lambda[k], 0),
        abs(mean(r[, k])))
  })
}

# K-fold CV: mean fold MSE, its SE, and the min / one-SE choices
cv_path <- function(x, y, lambda, method, folds) {
  K <- max(folds)
  err <- matrix(NA, K, length(lambda))
  for (k in 1:K) {
    tr <- folds != k
    fit <- fit_path(x[tr, ], y[tr], lambda, method)
    if (method == "lasso") {
      stopifnot(max(kkt_violation(x[tr, ], y[tr], fit, lambda)) < 1e-6 * lambda[1])
    }
    err[k, ] <- mse(y[!tr], predict_path(fit, x[!tr, ]))
  }
  cvm <- colMeans(err)
  se <- apply(err, 2, sd) / sqrt(K)
  imin <- which.min(cvm)
  i1se <- min(which(cvm <= cvm[imin] + se[imin]))   # lambda is decreasing
  list(cvm = cvm, se = se, imin = imin, i1se = i1se)
}

rules <- c("Test-grid oracle", "5-fold CV minimum", "5-fold CV 1-SE",
           "10-fold CV minimum", "10-fold CV 1-SE")

run_method <- function(dat, method, folds5, folds10) {
  x <- dat$x
  y <- dat$y
  n <- nrow(x)
  xc <- sweep(x, 2, colMeans(x))

  if (method == "ridge") {
    s <- svd(xc, nu = 0, nv = 0)$d[1]^2 / n
    lambda <- s * 10^seq(4, -8, length.out = 240)
  } else {
    lmax <- max(abs(crossprod(xc, y - mean(y)))) / n   # smallest lambda giving b = 0
    lambda <- lmax * 10^seq(0, -4, length.out = 240)
  }

  fit <- fit_path(x, y, lambda, method)
  train <- mse(y, predict_path(fit, x))
  test_sq <- (predict_path(fit, dat$xtest) - dat$ytest)^2
  test <- colMeans(test_sq)
  cv5 <- cv_path(x, y, lambda, method, folds5)
  cv10 <- cv_path(x, y, lambda, method, folds10)

  if (method == "ridge") {
    axis <- fit$df
  } else {
    # shrinkage fraction relative to the min-norm least squares fit
    ls <- ridge_fit(x, y, 0)$beta
    axis <- colSums(abs(fit$beta)) / sum(abs(ls))
    stopifnot(max(kkt_violation(x, y, fit, lambda)) < 1e-6 * lmax)
    stopifnot(all(axis <= 1 + 1e-6), all(diff(axis) >= -1e-6))   # proved in 1.18
  }

  sel <- c(which.min(test), cv5$imin, cv5$i1se, cv10$imin, cv10$i1se)
  tab <- data.frame(Selection = rules,
                    lambda = formatC(lambda[sel], digits = 4, format = "g"),
                    axis = formatC(axis[sel], digits = 3, format = "g"),
                    test = test[sel])
  names(tab) <- c("Selection", "$\\lambda$",
                  if (method == "ridge") "df" else "$f(\\lambda)$", "Test MSE")
  if (method == "lasso") {
    nz <- abs(fit$beta[, sel]) > 1e-8
    tab <- data.frame(tab[1:3],
                      Nonzero = as.character(colSums(nz)),
                      Kept = apply(nz[1:4, ], 2, function(v) paste(which(v), collapse = ",")),
                      tab[4], check.names = FALSE)
    names(tab)[5] <- "Kept of $x_1$--$x_4$"
  }

  list(method = method, lambda = lambda, fit = fit, axis = axis,
       train = train, test = test, cv5 = cv5, cv10 = cv10,
       sel = sel, tab = tab, test_sq = test_sq[, sel])
}

plot_method <- function(z, p) {
  stem <- file.path(out, paste0("p", p, "_", z$method))
  xlab <- if (z$method == "ridge") "Degrees of freedom" else "L1 shrinkage fraction"
  B <- t(z$fit$beta)
  sig_cols <- c("black", "red", "blue", "darkgreen")

  draw_path <- function(idx, main) {
    matplot(z$axis[idx], B[idx, -(1:4)], type = "l", lty = 1, col = "gray75",
            ylim = range(B[idx, ]), xlab = xlab, ylab = "Coefficient", main = main)
    matlines(z$axis[idx], B[idx, 1:4], lty = 1, lwd = 2.5, col = sig_cols)
    abline(h = 0, col = "gray50")
    abline(v = z$axis[c(z$cv10$imin, z$cv10$i1se)], col = "purple", lty = c(2, 3))
    legend("topleft", c(paste0("x", 1:4), "other x", "10-fold min", "10-fold 1-SE"),
           col = c(sig_cols, "gray75", "purple", "purple"),
           lty = c(1, 1, 1, 1, 1, 2, 3), bty = "n", cex = .75)
  }

  # p = 200 ridge nearly interpolates at small lambda, so add a zoomed panel
  zoom <- z$method == "ridge" && p == 200
  pdf(paste0(stem, "_path.pdf"), width = if (zoom) 11 else 8, height = 5.2)
  par(mfrow = c(1, if (zoom) 2 else 1), mar = c(4.5, 4, 3, 1))
  draw_path(seq_along(z$axis), paste(z$method, "path, p =", p))
  if (zoom) {
    draw_path(which(z$lambda >= z$lambda[z$cv10$imin] / 100), "Without smallest penalties")
  }
  dev.off()

  E <- cbind(z$train, z$test, z$cv5$cvm, z$cv10$cvm)
  pdf(paste0(stem, "_errors.pdf"), width = 11, height = 5)
  par(mfrow = c(1, 2), mar = c(4.5, 4, 3, 1))
  for (detail in c(FALSE, TRUE)) {
    ylim <- range(E)
    xlim <- range(z$axis)
    if (detail) {
      ylim <- c(0, max(1, 2.5 * min(E[, 2:4])))
      if (z$method == "lasso") {
        s <- z$axis[z$sel]
        w <- diff(range(s))
        xlim <- c(max(0, min(s) - w), min(max(z$axis), max(s) + w))
      }
    }
    matplot(z$axis, E, type = "l", lty = c(1, 1, 2, 3), lwd = 2,
            col = c("black", "red", "blue", "darkgreen"),
            xlim = xlim, ylim = ylim, xlab = xlab, ylab = "MSE",
            main = if (detail) "Detail" else "Full range")
    abline(v = z$axis[c(z$cv5$imin, z$cv5$i1se)], col = "blue", lty = c(2, 3))
    abline(v = z$axis[c(z$cv10$imin, z$cv10$i1se)], col = "darkgreen", lty = c(2, 3))
    legend("topleft", c("Train", "Test", "5-fold CV", "10-fold CV"),
           col = c("black", "red", "blue", "darkgreen"), lty = c(1, 1, 2, 3),
           lwd = 2, bty = "n", cex = .8)
  }
  dev.off()
}

for (p in c(20, 100, 200)) {
  cat("\n---- p =", p, "----\n")
  dat <- make_data(p)

  # same folds for ridge and lasso; drawn after the data
  set.seed(123)
  perm <- sample.int(200)
  folds5 <- folds10 <- integer(200)
  folds5[perm] <- rep(1:5, length.out = 200)
  folds10[perm] <- rep(1:10, length.out = 200)

  res <- list()
  for (method in c("ridge", "lasso")) {
    z <- run_method(dat, method, folds5, folds10)
    plot_method(z, p)
    tex_table(z$tab, file.path(out, paste0("p", p, "_", method, "_selection.tex")))
    print(z$tab, row.names = FALSE)
    res[[method]] <- z
  }

  # paired comparison on the same test points
  d <- res$ridge$test_sq - res$lasso$test_sq
  se <- apply(d, 2, sd) / sqrt(nrow(d))
  paired <- data.frame(Selection = rules,
                       Ridge = colMeans(res$ridge$test_sq),
                       Lasso = colMeans(res$lasso$test_sq),
                       Diff = colMeans(d), SE = se, z = colMeans(d) / se)
  names(paired)[4] <- "Ridge $-$ Lasso"
  tex_table(paired, file.path(out, paste0("p", p, "_paired.tex")), digits = 3)
  print(paired, row.names = FALSE, digits = 4)
}


#### Best linear predictor benchmark ####
# b* = (1, c, c, c, 0, ...) by Stein's lemma; S = x2 + x3 + x4 ~ N(0, v)
v <- 3 + 2 * (.8 + .8 + .64)
c_star <- (2 / 3) * sqrt(v) * dnorm(0)
floor_risk <- .25 + v^2 * (5 / 36 - 2 / (9 * pi))

# Monte Carlo check of the floor
set.seed(2026)
N <- 1e6
xb <- mvrnorm(N, rep(0, 20), toeplitz(.8^(0:19)))
sb <- rowSums(xb[, 2:4])
yb <- xb[, 1] + pmax(0, sb)^2 / 3 + rnorm(N, 0, .5)
e2 <- (yb - v / 6 - xb[, 1] - c_star * sb)^2
cat("\nc =", c_star, " linear floor =", floor_risk,
    " Monte Carlo =", mean(e2), "( SE", sd(e2) / sqrt(N), ")\n")