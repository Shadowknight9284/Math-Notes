# Homework 3 -- Exercises 2.3, 2.4, and 2.5
# Run this entire script in R. MASS is the only required package.
library(MASS)
options(contrasts = c("contr.treatment", "contr.poly"))
d <- MASS::birthwt
d$race <- factor(d$race, levels = 1:3,
                 labels = c("white", "black", "other"))
d$ptd <- factor(d$ptl > 0, levels = c(FALSE, TRUE))
d$ftv <- factor(ifelse(d$ftv >= 2, "2+", as.character(d$ftv)),
                levels = c("0", "1", "2+"))
form <- low ~ age + lwt + race + smoke + ptd + ht + ui + ftv
X <- model.matrix(form, data = d)
y <- d$low
n <- nrow(X)
p <- ncol(X)
stopifnot(n == 189, p == 11, qr(X)$rank == p)
softplus <- function(z) pmax(z, 0) + log1p(exp(-abs(z)))

parts <- function(beta, link) {
  eta <- drop(X %*% beta)
  if (link == "logit") {
    mu <- plogis(eta)
    ll <- sum(ifelse(y == 1, -softplus(-eta), -softplus(eta)))
    u <- y - mu
    wo <- wf <- mu * (1 - mu)
  } else if (link == "probit") {
    s <- 2 * y - 1
    t <- s * eta
    lp <- pnorm(t, log.p = TRUE)
    r <- exp(dnorm(t, log = TRUE) - lp)
    ll <- sum(lp)
    mu <- pnorm(eta)
    u <- s * r
    wo <- r * (r + t)
    wf <- exp(2 * dnorm(eta, log = TRUE) -
              pnorm(eta, log.p = TRUE) -
              pnorm(eta, lower.tail = FALSE, log.p = TRUE))
  } else stop("Unknown link")
  list(ll = ll, mu = mu, u = u, wo = wo, wf = wf,
       U = drop(crossprod(X, u)))
}
weighted_info <- function(w) crossprod(X, sweep(X, 1, w, "*"))
Rmm <- chol(crossprod(X) / 4)
solve_mm <- function(u) backsolve(Rmm, forwardsolve(t(Rmm), u))

fit_custom <- function(link = "logit", algorithm = "NR",
                       start = "zero", criterion = "beta",
                       tol = 1e-6, maxit = 200000L) {
  stopifnot(link %in% c("logit", "probit"),
            algorithm %in% c("NR", "FS", "MM"),
            start %in% c("zero", "LS"),
            criterion %in% c("beta", "loglik"))
  if (algorithm == "MM" && link != "logit") stop("MM is logit only")
  beta <- if (start == "zero") rep(0, p) else drop(qr.solve(X, y))
  old <- parts(beta, link)
  for (k in seq_len(maxit)) {
    step <- if (algorithm == "MM") solve_mm(old$U) else {
      w <- if (algorithm == "NR") old$wo else old$wf
      solve(weighted_info(w), old$U)
    }
    newbeta <- beta + drop(step)
    new <- parts(newbeta, link)
    if (!all(is.finite(newbeta)) || !is.finite(new$ll))
      stop("Nonfinite iterate; no step-halving is used")
    db <- sqrt(sum((newbeta - beta)^2))
    dl <- abs(new$ll - old$ll) / n
    err <- if (criterion == "beta") db else dl
    beta <- newbeta
    old <- new
    if (err <= tol) return(list(beta = setNames(beta, colnames(X)),
      iterations = k, link = link, algorithm = algorithm,
      start = start, criterion = criterion, tol = tol,
      loglik = new$ll, beta_change = db,
      average_loglik_change = dl))
  }
  stop("Maximum iterations reached")
}

# Exercise 2.3: 32 runs, with coefficients for every stopping rule.
methods <- data.frame(link = c("logit", "logit", "probit", "probit"),
                      algorithm = c("NR", "MM", "NR", "FS"))
runs <- list()
for (j in seq_len(nrow(methods))) {
  for (start in c("zero", "LS")) {
    for (criterion in c("beta", "loglik")) {
      for (tol in c(1e-3, 1e-6)) {
        f <- fit_custom(methods$link[j], methods$algorithm[j],
                        start, criterion, tol)
        runs[[length(runs) + 1L]] <- data.frame(
          link = f$link, algorithm = f$algorithm, start = f$start,
          criterion = f$criterion, tolerance = f$tol,
          iterations = f$iterations, average_loglik = f$loglik / n,
          beta_change = f$beta_change,
          average_loglik_change = f$average_loglik_change,
          as.list(f$beta), check.names = FALSE)
      }
    }
  }
}
results23 <- do.call(rbind, runs)
print(results23, digits = 8, row.names = FALSE)
write.csv(results23, "exercise_2_3_all_runs.csv", row.names = FALSE)

# Exercises 2.4 and 2.5.
fit_logit <- fit_custom("logit", "NR", "zero", "beta", 1e-10)
fit_probit <- fit_custom("probit", "FS", "zero", "beta", 1e-10)
inference <- function(fit) {
  a <- parts(fit$beta, fit$link)
  Jo <- weighted_info(a$wo)
  If <- weighted_info(a$wf)
  scores <- sweep(X, 1, a$u, "*")
  B <- crossprod(scores)
  Ainv <- solve(Jo)
  Vmodel <- solve(If)
  Vrobust <- Ainv %*% B %*% t(Ainv)
  p0 <- mean(y)
  ll0 <- sum(y * log(p0) + (1-y) * log1p(-p0))
  tab <- data.frame(Estimate = fit$beta,
                    SE_model = sqrt(diag(Vmodel)),
                    SE_robust_HC0 = sqrt(diag(Vrobust)),
                    SE_robust_HC1 = sqrt(n / (n-p) * diag(Vrobust)))
  list(coefficients = tab, null_deviance = -2 * ll0,
       residual_deviance = -2 * a$ll, null_df = n-1,
       residual_df = n-p, iterations = fit$iterations,
       Vmodel = Vmodel, Vrobust = Vrobust)
}
out_logit <- inference(fit_logit)
out_probit <- inference(fit_probit)
for (name in c("logit", "probit")) {
  out <- if (name == "logit") out_logit else out_probit
  cat("\nLINK:", name, "\n")
  print(out$coefficients, digits = 8)
  print(out[c("null_deviance", "residual_deviance", "null_df",
              "residual_df", "iterations")])
  write.csv(out$coefficients, paste0("exercises_2_4_2_5_", name, ".csv"))
}

# Generate LaTeX tables for inclusion in the solution document.
latex_table <- function(header, body, caption, filename, align) {
  lines <- c("\\begin{table}[htbp]", "\\centering", "\\small",
    paste0("\\caption{", caption, "}"),
    paste0("\\begin{tabular}{", align, "}"), "\\toprule",
    paste0(paste(header, collapse = " & "), " \\\\"), "\\midrule",
    apply(body, 1, function(z) paste0(paste(z, collapse = " & "), " \\\\")),
    "\\bottomrule", "\\end{tabular}", "\\end{table}")
  writeLines(lines, filename)
}
summary23 <- results23[, c("link", "algorithm", "start", "criterion",
                          "tolerance", "iterations", "average_loglik")]
summary23$tolerance <- format(summary23$tolerance, scientific = TRUE)
summary23$average_loglik <- sprintf("%.8f", summary23$average_loglik)
latex_table(c("Link", "Method", "Start", "Rule", "Tolerance", "$k$", "$K_n$"),
  as.matrix(summary23), "Exercise 2.3: convergence results.",
  "exercise_2_3_iterations.tex", "llllrrr")

# Longtable containing coefficients for all 32 runs, in a landscape page.
coefnames <- colnames(X)
coefnames[coefnames == "(Intercept)"] <- "Intercept"
coeflines <- c("\\begin{landscape}", "\\scriptsize",
  "\\begin{longtable}{rlllllllllll}",
  "\\caption{Exercise 2.3: coefficient estimates. Run numbers follow the convergence table.}\\\\",
  "\\toprule",
  paste0(paste(c("Run", coefnames), collapse = " & "), " \\\\"),
  "\\midrule", "\\endfirsthead", "\\toprule",
  paste0(paste(c("Run", coefnames), collapse = " & "), " \\\\"),
  "\\midrule", "\\endhead")
for (i in seq_len(nrow(results23))) {
  vals <- as.numeric(unlist(results23[i, colnames(X), drop = FALSE]))
  coeflines <- c(coeflines,
    paste0(paste(c(i, sprintf("%.6f", vals)), collapse = " & "), " \\\\"))
}
coeflines <- c(coeflines, "\\bottomrule", "\\end{longtable}", "\\end{landscape}")
writeLines(coeflines, "exercise_2_3_coefficients.tex")
for (name in c("logit", "probit")) {
  out <- if (name == "logit") out_logit else out_probit
  body <- cbind(coefnames, apply(out$coefficients, 2,
                                function(z) sprintf("%.6f", z)))
  latex_table(c("Term", "Estimate", "Model SE", "HC0 SE", "HC1 SE"),
    body, paste("Exercises 2.4--2.5:", name, "regression."),
    paste0("inference_", name, ".tex"), "lrrrr")
}

# Optional validation only; glm is not used to compute the results above.
g_logit <- glm(form, data = d, family = binomial("logit"),
               control = glm.control(epsilon = 1e-12, maxit = 100))
g_probit <- glm(form, data = d, family = binomial("probit"),
                control = glm.control(epsilon = 1e-12, maxit = 100))
print(c(logit_beta_error = max(abs(coef(g_logit) - fit_logit$beta)),
        probit_beta_error = max(abs(coef(g_probit) - fit_probit$beta))))
