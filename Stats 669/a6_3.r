# 25 locations on [0,1]^2: (i/4, j/4), i,j = 0,...,4
coords <- expand.grid(x = seq(0, 1, length.out = 5),
                      y = seq(0, 1, length.out = 5))

# index of the center point (0.5, 0.5)
center_idx <- which(coords$x == 0.5 & coords$y == 0.5)
center_idx

# pairwise Euclidean distances between all 25 sites
dmat <- as.matrix(dist(coords))   # 25 x 25 distance matrix

# generalized covariance matrix for a given phi2
Kphi <- function(phi2) {
  C <- gamma(-phi2) * (dmat^(2 * phi2))  # elementwise power

  # define K(0) = 0 on the diagonal (distance 0)
  diag(C) <- 0
  C
}

n <- nrow(coords)  # 25

D <- matrix(0, nrow = n, ncol = n - 1)   # 25 x 24

col <- 1
for (j in 1:n) {
  if (j == center_idx) next
  e <- rep(0, n)
  e[j] <-  1          # +1 at location j
  e[center_idx] <- -1 # -1 at center
  D[, col] <- e
  col <- col + 1
}
dim(D)   # should be 25 24

phi2_vals <- 1 - 0.1^(2:5)  # k = 2,3,4,5

eig_list <- list()

for (phi2 in phi2_vals) {
  C <- Kphi(phi2)               # 25 x 25
  Sigma <- t(D) %*% C %*% D     # 24 x 24 covariance of contrasts

  ev <- eigen(Sigma, symmetric = TRUE, only.values = TRUE)$values
  eig_list[[as.character(phi2)]] <- sort(ev, decreasing = TRUE)
}

eig_list

sapply(eig_list, function(v) v[1:5])


phi2_vals <- c(0.99, 0.999, 0.9999, 0.99999)

top_evectors <- list()

for (phi2 in phi2_vals) {
  C <- Kphi(phi2)            # 25 x 25 covariance (your function)
  Sigma <- t(D) %*% C %*% D  # 24 x 24 contrast covariance

  ee <- eigen(Sigma, symmetric = TRUE)   # values + vectors
  vals <- ee$values
  vecs <- ee$vectors                     # 24 x 24, columns = eigenvectors

  # store top 2 eigenvalues and eigenvectors
  top_evectors[[as.character(phi2)]] <- list(
    lambda = vals[1:2],
    vectors = vecs[, 1:2]   # 24 x 2
  )
}

phi2_vals <- c(0.99, 0.999, 0.9999, 0.99999)

# assuming top_evectors list already built as before
dot_list <- sapply(phi2_vals, function(phi2) {
  key <- as.character(phi2)
  v1 <- top_evectors[[key]]$vectors[, 1]
  v2 <- top_evectors[[key]]$vectors[, 2]
  sum(v1 * v2)
})

names(dot_list) <- phi2_vals
dot_list



## function: take eigenvector (length 24) -> 5x5 grid -> image plot
plot_eigenvector_grid <- function(phi2, k = 1) {
  key <- as.character(phi2)
  v24 <- top_evectors[[key]]$vectors[, k]  # kth eigenvector, length 24

  # insert 0 for the center contrast to make length 25
  v25 <- numeric(25)
  v25[-center_idx] <- v24

  # reshape to 5x5 matrix; coords are in expand.grid(x, y) order
  Vmat <- matrix(v25, nrow = 5, ncol = 5, byrow = FALSE)

  # plot as heatmap on [0,1]x[0,1]
  xgrid <- seq(0, 1, length.out = 5)
  ygrid <- seq(0, 1, length.out = 5)

  image(xgrid, ygrid, Vmat,
        main = bquote("Eigenvector " * .(k) *
                      " for " * phi[2] == .(phi2)),
        xlab = "x", ylab = "y", col = terrain.colors(20))
  contour(xgrid, ygrid, Vmat, add = TRUE, drawlabels = FALSE)
}

## example: plot top 2 eigenvectors for phi2 = 0.99
par(mfrow = c(1, 2), mar = c(3,3,3,1))
plot_eigenvector_grid(0.99, k = 1)
plot_eigenvector_grid(0.99, k = 2)
par(mfrow = c(1,1))

## optional: show both eigenvectors for all phi2 values in a 4x2 grid
par(mfrow = c(4, 2), mar = c(3,3,3,1))
for (phi2 in phi2_vals) {
  plot_eigenvector_grid(phi2, k = 1)
  plot_eigenvector_grid(phi2, k = 2)
}
par(mfrow = c(1,1))





# 7.6


set.seed(123)

# 1) Regular grid
design_grid <- expand.grid(x = seq(0, 1, length.out = 5),
                           y = seq(0, 1, length.out = 5))

# 2) Jittered grid
design_jitter <- within(design_grid, {
  x <- x + rnorm(length(x), sd = 0.03)
  y <- y + rnorm(length(y), sd = 0.03)
})


# 4) Central cluster
design_center <- data.frame(
  x = rnorm(25, mean = 0.5, sd = 0.12),
  y = rnorm(25, mean = 0.5, sd = 0.12)
)
design_center$x <- pmin(pmax(design_center$x, 0), 1)
design_center$y <- pmin(pmax(design_center$y, 0), 1)

designs <- list(
  grid    = design_grid,
  jitter  = design_jitter,
  uniform = design_uniform,
  center  = design_center
)

# pairwise distances for any design
dist_mat <- function(coords) as.matrix(dist(coords))

# Power law covariance and derivatives at given params
pl_cov_and_deriv <- function(dmat, phi1, phi2) {
  r <- dmat
  r2p <- r^(2 * phi2)
  C <- phi1 * gamma(-phi2) * r2p
  diag(C) <- 0

  # dC/dphi1
  C_phi1 <- C / phi1

  # dC/dphi2  (differentiate log gamma and r^(2phi2))
  dig <- digamma(-phi2)
  C_phi2 <- C * (-dig + 2 * log(pmax(r, 1e-8)))  # safe at 0
  diag(C_phi2) <- 0

  list(C = C, d1 = C_phi1, d2 = C_phi2)
}

# Exponential covariance and derivatives
exp_cov_and_deriv <- function(dmat, sigma2, rho) {
  r <- dmat
  E <- exp(-r / rho)
  C <- sigma2 * E

  C_s2 <- E
  C_rho <- sigma2 * E * (r / rho^2)

  list(C = C, d1 = C_s2, d2 = C_rho)
}

fisher_info_2par <- function(C, d1, d2) {
  n <- nrow(C)
  X <- matrix(1, n, 1)          # n x 1
  Cinv <- solve(C)              # n x n
  XtCinv <- t(X) %*% Cinv       # 1 x n

  # P = C^{-1} - C^{-1} X (X^T C^{-1} X)^{-1} X^T C^{-1}
  mid <- solve(XtCinv %*% X)    # 1 x 1
  P <- Cinv - Cinv %*% X %*% mid %*% XtCinv  # n x n

  A1 <- P %*% d1
  A2 <- P %*% d2

  I11 <- 0.5 * sum(A1 * (P %*% d1))
  I22 <- 0.5 * sum(A2 * (P %*% d2))
  I12 <- 0.5 * sum(A1 * (P %*% d2))

  matrix(c(I11, I12, I12, I22), nrow = 2, byrow = TRUE)
}

phi1 <- 1; phi2 <- 0.7
sigma2 <- 1; rho <- 0.3

info_power <- list()
info_exp   <- list()

for (nm in names(designs)) {
  coords <- designs[[nm]]
  dmat <- dist_mat(coords)

  # power law
  pl <- pl_cov_and_deriv(dmat, phi1, phi2)
  info_power[[nm]] <- fisher_info_2par(pl$C, pl$d1, pl$d2)

  # exponential
  ex <- exp_cov_and_deriv(dmat, sigma2, rho)
  info_exp[[nm]] <- fisher_info_2par(ex$C, ex$d1, ex$d2)
}

info_power
info_exp


summarize_I <- function(Ilist) {
  sapply(Ilist, function(M) {
    ev <- eigen(M, symmetric = TRUE)$values
    c(trace = sum(ev),
      det = prod(ev),
      cond = max(ev) / min(ev))
  })
}

power_summ <- summarize_I(info_power)
exp_summ   <- summarize_I(info_exp)

power_summ
exp_summ



