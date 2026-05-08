library(terra)
library(doParallel)
library(foreach)

# Set wd
setwd("C:/Users/prana/OneDrive/Desktop/MathNotes/Stats 669")

ourdir <- "hw9/output"
dir.create(ourdir, showWarnings = FALSE, recursive = TRUE)

tif_file <- "data/USGS_13_n37w118_20260112.tif"
raster_data <- rast(tif_file)

# Approach: aggregate to a coarser grid, then crop by row/col indices
nr <- nrow(raster_data)
nc <- ncol(raster_data)
cat("Full raster size:", nr, "x", nc, "\n")
# 1. Decide how many rows/cols you want; 2000x2000
target_size <- 2000L
row_factor <- floor(nr / target_size)
col_factor <- floor(nc / target_size)
fact <- min(row_factor, col_factor)

# 2. Aggregate to reduce the raster so that it has at least 2000x2000 but not fewer cells per dimension
r_agg <- aggregate(raster_data, fact = fact, fun = mean)

nr_a <- nrow(r_agg); nc_a <- ncol(r_agg)
cat("Aggregated raster size:", nr_a, "x", nc_a, "\n")

target_size <- min(2000L, nr_a, nc_a)
half <- target_size / 2L

center_row <- nr_a %/% 2
center_col <- nc_a %/% 2

row_start <- center_row - (half - 1L)
row_end   <- center_row + half
col_start <- center_col - (half - 1L)
col_end   <- center_col + half

dem_2k <- crop(
  r_agg,
  ext(
    xFromCol(r_agg, col_start),
    xFromCol(r_agg, col_end + 1),
    yFromRow(r_agg, row_end + 1),
    yFromRow(r_agg, row_start)
  )
)

Z <- as.matrix(dem_2k, wide = TRUE)
cat("Final cropped size:", nrow(Z), "x", ncol(Z), "\n")


# M matrix for 6.8 with diff ALCs
create_M <- function(coords, irf_order) {
  x <- coords$x; y <- coords$y
  if (irf_order == 0) return(as.matrix(cbind(1)))
  if (irf_order == 1) return(as.matrix(cbind(1, x, y)))
  if (irf_order == 2) return(as.matrix(cbind(1, x, y, x^2, y^2, x * y)))
  stop("Error: irf_order must be 0, 1, or 2")
}

restricted_log_lik <- function(kappa, z_vec, D, M) {
  n <- length(z_vec)  
  p <- ncol(M)        
  
  # A. Generalized Covariancewith IRF Sign Rules
  # (this is a generalization of the gamma(-kappa) since that has computation issues)
  if (kappa > 0 && kappa < 2) {
    Omega <- -(D^kappa)
  } else if (kappa >= 2 && kappa < 4) {
    Omega <- +(D^kappa)
  } else if (kappa >= 4){
    Omega <- -(D^kappa)
  }
  #in case diagonal is 0 fix it
  diag(Omega) <- diag(Omega) + 1e-6 
  
  Omega_inv <- tryCatch(solve(Omega), error = function(e) NULL)
  if (is.null(Omega_inv)) return(1e15)
  
  MO  <- t(M) %*% Omega_inv           
  MOM <- MO %*% M                     
  
  MOM_inv <- tryCatch(solve(MOM), error = function(e) NULL)
  if (is.null(MOM_inv)) return(1e15)
  
  beta_hat  <- MOM_inv %*% MO %*% z_vec
  residuals <- z_vec - (M %*% beta_hat)
  
  QF    <- t(residuals) %*% Omega_inv %*% residuals
  theta <- as.numeric(QF / (n - p))
  if (theta <= 0) return(1e15)
  
  # 6.8
  logdet_Omega <- as.numeric(determinant(Omega, logarithm = TRUE)$modulus)
  logdet_MOM   <- as.numeric(determinant(MOM, logarithm = TRUE)$modulus)
  
  treml <- 0.5 * logdet_Omega + 0.5 * logdet_MOM + ((n - p) / 2) * log(as.numeric(QF))
  return(as.numeric(treml))
}

grid_size <- 50
global_coords <- expand.grid(x = seq_len(grid_size), y = seq_len(grid_size))

D_global <- as.matrix(dist(global_coords))
M_global <- create_M(global_coords, irf_order = 2) # IRF-2



cat("Chopping map into blocks...\n")

# Divide by 50 because we extract 50x50 chunks from Z
n_row_blocks <- floor(nrow(Z) / 50)
n_col_blocks <- floor(ncol(Z) / 50)

# === 100 BLOCKS FIX: Hardcoded to a 10x10 grid from the top-left ===
#blocks_df <- expand.grid(row_blk = 1:10, col_blk = 1:10)
blocks_df <- expand.grid(row_blk = seq_len(n_row_blocks), col_blk = seq_len(n_col_blocks))

z_blocks_list <- lapply(seq_len(nrow(blocks_df)), function(i) {
  r <- blocks_df$row_blk[i]
  c <- blocks_df$col_blk[i]
  
  r_idx <- ((r - 1) * 50 + 1):(r * 50)
  c_idx <- ((c - 1) * 50 + 1):(c * 50)
  sub_matrix <- Z[r_idx, c_idx]
  
  sub_matrix <- sub_matrix[seq(1, 50, by = 2), seq(1, 50, by = 2)]
  
  return(as.vector(sub_matrix))
})

# Delete the giant original data from RAM
rm(Z, raster_data, dem_sub)
gc()

## ------------------------------------------------------------
## 4. Parallel Execution
## ------------------------------------------------------------
cat("Starting parallel optimization...\n")
n_cores <- max(1L, parallel::detectCores() - 1L)
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Notice we no longer export "Z" or "get_block_z"
results_reml <- foreach(
  i = seq_along(z_blocks_list),
  .export = c("restricted_log_lik", "D_global", "M_global"),
  .combine = rbind,
  .errorhandling = "pass"
) %dopar% {
  
  # Grab just the tiny vector for this specific core
  z_vec <- z_blocks_list[[i]]
  
  opt <- optim(
    par = 1.5, 
    fn = restricted_log_lik, 
    z_vec = z_vec, 
    D = D_global, 
    M = M_global, 
    method = "Brent", 
    lower = 0.01, 
    upper = 5.99
  )
  
  # Return the results
  data.frame(
    row_block   = blocks_df$row_blk[i],
    col_block   = blocks_df$col_blk[i],
    kappa_hat   = round(opt$par, 2), 
    reml_loglik = round(opt$value, 2),
    converged   = opt$convergence == 0
  )
}

stopCluster(cl)


###################### HISTOGRAM
reml_df <- as.data.frame(results_reml)

cat("\nSummary of Power/Smoothness (Kappa) Estimates (IRF-2):\n")
print(summary(reml_df$kappa_hat))

out_csv <- file.path(ourdir, "hw9_reml_powerlaw_irf2_results.csv")
write.csv(reml_df, out_csv, row.names = FALSE)

png(file.path(ourdir, "hw9_kappa_histogram_irf2_reml.png"), width = 700, height = 600, res = 96)
hist(
  reml_df$kappa_hat/2,
  breaks = 30,
  col = "steelblue",
  border = "white",
  xlab = "Kappa",
  ylab = "Frequency",
  main = "Histogram of Kappa Estimates (IRF-2 Power-Law)",
  font.main = 2
)
dev.off()