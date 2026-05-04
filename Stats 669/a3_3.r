## Stats 669 – Death Valley elevation diagnostics
## Data: USGS_13_n37w118_20260112.tif

library(terra)

## 1. Load raster and build 2000 x 2000 subset -----------------------------

raster_data <- rast("Stats 669/data/USGS_13_n37w118_20260112.tif")

nc <- ncol(raster_data)
nr <- nrow(raster_data)

center_col <- nc %/% 2
center_row <- nr %/% 2

start_col <- center_col - 999
end_col   <- center_col + 1000
start_row <- center_row - 999
end_row   <- center_row + 1000

dem_2k <- crop(
  raster_data,
  ext(xFromCol(raster_data, start_col),
      xFromCol(raster_data, end_col + 1),
      yFromRow(raster_data, end_row + 1),
      yFromRow(raster_data, start_row))
)

# Matrix of elevations (meters)
z <- as.matrix(dem_2k, wide = TRUE)

## 2. Coordinate system in kilometers --------------------------------------

# Grid spacing in degrees
dx_deg <- xres(dem_2k)
dy_deg <- yres(dem_2k)

# Mid latitude and spacing in km (1 deg lat ~ 111.2 km; 1 deg lon ~ 111.2*cos(lat))
lat_min <- ymin(dem_2k)
lat_max <- ymax(dem_2k)
mid_lat <- (lat_min + lat_max) / 2

dy_km <- dy_deg * 111.2
dx_km <- dx_deg * 111.2 * cos(mid_lat * pi/180)

# Coordinate vectors (in km)
x_km <- seq(0, dx_km * (ncol(z) - 1), length.out = ncol(z))
y_km <- seq(0, dy_km * (nrow(z) - 1), length.out = nrow(z))

x_lim <- range(x_km)
y_lim <- range(y_km)

## 3. Helper functions ------------------------------------------------------

# squared second differences at lag 1
sq2diff <- function(v) {
  d2 <- v[-c(1, length(v))] - 2 * v[-c(1, 2)] + v[-c(length(v)-1, length(v))]
  d2^2
}

# squared second differences at arbitrary lag r (for kappa estimation)
sq2diff_lag <- function(v, r = 1) {
  n <- length(v)
  idx <- (1 + r):(n - r)
  d2 <- v[idx + r] - 2 * v[idx] + v[idx - r]
  d2^2
}

# estimate kappa for one 1D transect using multiple lags
kappa_transect <- function(v, lags = 1:3) {
  ms2 <- numeric(length(lags))
  for (k in seq_along(lags)) {
    s2 <- sq2diff_lag(v, r = lags[k])
    ms2[k] <- mean(s2, na.rm = TRUE)
  }
  ok <- is.finite(ms2) & (ms2 > 0)
  if (sum(ok) < 2) return(NA_real_)
  l_r   <- log(lags[ok])
  l_ms2 <- log(ms2[ok])
  fit <- lm(l_ms2 ~ l_r)
  beta <- coef(fit)[2]      # slope in log-log
  kappa_hat <- beta / 2     # because E[(Δ²Z)^2] ∝ r^{2κ}
  as.numeric(kappa_hat)
}

## 4. Indices for interior rows/columns ------------------------------------

ns_idx <- 2:(nrow(z) - 1)   # rows (northing)
ew_idx <- 2:(ncol(z) - 1)   # columns (easting)

## 5. Figure 3.18 – mean squared 2nd differences ---------------------------

# East–west transects (columns): function of Easting
ms2_ew <- numeric(length(ew_idx))
for (j in seq_along(ew_idx)) {
  col_j <- z[, ew_idx[j]]
  s2 <- sq2diff(col_j)
  ms2_ew[j] <- mean(s2, na.rm = TRUE)
}
east_km_mid <- x_km[ew_idx]

# North–south transects (rows): function of Northing
ms2_ns <- numeric(length(ns_idx))
for (i in seq_along(ns_idx)) {
  row_i <- z[ns_idx[i], ]
  s2 <- sq2diff(row_i)
  ms2_ns[i] <- mean(s2, na.rm = TRUE)
}
north_km_mid <- y_km[ns_idx]

png("Stats 669/img/fig3_18_ms2_easting_northing.png", width = 900, height = 700)
par(mfrow = c(2,1), mar = c(4,4,2,1))

plot(east_km_mid, ms2_ew, log = "y", type = "l",
     xlab = "Easting (km)",
     ylab = expression("Mean squared 2nd difference ("*m^2*")"),
     xlim = x_lim,
     main = "Mean squared 2nd differences along north–south transects")

plot(north_km_mid, ms2_ns, log = "y", type = "l",
     xlab = "Northing (km)",
     ylab = expression("Mean squared 2nd difference ("*m^2*")"),
     xlim = y_lim,
     main = "Mean squared 2nd differences along east–west transects")

dev.off()

## 6. Figure 3.17 – kappa via phi r^{2 kappa} (multi-lag) ------------------

# kappa from north–south transects (columns): function of Easting
kappa_ew <- numeric(length(ew_idx))
for (j in seq_along(ew_idx)) {
  col_j <- z[, ew_idx[j]]
  kappa_ew[j] <- kappa_transect(col_j, lags = 1:3)
}
# kappa from east–west transects (rows): function of Northing
kappa_ns <- numeric(length(ns_idx))
for (i in seq_along(ns_idx)) {
  row_i <- z[ns_idx[i], ]
  kappa_ns[i] <- kappa_transect(row_i, lags = 1:3)
}

png("Stats 669/img/fig3_17_kappa_easting_northing.png", width = 900, height = 700)
par(mfrow = c(2,1), mar = c(4,4,2,1))

plot(x_km[ew_idx], kappa_ew, type = "l",
     xlab = "Easting (km)",
     ylab = expression(hat(kappa)),
     xlim = x_lim,
     main = expression(hat(kappa)~"from north–south transects second differences"))

plot(y_km[ns_idx], kappa_ns, type = "l",
     xlab = "Northing (km)",
     ylab = expression(hat(kappa)),
     xlim = y_lim,
     main = expression(hat(kappa)~"from east–west transects second differences"))

dev.off()

## 7. Figure 4.4 – bubble plot of local roughness --------------------------

# local mean squared 2nd differences using x and y directions
local_ms2 <- matrix(NA_real_, nrow = nrow(z), ncol = ncol(z))
for (i in 2:(nrow(z) - 1)) {
  for (j in 2:(ncol(z) - 1)) {
    zx <- z[i, j + 1] - 2*z[i, j] + z[i, j - 1]
    zy <- z[i + 1, j] - 2*z[i, j] + z[i - 1, j]
    local_ms2[i, j] <- mean(c(zx^2, zy^2))
  }
}

# subsample grid to avoid overplotting
step <- 15
ii <- seq(2, nrow(z) - 1, by = step)
jj <- seq(2, ncol(z) - 1, by = step)

x_grid <- x_km[jj]
y_grid <- y_km[ii]

kk <- local_ms2[ii, jj]
kk_vec <- as.vector(kk)
kk_scaled <- kk_vec / max(kk_vec, na.rm = TRUE)

png("Stats 669/img/fig4_4_bubble_ms2.png", width = 700, height = 700)
plot(x_lim, y_lim, type = "n",
     xlab = "Easting (km)",
     ylab = "Northing (km)",
     main = expression("Local roughness: mean squared 2nd differences ("*m^2*")"))

symbols(rep(x_grid, each = length(y_grid)),
        rep(y_grid, times = length(x_grid)),
        circles = kk_scaled,
        inches = 0.15, add = TRUE, bg = NA)
box()
dev.off()

# 1. Compute slope in degrees (if not already done)
slope <- terrain(dem_2k, v = "slope", unit = "degrees")

# 2. Extract elevation and slope as vectors
elev_vec  <- values(dem_2k, mat = FALSE)
slope_vec <- values(slope,  mat = FALSE)

# Remove NA pairs
ok <- is.finite(elev_vec) & is.finite(slope_vec)
elev_vec  <- elev_vec[ok]
slope_vec <- slope_vec[ok]

# 3. Basic scatterplot (subsample to avoid overplotting)
set.seed(669)
n_plot <- min(20000, length(elev_vec))   # up to 20k points
idx    <- sample(seq_along(elev_vec), n_plot)

png("Stats 669/img/slope_vs_elevation_scatter.png",
    width = 900, height = 600)
plot(elev_vec[idx], slope_vec[idx],
     pch = 16, cex = 0.4, col = rgb(0, 0, 0, 0.3),
     xlab = "Elevation (m)",
     ylab = "Slope (degrees)",
     main = "Slope vs elevation for Death Valley")
# optional LOWESS smooth (not saved to file above)
fit <- lowess(elev_vec[idx], slope_vec[idx], f = 0.2)
dev.off()


png("Stats 669/img/slope_vs_elevation_scatter_lines.png",
    width = 900, height = 600)
plot(elev_vec[idx], slope_vec[idx],
     pch = 16, cex = 0.4, col = rgb(0, 0, 0, 0.3),
     xlab = "Elevation (m)",
     ylab = "Slope (degrees)",
     main = "Slope vs elevation for Death Valley")
# optional LOWESS smooth (not saved to file above)
fit <- lowess(elev_vec[idx], slope_vec[idx], f = 0.2)
lines(fit, col = "red", lwd = 2)

dev.off()
