library(terra)
library(tidyverse)




# 1. Read the original DEM
tif_in  <- "Stats 669/data/USGS_13_n37w118_20260112.tif"
dem_big <- rast(tif_in)

# Check original size
nr <- nrow(dem_big)
nc <- ncol(dem_big)
cat("Original size:", nr, "x", nc, "\n")

# 2. Compute row/col indices for a 2000 x 2000 central window
target_size <- 2000

if (nr < target_size || nc < target_size) {
  stop("Raster is smaller than 2000 x 2000; pick a smaller target_size.")
}

row_start <- floor((nr - target_size) / 2) + 1
row_end   <- row_start + target_size - 1
col_start <- floor((nc - target_size) / 2) + 1
col_end   <- col_start + target_size - 1

cat("Row indices:", row_start, "to", row_end, "\n")
cat("Col indices:", col_start, "to", col_end, "\n")

# 3. Turn row/col into an extent and crop the SpatRaster
#    (this keeps it as a SpatRaster, so writeRaster works)
e <- ext(dem_big, r1 = row_start, r2 = row_end,
                  c1 = col_start, c2 = col_end)

dem_small <- crop(dem_big, e)

cat("Cropped size:", nrow(dem_small), "x", ncol(dem_small), "\n")

# 4. Write out the smaller GeoTIFF
tif_out <- "Stats 669/data/USGS_13_n37w118_small_2000.tif"

writeRaster(
  dem_small,
  filename  = tif_out,
  overwrite = TRUE,
  filetype  = "GTiff"
)

cat("Wrote:", tif_out, "\n")


pic <- rast("Stats 669/data/USGS_13_n37w118_20260112.tif") 

x0 <- -75.146783
y0 <- 41.269037
n <- 2000

res_x <- res(pic)[1]
res_y <- res(pic)[2]

e <- ext(x0 - n * res_x, x0,
         y0, y0 + n * res_y)

pic_cropped <- crop(pic, e, snap = 'near')

pic_projected <- project(pic_cropped, 'EPSG:26918')

## Start here if you have your picture already

df <- as.data.frame(pic_projected, xy = T, na.rm = T)
df$x <- (df$x - min(df$x)) / 1000
df$y <- (df$y - min(df$y)) / 1000
names(df) <- c('east', 'north', 'x')

gamma_hat <- function(x, lag = 1, m = 1){
  d <- diff(x, lag = lag, differences = m)
  mean(d^2, na.rm = T) / 2
}

kappa_hat <- function(x, m = 1){
  g1 <- gamma_hat(x, 1, m = m)
  g2 <- gamma_hat(x, 2, m = m)
  log(g2 / g1, base = 2) / 2
}

### Plot 1

## East and West Kappas

EW_kappas <- df %>% 
  arrange(east) %>%
  pivot_wider(names_from = east, values_from = x) %>%
  arrange(north) %>%
  select(-north) %>%
  summarise(across(everything(), ~ kappa_hat(.x, m = 2))) %>%
  pivot_longer(cols = everything(), names_to = 'east', values_to = 'kappa') %>%
  mutate(east = as.numeric(east))

ggplot(EW_kappas, aes(x = east, y = kappa)) +
  geom_point() +
  labs(
    title = 'Estimated Principal Irregular Term for North-South Transects in the Ponoco Mountains, PA',
    x = 'Easting (km)',
    y = expression(kappa)
  ) +
  theme_bw() +
  theme(panel.border = element_rect(color = 'black'),
        axis.text.x = element_text(angle = 45, hjust = 1))

## North and South Kappas

NS_kappas <- df %>% 
  arrange(north) %>%
  pivot_wider(names_from = north, values_from = x) %>%
  arrange(east) %>%
  select(-east) %>%
  summarise(across(everything(), ~ kappa_hat(.x, m = 2))) %>%
  pivot_longer(cols = everything(), names_to = 'north', values_to = 'kappa') %>%
  mutate(north = as.numeric(north))

ggplot(NS_kappas, aes(x = north, y = kappa)) +
  geom_point() +
  labs(
    title = 'Estimated Principal Irregular Term for East-West Transects in the Ponoco Mountains, PA',
    x = 'Northing (km)',
    y = expression(kappa)
  ) +
  theme_bw() +
  theme(panel.border = element_rect(color = 'black'),
        axis.text.x = element_text(angle = 45, hjust = 1))

### Plot 2

## East and West Average Squared Differences

EW_sqd <- df %>%
  arrange(east) %>%
  pivot_wider(names_from = east, values_from = x) %>%
  arrange(north) %>%
  select(-north) %>%
  summarise(across(everything(), ~ 2 * gamma_hat(.x, m = 2))) %>%
  pivot_longer(cols = everything(), names_to = 'east', values_to = 'sqd') %>%
  mutate(east = as.numeric(east))

ggplot(EW_sqd, aes(x = east, y = sqd)) +
  geom_point() +
  labs(
    title = 'Average Squared Differences for North-South Transects in the Ponoco Mountains, PA',
    x = 'Easting (km)',
    y = expression(kappa)
  ) +
  theme_bw() +
  theme(panel.border = element_rect(color = 'black'),
        axis.text.x = element_text(angle = 45, hjust = 1))


## North and South Average Squared Differences

NS_sqd <- df %>%
  arrange(north) %>%
  pivot_wider(names_from = north, values_from = x) %>%
  arrange(east) %>%
  select(-east) %>%
  summarise(across(everything(), ~ 2 * gamma_hat(.x, m = 2))) %>%
  pivot_longer(cols = everything(), names_to = 'north', values_to = 'sqd') %>%
  mutate(north = as.numeric(north))

ggplot(NS_sqd, aes(x = north, y = sqd)) +
  geom_point() +
  labs(
    title = 'Average Squared Differences for East-West Transects in the Ponoco Mountains, PA',
    x = 'Northing (km)',
    y = expression(kappa)
  ) +
  theme_bw() +
  theme(panel.border = element_rect(color = 'black'),
        axis.text.x = element_text(angle = 45, hjust = 1))

### Plot 3

Z <- df %>%
  arrange(east) %>%
  pivot_wider(names_from = east, values_from = x) %>%
  arrange(north) %>%
  select(-north) %>%
  t()

Z <- Z[nrow(Z):1,]

colnames(Z) <- NULL
rownames(Z) <- NULL

delta <- function(x, s, t, j){sum(c(-4*x[s,t], x[s+j,t], x[s-j,t], x[s,t+j], x[s,t-j]), na.rm = T)}

eta <- function(x, s, t, j, n){
  grid_range <- (1+j):(n-j)
  grid <- expand.grid(grid_range, grid_range)
  diffs <- mapply(function(l,k) delta(x, s+l, t+k, j)^2, grid$Var1, grid$Var2)
  mean(diffs)
}

kappa_hat_2d <- function(x, s, t, n){
  e1 <- eta(x, s, t, 1, n)
  e2 <- eta(x, s, t, 2, n)
  log(e2 / e1, base = 2) / 2
}

row_idx <- seq(1, nrow(Z), 50)
row_idx <- row_idx[row_idx <= nrow(Z) - 50]
col_idx <- seq(1, ncol(Z), 50)
col_idx <- col_idx[col_idx <= ncol(Z) - 50]

grid <- expand.grid(row_idx, col_idx)
names(grid) <- c('row_idx', 'col_idx')
grid$kappa <- NA

for(i in 1:nrow(grid)){
  s <- grid[i, 1]
  t <- grid[i, 2]
  grid[i,3] <- kappa_hat_2d(Z, s, t, 50)
}

ggplot(grid, aes(x = abs(row_idx - max(row_idx)), y = col_idx, size = abs(kappa - 1))) +
  geom_point() +
  scale_size_continuous(range = c(0.5, 5)) +
  labs(
    title = 'Estimated Kappa Per 50 x 50 Pixel Grid in the Ponoco Mountains, PA',
    x = 'Easting (km)',
    y = 'Northing (km)'
  ) +
  theme_bw() +
  theme(panel.border = element_rect(color = 'black'),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = 'none')




