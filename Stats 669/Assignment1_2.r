library(terra)

# Load the raster data
raster_data <- rast("Stats 669/data/USGS_13_n37w118_20260112.tif")

# plot of raster data
png("Stats 669/img/raster_plot.png")
plot(raster_data)
dev.off()

# contour plot of raster 
png("Stats 669/img/contour_plot.png")
contour(raster_data, asp=1, drawlabels = TRUE, xlab = "Longitude (degrees)", ylab = "Latitude (degrees)", main = "Contour Plot of Elevation Data")
dev.off()

