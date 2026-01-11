# \begin{exercise}[21]
#   Please install the {\tt R} package {\tt datasets}, and use {\tt
#     data(sunspot.year)} to load the W\"olfer sunspot numbers from
#   1700 to 1988. Let $\{X_t\}$ denote the original data, and $\{Y_t\}$
#   denote the mean-corrected series, $Y_t=X_t-49.13$. The following
#   AR(2) model for $\{Y_t\}$ is obtained
#   \begin{equation*}
#     Y_t=1.389Y_{t-1}-.691Y_{t-2}+Z_t,\quad\{Z_t\}\sim\hbox{WN}(0,273.6).
#   \end{equation*}
#   Determine and plot the spectral density of the fitted model and find
#   the frequency at which it achieves its maximum value. What is the
#   corresponding period?
#   \begin{solution}
    
#   \end{solution}
# \end{exercise}


library(datasets)
data(sunspot.year)
X_t <- sunspot.year
Y_t <- X_t - mean(X_t)
ar2_model <- arima(Y_t, order = c(2, 0, 0))
phi1 <- ar2_model$coef[1]
print(phi1)
phi2 <- ar2_model$coef[2]
print(phi2)
sigma2 <- ar2_model$sigma2
print(sigma2)
spectrum_values <- function(omega) {
  num <- sigma2
  denom <- abs(1 - phi1 * exp(-1i * omega) - phi2 * exp(-2i * omega))^2
  return(num / denom)
}
omega_seq <- seq(0, pi, length.out = 1000)
spectrum_vals <- sapply(omega_seq, spectrum_values)
max_index <- which.max(spectrum_vals)
max_frequency <- omega_seq[max_index]
max_period <- 2 * pi / max_frequency
plot(omega_seq, spectrum_vals, type = "l", xlab = "Frequency (rad/year)", ylab = "Spectral Density", main = "Spectral Density of AR(2) Model")
points(max_frequency, spectrum_vals[max_index], col = "red", pch = 19)
text(max_frequency, spectrum_vals[max_index], labels = paste("Max at", round(max_frequency  , 3), "rad/year"), pos = 4)
cat("Maximum frequency:", max_frequency, "rad/year\n")
cat("Corresponding period:", max_period, "years\n")


# Use the W\"olfer sunspot numbers data to Calculate the sample autocovariances up to lag 3. Also plot the sample ACF using the {\tt R} function {\tt acf()}.

acf_values <- acf(Y_t, plot = FALSE, lag.max = 3)
print(acf_values$acf)
plot(acf_values, main = "Sample ACF of Mean-Corrected Sunspot Numbers")





## Load the Wolfer sunspot numbers
library(datasets)
data(sunspot.year)   # yearly sunspot numbers 1700–1988

x <- as.numeric(sunspot.year)  # original series X_t
n <- length(x)

## (a) Sample autocovariances up to lag 3
x_bar <- mean(x)

## function for sample autocovariance with denominator n
sample_acov <- function(x, h) {
  n <- length(x)
  x_bar <- mean(x)
  sum( (x[1:(n-h)] - x_bar) * (x[(1+h):n] - x_bar) ) / n
}

gamma_hat <- sapply(0:30, function(h) sample_acov(x, h))
names(gamma_hat) <- paste0("gamma(", 0:30, ")")
gamma_hat
acf(x, lag.max = 30)  # default type = "correlation", with plot


plot(0:30, gamma_hat[1:31], type="h", xlab="Lag h", ylab="Sample Autocovariance", main="Sample Autocovariances up to Lag 30")


## 1. Load data and mean-correct -----------------------------------------

Y <- sunspots                      # Wolfer sunspot numbers [web:37][web:42]
Y <- Y - mean(Y)                   # mean-corrected series

## 2. Plot sample PACF up to lag 30 -------------------------------------

pacf(Y,
     lag.max = 30,
     main    = "Sample PACF of mean-corrected Wolfer sunspot numbers")

## 3. Extract PACF at lags 1, 2, 3 --------------------------------------

pacf_res <- pacf(Y, lag.max = 30, plot = FALSE)

lags <- as.numeric(pacf_res$lag)   # numeric lag values
phi  <- as.numeric(pacf_res$acf)   # PACF estimates at those lags

pacf_1_3 <- data.frame(lag = lags, pacf = phi)
pacf_1_3 <- subset(pacf_1_3, lag %in% 1:3)

# Nicely formatted table for LaTeX
pacf_1_3_rounded <- transform(pacf_1_3, pacf = round(pacf, 4))
print(pacf_1_3_rounded)


gamma0 <- 1552.81307
gamma1 <- 1264.19939
gamma2 <- 693.89068
gamma3 <- 66.49035

## k = 1
phi11 <- gamma1 / gamma0  # PACF lag 1

## k = 2
Gamma2 <- matrix(c(gamma0, gamma1,
                   gamma1, gamma0), nrow = 2, byrow = TRUE)
gamma_vec2 <- c(gamma1, gamma2)
phi2 <- solve(Gamma2, gamma_vec2)   # (phi21, phi22)
phi22 <- phi2[2]                    # PACF lag 2

## k = 3
Gamma3 <- matrix(c(gamma0, gamma1, gamma2,
                   gamma1, gamma0, gamma1,
                   gamma2, gamma1, gamma0),
                 nrow = 3, byrow = TRUE)
gamma_vec3 <- c(gamma1, gamma2, gamma3)
phi3 <- solve(Gamma3, gamma_vec3)   # (phi31, phi32, phi33)
phi33 <- phi3[3]                    # PACF lag 3

phi3
c(phi11 = phi11, phi22 = phi22, phi33 = phi33)
