# Problem 3.11: Portland cement tensile strength

# Data
mixing <- factor(rep(1:4, each = 4))
strength <- c(3129, 3000, 2865, 2890,
              3200, 3300, 2975, 3150,
              2800, 2900, 2985, 3050,
              2600, 2700, 2600, 2765)

cement <- data.frame(mixing, strength)

# (a) One-way ANOVA
fit311 <- aov(strength ~ mixing, data = cement)
summary(fit311)

# (b) Graphical display of mean strengths with 95% CIs
library(ggplot2)
library(dplyr)

means311 <- cement %>%
  group_by(mixing) %>%
  summarise(mean = mean(strength),
            sd = sd(strength),
            n = n(),
            se = sd / sqrt(n),
            .groups = "drop")

p_means <- ggplot(means311, aes(x = mixing, y = mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean - qt(0.975, n - 1) * se,
                    ymax = mean + qt(0.975, n - 1) * se),
                width = 0.1) +
  xlab("Mixing technique") +
  ylab("Mean tensile strength (lb/in^2)")

ggsave("Stats 490/HW2/img/3_11_mean_CI_plot.png",
       plot = p_means, width = 6, height = 4, dpi = 300)

# (c) Fisher LSD comparisons at alpha = 0.05
anova311 <- summary(fit311)
MSE311   <- anova311[[1]]["Residuals", "Mean Sq"]
df_e311  <- anova311[[1]]["Residuals", "Df"]

pairwise.t.test(cement$strength, cement$mixing,
                p.adjust.method = "none",
                pool.sd = TRUE)

# (d,e) diagnostic plots: save QQ-plot and residuals vs fitted

# QQ-plot of residuals
png("Stats 490/HW2/img/3_11_residuals_qqplot.png",
    width = 1600, height = 1200, res = 200)
qqnorm(residuals(fit311),
       main = "Problem 3.11: Normal Q-Q Plot of Residuals")
qqline(residuals(fit311), col = "red", lwd = 2)
dev.off()

# Residuals vs fitted
png("Stats 490/HW2/img/3_11_residuals_vs_fitted.png",
    width = 1600, height = 1200, res = 200)
plot(fitted(fit311), residuals(fit311),
     xlab = "Fitted tensile strength",
     ylab = "Residuals",
     main = "Problem 3.11: Residuals vs Fitted")
abline(h = 0, col = "red", lwd = 2)
dev.off()

# (f) scatter plot of raw data
p_scatter <- ggplot(cement, aes(x = mixing, y = strength)) +
  geom_point(position = position_jitter(width = 0.05), size = 2) +
  xlab("Mixing technique") +
  ylab("Tensile strength (lb/in^2)") +
  ggtitle("Problem 3.11: Raw tensile strength data")

ggsave("Stats 490/HW2/img/3_11_scatter_rawdata.png",
       plot = p_scatter, width = 6, height = 4, dpi = 300)


# Problem 3.12: follow‑up to 3.11

# Reuse data from 3.11
mixing <- factor(rep(1:4, each = 4))
strength <- c(3129, 3000, 2865, 2890,
              3200, 3300, 2975, 3150,
              2800, 2900, 2985, 3050,
              2600, 2700, 2600, 2765)

cement <- data.frame(mixing, strength)
fit311 <- aov(strength ~ mixing, data = cement)

# Tukey HSD at alpha = 0.05
tuk311 <- TukeyHSD(fit311, "mixing", conf.level = 0.95)
tuk311

# Save Tukey plot
png("Stats 490/HW2/img/3_12_TukeyHSD_plot.png",
    width = 1600, height = 1200, res = 200)
plot(tuk311, las = 1)
dev.off()


## Problem 3.14 and 3.15: Cotton content and tensile strength

# Data from Problem 3.14
cotton <- factor(rep(c(15, 20, 25, 30, 35), each = 5))
strength <- c( 7,  7, 15, 11,  9,   # 15%
              12, 17, 12, 18, 18,   # 20%
              14, 19, 19, 18, 18,   # 25%
              19, 25, 22, 19, 23,   # 30%
               7, 10, 11, 15, 11)   # 35%

dat314 <- data.frame(cotton, strength)

## Problem 3.14: One-way ANOVA and diagnostics
# One-way ANOVA
fit314 <- aov(strength ~ cotton, data = dat314)
summary(fit314)

# Mean + 95% CI plot by cotton level
library(ggplot2)
library(dplyr)

means314 <- dat314 %>%
  group_by(cotton) %>%
  summarise(mean = mean(strength),
            sd   = sd(strength),
            n    = n(),
            se   = sd / sqrt(n),
            .groups = "drop")

p_means_314 <- ggplot(means314, aes(x = cotton, y = mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean - qt(0.975, n - 1) * se,
                    ymax = mean + qt(0.975, n - 1) * se),
                width = 0.1) +
  xlab("Cotton weight percent") +
  ylab("Mean tensile strength") +
  ggtitle("Problem 3.14/3.15: Mean tensile strength by cotton content")

ggsave("Stats 490/HW2/img/3_15_mean_CI_plot.png",
       plot = p_means_314, width = 6, height = 4, dpi = 300)

# Normal Q-Q plot of residuals
png("Stats 490/HW2/img/3_14_residuals_qqplot.png",
    width = 1600, height = 1200, res = 200)
qqnorm(residuals(fit314),
       main = "Problem 3.14: Normal Q-Q Plot of Residuals")
qqline(residuals(fit314), col = "red", lwd = 2)
dev.off()

# Residuals vs fitted
png("Stats 490/HW2/img/3_14_residuals_vs_fitted.png",
    width = 1600, height = 1200, res = 200)
plot(fitted(fit314), residuals(fit314),
     xlab = "Fitted tensile strength",
     ylab = "Residuals",
     main = "Problem 3.14: Residuals vs Fitted")
abline(h = 0, col = "red", lwd = 2)
dev.off()

## Problem 3.15: Dunnett-style comparison vs 30% control

# Extract ANOVA quantities
anova314 <- summary(fit314)
MSE314   <- anova314[[1]]["Residuals", "Mean Sq"]
df_e314  <- anova314[[1]]["Residuals", "Df"]

MSE314
df_e314

# Treatment means and sample sizes
t_means <- tapply(dat314$strength, dat314$cotton, mean)
t_n     <- tapply(dat314$strength, dat314$cotton, length)

t_means
t_n

# Control and treatment levels
control_level <- "30"
other_levels  <- setdiff(names(t_means), control_level)

mu_c <- t_means[control_level]
n_c  <- t_n[control_level]

# Differences and standard errors for (treatment - control)
results <- data.frame(
  level = other_levels,
  diff  = NA_real_,
  se    = NA_real_,
  tstat = NA_real_
)

for (i in seq_along(other_levels)) {
  lev   <- other_levels[i]
  mu_i  <- t_means[lev]
  n_i   <- t_n[lev]
  diff_ <- mu_i - mu_c
  se_   <- sqrt(MSE314 * (1 / n_i + 1 / n_c))
  t_    <- diff_ / se_
  results$diff[i]  <- diff_
  results$se[i]    <- se_
  results$tstat[i] <- t_
}

results

# Approximate Dunnett critical value via simulation
# 5 treatments total, 1 control, 4 comparisons, df = df_e314

set.seed(123)
B <- 200000          # simulations; reduce if too slow
a <- length(t_means)
m <- a - 1           # number of comparisons vs control

max_t <- numeric(B)
for (b in 1:B) {
  ts <- rt(m, df = df_e314)   # independent t's with df_e314
  max_t[b] <- max(abs(ts))
}

crit_D <- quantile(max_t, 0.95)  # 95% Dunnett-like critical value
crit_D

# Dunnett-style simultaneous 95% CIs and significance flags
results$lower <- results$diff - crit_D * results$se
results$upper <- results$diff + crit_D * results$se
results$signif_0.05 <- ifelse(results$lower > 0 | results$upper < 0,
                              "Yes", "No")

results



# Problem 3.22: coating type and conductivity

# Data from the problem
coating <- factor(rep(1:4, each = 4))
conductivity <- c(143, 141, 150, 146,
                  152, 149, 137, 143,
                  134, 136, 132, 127,
                  129, 127, 132, 129)

dat322 <- data.frame(coating, conductivity)

# (a) One-way ANOVA at alpha = 0.05
fit322 <- aov(conductivity ~ coating, data = dat322)
summary(fit322)

# (b) overall mean and treatment effects
overall_mean_322 <- mean(dat322$conductivity)
treatment_means_322 <- tapply(dat322$conductivity, dat322$coating, mean)
treatment_effects_322 <- treatment_means_322 - overall_mean_322

overall_mean_322
treatment_means_322
treatment_effects_322

# (c) CIs for mean of coating 4 and difference (1 - 4)

# extract MSE and df
anova322 <- summary(fit322)
MSE322   <- anova322[[1]]["Residuals", "Mean Sq"]
df_e322  <- anova322[[1]]["Residuals", "Df"]

MSE322
df_e322

# sample sizes
n_per <- table(dat322$coating)
n4 <- n_per["4"]
n1 <- n_per["1"]

# 95% CI for mean of coating 4
mean4 <- treatment_means_322["4"]
se_mean4 <- sqrt(MSE322 / n4)
t_95 <- qt(0.975, df_e322)
CI_mean4 <- mean4 + c(-1, 1) * t_95 * se_mean4
CI_mean4

# 99% CI for mean difference (1 - 4)
mean1 <- treatment_means_322["1"]
se_diff_1_4 <- sqrt(MSE322 * (1/n1 + 1/n4))
t_99 <- qt(0.995, df_e322)
diff_1_4 <- mean1 - mean4
CI_diff_1_4 <- diff_1_4 + c(-1, 1) * t_99 * se_diff_1_4
CI_diff_1_4

# (d) Fisher LSD at alpha = 0.05 using pooled MSE
pairwise.t.test(dat322$conductivity, dat322$coating,
                p.adjust.method = "none",
                pool.sd = TRUE)

# (e) Graphical mean comparison (mean + 95% CI)
library(ggplot2)
library(dplyr)

means322 <- dat322 %>%
  group_by(coating) %>%
  summarise(mean = mean(conductivity),
            sd = sd(conductivity),
            n = n(),
            se = sd / sqrt(n),
            .groups = "drop")

p_means_322 <- ggplot(means322, aes(x = coating, y = mean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = mean - qt(0.975, n - 1) * se,
                    ymax = mean + qt(0.975, n - 1) * se),
                width = 0.1) +
  xlab("Coating type") +
  ylab("Mean conductivity") +
  ggtitle("Problem 3.22: Mean conductivity by coating type")

ggsave("Stats 490/HW2/img/3_22_mean_CI_plot.png",
       plot = p_means_322, width = 6, height = 4, dpi = 300)

# (f) No extra code needed beyond means; interpretation is conceptual.
# But we can also save raw-data scatterplot for later use.

p_scatter_322 <- ggplot(dat322, aes(x = coating, y = conductivity)) +
  geom_point(position = position_jitter(width = 0.05), size = 2) +
  xlab("Coating type") +
  ylab("Conductivity") +
  ggtitle("Problem 3.22: Conductivity data")

ggsave("Stats 490/HW2/img/3_22_scatter_rawdata.png",
       plot = p_scatter_322, width = 6, height = 4, dpi = 300)

# Diagnostics for later (used in 3.23 as well)

# QQ-plot
png("Stats 490/HW2/img/3_22_residuals_qqplot.png",
    width = 1600, height = 1200, res = 200)
qqnorm(residuals(fit322),
       main = "Problem 3.22: Normal Q-Q Plot of Residuals")
qqline(residuals(fit322), col = "red", lwd = 2)
dev.off()

# Residuals vs fitted
png("Stats 490/HW2/img/3_22_residuals_vs_fitted.png",
    width = 1600, height = 1200, res = 200)
plot(fitted(fit322), residuals(fit322),
     xlab = "Fitted conductivity",
     ylab = "Residuals",
     main = "Problem 3.22: Residuals vs Fitted")
abline(h = 0, col = "red", lwd = 2)
dev.off()

# Problem 3.26: circuit response time

# Data
circuit <- factor(rep(1:3, each = 5))
time <- c(9, 12, 10, 8, 15,
          20, 21, 23, 17, 30,
          6, 5, 8, 16, 7)

dat326 <- data.frame(circuit, time)

# (a) One-way ANOVA at alpha = 0.01
fit326 <- aov(time ~ circuit, data = dat326)
summary(fit326)

# (b) Tukey pairwise comparisons with familywise alpha = 0.01
tuk326 <- TukeyHSD(fit326, "circuit", conf.level = 0.99)
tuk326

# Save Tukey plot
png("Stats 490/HW2/img/3_26_TukeyHSD_plot.png",
    width = 1600, height = 1200, res = 200)
plot(tuk326, las = 1)
dev.off()

# Diagnostic plots: QQ-plot and residuals vs fitted

# QQ-plot
png("Stats 490/HW2/img/3_26_residuals_qqplot.png",
    width = 1600, height = 1200, res = 200)
qqnorm(residuals(fit326),
       main = "Problem 3.26: Normal Q-Q Plot of Residuals")
qqline(residuals(fit326), col = "red", lwd = 2)
dev.off()

# Residuals vs fitted
png("Stats 490/HW2/img/3_26_residuals_vs_fitted.png",
    width = 1600, height = 1200, res = 200)
plot(fitted(fit326), residuals(fit326),
     xlab = "Fitted response time",
     ylab = "Residuals",
     main = "Problem 3.26: Residuals vs Fitted")
abline(h = 0, col = "red", lwd = 2)
dev.off()

# Raw data scatterplot
library(ggplot2)

p_scatter_326 <- ggplot(dat326, aes(x = circuit, y = time)) +
  geom_point(position = position_jitter(width = 0.05), size = 2) +
  xlab("Circuit type") +
  ylab("Response time (ms)") +
  ggtitle("Problem 3.26: Raw response time data")

ggsave("Stats 490/HW2/img/3_26_scatter_rawdata.png",
       plot = p_scatter_326, width = 6, height = 4, dpi = 300)



# Problem 3.36: wafer position and uniformity

# Data
wafer <- factor(rep(1:4, each = 3))
uniformity <- c(2.76, 5.67, 4.49,
                1.43, 1.70, 2.19,
                2.34, 1.97, 1.47,
                0.94, 1.36, 1.65)

dat336 <- data.frame(wafer, uniformity)

# (a) One-way ANOVA at alpha = 0.05
fit336 <- aov(uniformity ~ wafer, data = dat336)
summary(fit336)

# (b) and (c): variance components
anova336 <- summary(fit336)
MS_wafer <- anova336[[1]]["wafer", "Mean Sq"]
MS_error <- anova336[[1]]["Residuals", "Mean Sq"]

MS_wafer
MS_error

# number of replicates per wafer
r <- 3

# variance component due to wafer positions (random effect)
sigma_wafer2_hat <- (MS_wafer - MS_error) / r
sigma_wafer2_hat

# random error variance component
sigma_error2_hat <- MS_error
sigma_error2_hat

# Diagnostic plots: QQ-plot and residuals vs fitted

# QQ-plot
png("Stats 490/HW2/img/3_36_residuals_qqplot.png",
    width = 1600, height = 1200, res = 200)
qqnorm(residuals(fit336),
       main = "Problem 3.36: Normal Q-Q Plot of Residuals")
qqline(residuals(fit336), col = "red", lwd = 2)
dev.off()

# Residuals vs fitted
png("Stats 490/HW2/img/3_36_residuals_vs_fitted.png",
    width = 1600, height = 1200, res = 200)
plot(fitted(fit336), residuals(fit336),
     xlab = "Fitted uniformity",
     ylab = "Residuals",
     main = "Problem 3.36: Residuals vs Fitted")
abline(h = 0, col = "red", lwd = 2)
dev.off()

# Raw data plot: uniformity by wafer position
library(ggplot2)

p_scatter_336 <- ggplot(dat336, aes(x = wafer, y = uniformity)) +
  geom_point(position = position_jitter(width = 0.05), size = 2) +
  xlab("Wafer position") +
  ylab("Uniformity") +
  ggtitle("Problem 3.36: Film thickness uniformity data")

ggsave("Stats 490/HW2/img/3_36_scatter_rawdata.png",
       plot = p_scatter_336, width = 6, height = 4, dpi = 300)


# Problem 3.52: sample size for one-way ANOVA with 4 groups

mu <- c(50, 60, 50, 60)   # population means
sigma2 <- 25              # error variance
alpha <- 0.05
target_power <- 0.90
a <- length(mu)

compute_power <- function(n) {
  grand_mean <- mean(mu)
  lambda <- n * sum((mu - grand_mean)^2) / sigma2
  df1 <- a - 1
  df2 <- a * (n - 1)
  F_crit <- qf(1 - alpha, df1, df2)
  1 - pf(F_crit, df1, df2, ncp = lambda)
}

n_values <- 2:50
powers <- sapply(n_values, compute_power)

power_table <- cbind(n = n_values, power = round(powers, 4))
power_table

min_n_90 <- n_values[which(powers >= target_power)[1]]
min_n_90

