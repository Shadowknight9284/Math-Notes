means  <- c(Public = 53.4, Private = 70.8, Church = 56.1)
vars   <- c(Public = 72.8, Private = 48.2, Church = 60.3)
n      <- 5

# (b)
grand_mean <- mean(means)   # 60.1
grand_mean
SS_trt <- n * sum((means - grand_mean)^2)  # 876.9
SS_trt
SS_E <- sum((n-1)*vars)     # 725.2
SS_E
df_trt <- 2
df_E   <- 12
MS_trt <- SS_trt / df_trt
MS_E   <- SS_E   / df_E
F_stat <- MS_trt / MS_E    # ~7.259
p_val  <- 1 - pf(F_stat, df_trt, df_E)
p_val

# (c)
sigma_hat <- sqrt(MS_E)  # ~7.77
sigma_hat

# (d)
SE_diff <- sqrt(2 * MS_E / n)   # ~4.916
SE_diff
diffs <- c(Priv_Pub = means["Private"] - means["Public"],
           Ch_Pub   = means["Church"]  - means["Public"],
           Ch_Priv  = means["Church"]  - means["Private"])
diffs
qtukey(0.95, nmeans = 3, df = 12)  # ~3.77
qcrit <- qtukey(0.95, 3, 12)
HSD   <- qcrit * sqrt(MS_E / n)   # ~13.1
Fcrit  <- qf(0.95, df1 = 2, df2 = 12)   # ~3.89
k_S    <- (3 - 1) * Fcrit               # ~7.78
T2 <- (diffs^2) / (SE_diff^2)
T2  # check which exceed k_S


# (e)


# (f)



# Question 2
## Given summary information
means_P2 <- c(G1 = 72.3, G2 = 79.5, G3 = 65.8)
n_P2     <- 6              # per group
a_P2     <- 3
N_P2     <- a_P2 * n_P2    # 18

overall_var <- 84.03       # variance of all 18 scores
MS_E_P2     <- 57.84       # given error variance estimate

## (a) ANOVA table quantities
grand_mean_P2 <- mean(means_P2)
grand_mean_P2

# Total SS = overall variance * (N - 1)
SS_T_P2 <- overall_var * (N_P2 - 1)
SS_T_P2

# Treatment SS = n * sum( (group mean - grand mean)^2 )
SS_trt_P2 <- n_P2 * sum( (means_P2 - grand_mean_P2)^2 )
SS_trt_P2

# Error SS = Total SS - Treatment SS (should be close to MS_E * (N - a))
SS_E_P2_alt <- SS_T_P2 - SS_trt_P2
SS_E_P2_alt

# Or directly from MS_E: SSE = MS_E * df_E
df_trt_P2 <- a_P2 - 1
df_E_P2   <- N_P2 - a_P2
SSE_from_MS <- MS_E_P2 * df_E_P2
SSE_from_MS

# Use the given MS_E and its implied SSE for the ANOVA table
SS_E_P2 <- SSE_from_MS
SS_E_P2

# Recompute SST using that SSE (for a fully self-consistent ANOVA)
SS_T_P2_consistent <- SS_trt_P2 + SS_E_P2
SS_T_P2_consistent

# Mean squares and F
MS_trt_P2 <- SS_trt_P2 / df_trt_P2
MS_trt_P2

F_stat_P2 <- MS_trt_P2 / MS_E_P2
F_stat_P2

p_val_P2 <- 1 - pf(F_stat_P2, df_trt_P2, df_E_P2)
p_val_P2

## (c) & (d): contrasts vs control (group 2)

# Standard error for difference of a single treatment mean vs control (same n)
SE_diff_P2 <- sqrt(2 * MS_E_P2 / n_P2)
SE_diff_P2

# Individual differences relative to control (group 2)
diffs_vs_control <- c(
  G1_minus_G2 = means_P2["G1"] - means_P2["G2"],
  G3_minus_G2 = means_P2["G3"] - means_P2["G2"]
)
diffs_vs_control

t_stats_vs_control <- diffs_vs_control / SE_diff_P2
t_stats_vs_control

# (d) Average treatment (groups 1 and 3) vs control (group 2)
# Contrast: C = 0.5 * mu1 + 0.5 * mu3 - mu2
C_hat <- 0.5 * means_P2["G1"] + 0.5 * means_P2["G3"] - means_P2["G2"]
C_hat

# SE of that contrast:
# var(C_hat) = MS_E * (0.5^2/n + 0.5^2/n + 1^2/n) = MS_E * (0.25 + 0.25 + 1)/n = MS_E * 1.5 / n
SE_C <- sqrt(MS_E_P2 * 1.5 / n_P2)
SE_C

t_C <- C_hat / SE_C
t_C

# Bonferroni adjustment: total K = 10 post-hoc tests, alpha = 0.05
alpha <- 0.05
K     <- 10
alpha_star <- alpha / K
alpha_star

# two-sided test, so use alpha_star/2
t_crit_Bonf <- qt(1 - alpha_star/2, df = df_E_P2)
t_crit_Bonf






# Question 3
## Load data
dat <- read.csv("Stats 490/Exam1/Mid1Prob3.csv")
str(dat)
head(dat)

# Make sure Treatment is a factor
dat$Treatment <- factor(dat$Treatment)

# Quick summary by treatment
library(dplyr)
group_stats <- dat %>%
  group_by(Treatment) %>%
  summarise(
    n   = n(),
    mean = mean(Gene.Expression),
    sd   = sd(Gene.Expression)
  )
group_stats

########################################################
## (a) ANOVA on raw data: do treatment means differ?
########################################################

fit_raw <- aov(Gene.Expression ~ Treatment, data = dat)
summary(fit_raw)

# Extract F and p-value explicitly if you like
anova_raw <- summary(fit_raw)[[1]]
anova_raw

########################################################
## (b) Normality check for raw residuals
########################################################

res_raw <- residuals(fit_raw)

# QQ-plot and histogram
par(mfrow = c(1, 2))
qqnorm(res_raw, main = "QQ Plot - Raw Data Residuals")
qqline(res_raw)
hist(res_raw, main = "Histogram - Raw Data Residuals",
     xlab = "Residuals")

# Shapiro-Wilk normality test
shapiro_raw <- shapiro.test(res_raw)
shapiro_raw

# Reset plotting
par(mfrow = c(1, 1))

########################################################
## (c) Log transform and ANOVA on log data
########################################################

# Create log-transformed response
dat$logGE <- log(dat$Gene.Expression)

# Check group stats on log scale
group_stats_log <- dat %>%
  group_by(Treatment) %>%
  summarise(
    n    = n(),
    mean = mean(logGE),
    sd   = sd(logGE)
  )
group_stats_log

# ANOVA on log-transformed data
fit_log <- aov(logGE ~ Treatment, data = dat)
summary(fit_log)

anova_log <- summary(fit_log)[[1]]
anova_log

########################################################
## (d) Residual diagnostics for transformed model
########################################################

res_log <- residuals(fit_log)
fitted_log <- fitted(fit_log)

# Residual vs fitted and QQ plot
par(mfrow = c(1, 2))
plot(fitted_log, res_log,
     main = "Residuals vs Fitted (log model)",
     xlab = "Fitted values", ylab = "Residuals")
abline(h = 0, col = "red")

qqnorm(res_log, main = "QQ Plot - log model residuals")
qqline(res_log, col = "red")

par(mfrow = c(1, 1))

# Shapiro-Wilk on log residuals
shapiro_log <- shapiro.test(res_log)
shapiro_log




#

























########################
# Problem 1
########################

# 1(a–d): One-way ANOVA from summary stats
means  <- c(Public = 53.4, Private = 70.8, Church = 56.1)
vars   <- c(Public = 72.8, Private = 48.2, Church = 60.3)
n      <- 5

grand_mean <- mean(means)
grand_mean

SS_trt <- n * sum((means - grand_mean)^2)
SS_trt

SS_E <- sum((n - 1) * vars)
SS_E

df_trt <- 2
df_E   <- 12

MS_trt <- SS_trt / df_trt
MS_E   <- SS_E   / df_E

F_stat <- MS_trt / MS_E
F_stat

p_val  <- 1 - pf(F_stat, df_trt, df_E)
p_val

# 1(c): error SD
sigma_hat <- sqrt(MS_E)
sigma_hat

# 1(d): pairwise differences, Tukey HSD and Scheffé
SE_diff <- sqrt(2 * MS_E / n)
SE_diff

diffs <- c(
  Priv_Pub = means["Private"] - means["Public"],
  Ch_Pub   = means["Church"]  - means["Public"],
  Ch_Priv  = means["Church"]  - means["Private"]
)
diffs

qcrit <- qtukey(0.95, nmeans = 3, df = df_E)
qcrit

HSD   <- qcrit * sqrt(MS_E / n)
HSD

Fcrit <- qf(0.95, df1 = df_trt, df2 = df_E)
Fcrit

k_S   <- (3 - 1) * Fcrit
k_S

T2 <- (diffs^2) / (SE_diff^2)
T2


########################
# Problem 2
########################

# 2(a): One-way ANOVA from summary stats
means_P2 <- c(G1 = 72.3, G2 = 79.5, G3 = 65.8)
n_P2     <- 6
a_P2     <- 3
N_P2     <- a_P2 * n_P2

overall_var <- 84.03
MS_E_P2     <- 57.84

grand_mean_P2 <- mean(means_P2)
grand_mean_P2

SS_T_P2 <- overall_var * (N_P2 - 1)
SS_T_P2

SS_trt_P2 <- n_P2 * sum((means_P2 - grand_mean_P2)^2)
SS_trt_P2

SS_E_P2_alt <- SS_T_P2 - SS_trt_P2
SS_E_P2_alt

df_trt_P2 <- a_P2 - 1
df_E_P2   <- N_P2 - a_P2

SSE_from_MS <- MS_E_P2 * df_E_P2
SSE_from_MS

SS_E_P2 <- SSE_from_MS
SS_E_P2

SS_T_P2_consistent <- SS_trt_P2 + SS_E_P2
SS_T_P2_consistent

MS_trt_P2 <- SS_trt_P2 / df_trt_P2
MS_trt_P2

F_stat_P2 <- MS_trt_P2 / MS_E_P2
F_stat_P2

p_val_P2 <- 1 - pf(F_stat_P2, df_trt_P2, df_E_P2)
p_val_P2

# 2(c): pairwise differences vs control (group 2)
SE_diff_P2 <- sqrt(2 * MS_E_P2 / n_P2)
SE_diff_P2

diffs_vs_control <- c(
  G1_minus_G2 = means_P2["G1"] - means_P2["G2"],
  G3_minus_G2 = means_P2["G3"] - means_P2["G2"]
)
diffs_vs_control

t_stats_vs_control <- diffs_vs_control / SE_diff_P2
t_stats_vs_control

# 2(d): contrast = average of groups 1 and 3 vs control
C_hat <- 0.5 * means_P2["G1"] + 0.5 * means_P2["G3"] - means_P2["G2"]
C_hat

SE_C <- sqrt(MS_E_P2 * 1.5 / n_P2)
SE_C

t_C <- C_hat / SE_C
t_C

alpha      <- 0.05
K          <- 10
alpha_star <- alpha / K
alpha_star

t_crit_Bonf <- qt(1 - alpha_star / 2, df = df_E_P2)
t_crit_Bonf


########################
# Problem 3
########################

# 3(a): One-way ANOVA on raw gene expression
dat <- read.csv("Stats 490/Exam1/Mid1Prob3.csv")

str(dat)
head(dat)

dat$Treatment <- factor(dat$Treatment)

library(dplyr)

group_stats <- dat %>%
  group_by(Treatment) %>%
  summarise(
    n    = n(),
    mean = mean(Gene.Expression),
    sd   = sd(Gene.Expression)
  )
group_stats

fit_raw <- aov(Gene.Expression ~ Treatment, data = dat)
summary(fit_raw)

anova_raw <- summary(fit_raw)[[1]]
anova_raw

# 3(b): normality check for raw residuals
res_raw <- residuals(fit_raw)

par(mfrow = c(1, 2))
qqnorm(res_raw, main = "QQ Plot - Raw Data Residuals")
qqline(res_raw)
hist(res_raw, main = "Histogram - Raw Data Residuals",
     xlab = "Residuals")
par(mfrow = c(1, 1))

shapiro_raw <- shapiro.test(res_raw)
shapiro_raw

# 3(c): log transform and ANOVA
dat$logGE <- log(dat$Gene.Expression)

group_stats_log <- dat %>%
  group_by(Treatment) %>%
  summarise(
    n    = n(),
    mean = mean(logGE),
    sd   = sd(logGE)
  )
group_stats_log

fit_log <- aov(logGE ~ Treatment, data = dat)
summary(fit_log)

anova_log <- summary(fit_log)[[1]]
anova_log

# 3(d): residual diagnostics for log model
res_log    <- residuals(fit_log)
fitted_log <- fitted(fit_log)

par(mfrow = c(1, 2))
plot(fitted_log, res_log,
     main = "Residuals vs Fitted (log model)",
     xlab = "Fitted values", ylab = "Residuals")
abline(h = 0, col = "red")

qqnorm(res_log, main = "QQ Plot - log model residuals")
qqline(res_log, col = "red")
par(mfrow = c(1, 1))

shapiro_log <- shapiro.test(res_log)
shapiro_log




