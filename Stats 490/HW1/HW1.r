## 2.21: sample size for CI length

sigma2 <- 9          # given variance
sigma  <- sqrt(sigma2)
L      <- 1.0        # desired TOTAL length of CI
conf   <- 0.95

alpha  <- 1 - conf
z      <- qnorm(1 - alpha/2)        # z_{0.975} for 95% CI

# Formula: L = 2 * z * sigma / sqrt(n)  =>  n = (2*z*sigma / L)^2
n_exact <- (2 * z * sigma / L)^2
n_exact

# Required integer sample size (round up)
n_required <- ceiling(n_exact)
n_required


# 2.22
## Data: shelf life in days
x <- c(108, 124, 124, 106, 115, 138, 163, 159, 134, 139)

## Basic summaries
n  <- length(x)
xbar <- mean(x)
s  <- sd(x)
n; xbar; s

## (b) One-sample t test: H0: mu = 120 vs Ha: mu > 120, alpha = 0.01
mu0 <- 120
# Test statistic (by hand)
t_stat <- (xbar - mu0) / (s / sqrt(n))
df <- n - 1
t_stat
df
# Critical value for alpha = 0.01, right-tailed
t_crit <- qt(0.99, df = df)   # t_{0.99,9}
t_crit

## (c) P-value for right-tailed test
p_value <- 1 - pt(t_stat, df = df)  # P(T >= t_stat)
p_value

## (d) 99% confidence interval for the mean (by hand)
alpha <- 0.01
t_crit_2sided <- qt(1 - alpha/2, df = df)  # t_{0.995,9}
SE <- s / sqrt(n)
ME <- t_crit_2sided * SE
lower_99 <- xbar - ME
upper_99 <- xbar + ME
c(lower_99, upper_99)

## 2.23  Normality check for shelf life data (from 2.22)

x <- c(108, 124, 124, 106, 115, 138, 163, 159, 134, 139)

## Basic histogram and boxplot
hist(x,
     main = "Shelf life (days)",
     xlab = "Days")

boxplot(x,
        horizontal = TRUE,
        main = "Boxplot of shelf life")

## Normal Q–Q plot
qqnorm(x,
       main = "Normal Q–Q plot of shelf life")
qqline(x, col = "red", lwd = 2)

## Shapiro–Wilk normality test
shapiro.test(x)

## 2.31  Photoresist thickness data (kA)

thick_95  <- c(11.176, 7.089, 8.097, 11.739, 11.291, 10.759, 6.467, 8.315)
thick_100 <- c( 5.263, 6.748, 7.461,  7.015,  8.133,  7.418, 3.772, 8.963)

# Basic summaries
n1  <- length(thick_95)
n2  <- length(thick_100)
m1  <- mean(thick_95)
m2  <- mean(thick_100)
s1  <- sd(thick_95)
s2  <- sd(thick_100)

n1; n2; m1; m2; s1; s2

## (a) / (b) Two-sample t-test (higher temp => lower mean thickness)
## H0: mu_100 >= mu_95  vs  Ha: mu_100 < mu_95  (or equivalently mu_95 - mu_100 > 0)

t_test_res <- t.test(thick_95, thick_100,
                     alternative = "greater",   # mean(95) > mean(100)
                     var.equal   = TRUE)        # pooled two-sample t-test

t_test_res$statistic   # t
t_test_res$parameter   # df
t_test_res$p.value     # P-value

## (c) 95% CI for difference in means (mu_95 - mu_100)

ci_95 <- t.test(thick_95, thick_100,
                var.equal = TRUE,
                conf.level = 0.95)$conf.int
ci_95

## (d) Dot (strip) plots for each temperature

par(mfrow = c(1, 2))

stripchart(thick_95,
           method = "stack",
           main   = "Dot plot: 95°C",
           xlab   = "Thickness (kA)")

stripchart(thick_100,
           method = "stack",
           main   = "Dot plot: 100°C",
           xlab   = "Thickness (kA)")

## (e) Normality checks for each temperature

par(mfrow = c(1, 2))

qqnorm(thick_95,  main = "Q-Q plot: 95°C");  qqline(thick_95,  col = "red")
qqnorm(thick_100, main = "Q-Q plot: 100°C"); qqline(thick_100, col = "red")

shapiro.test(thick_95)
shapiro.test(thick_100)

## (f) Power to detect true difference of 2.5 kA with current n1 = n2 = 8
## Use pooled sd from the data as estimate

sp2 <- ((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2)
sp  <- sqrt(sp2)

power_2_5 <- power.t.test(n = n1,
                          delta = 2.5,
                          sd = sp,
                          sig.level = 0.05,
                          type = "two.sample",
                          alternative = "two.sided")
power_2_5$power

## (g) Required n to detect true difference of 1.5 kA with power 0.9

n_req_1_5 <- power.t.test(n = NULL,
                          delta = 1.5,
                          sd = sp,
                          sig.level = 0.05,
                          power = 0.9,
                          type = "two.sample",
                          alternative = "two.sided")
n_req_1_5$n    # per group (round up)

## 2.32  Cool-down time / appearance score data

score_10 <- c(1, 2, 3, 5, 1,
              5, 2, 8, 3, 5,
              7, 8, 2, 3, 3,
              5, 8, 2, 5, 3)

score_20 <- c(3, 6, 3, 2, 1,
              3, 2, 8, 2, 3,
              7, 6, 8, 8, 4,
              6, 2, 6, 7, 7)

## Basic summaries
n1  <- length(score_10)
n2  <- length(score_20)
m1  <- mean(score_10)
m2  <- mean(score_20)
s1  <- sd(score_10)
s2  <- sd(score_20)

n1; n2; m1; m2; s1; s2

## (a), (b) Two-sample t-test
## Claim: longer cool-down (20 s) -> fewer defects -> higher mean score
## So test mu_20 > mu_10  (or equivalently mu_20 - mu_10 > 0)

t_test_res <- t.test(score_20, score_10,
                     alternative = "greater",
                     var.equal   = TRUE)

t_test_res$statistic   # t
t_test_res$parameter   # df
t_test_res$p.value     # P-value

## (c) 95% CI for difference in means (mu_20 - mu_10)

ci_95 <- t.test(score_20, score_10,
                var.equal = TRUE,
                conf.level = 0.95)$conf.int
ci_95

## (d) Dot (strip) plots for each cool-down time

par(mfrow = c(1, 2))

stripchart(score_10,
           method = "stack",
           main   = "Dot plot: 10 seconds",
           xlab   = "Appearance score")

stripchart(score_20,
           method = "stack",
           main   = "Dot plot: 20 seconds",
           xlab   = "Appearance score")

## (e) Normality checks by group

par(mfrow = c(1, 2))
qqnorm(score_10, main = "Q-Q plot: 10 s"); qqline(score_10, col = "red")
qqnorm(score_20, main = "Q-Q plot: 20 s"); qqline(score_20, col = "red")

shapiro.test(score_10)
shapiro.test(score_20)

## 2.35  Twin intelligence scores

birth1 <- c(6.08, 6.22, 7.99, 7.44, 6.48,
            7.99, 6.32, 7.60, 6.03, 7.52)

birth2 <- c(5.73, 5.80, 8.42, 6.84, 6.43,
            8.76, 7.02, 7.62, 6.59, 7.67)

## Differences: Birth order 1 minus birth order 2
d <- birth1 - birth2

length(d); mean(d); sd(d)

## (b) 95% CI for mean difference, and evidence that mean score depends on birth order?

t.test(birth1, birth2,
       paired     = TRUE,
       conf.level = 0.95)

## (c) Hypothesis test that mean score does NOT depend on birth order
## (this is exactly the same paired t-test; we just focus on the test part)

t.test(birth1, birth2,
       paired     = TRUE,
       alternative = "two.sided")

qqnorm(d); qqline(d, col = "red")
shapiro.test(d)

## 2.47  Construct data: strong within-pair pattern, but noisy between subjects

set.seed(1)

# 8 subjects
x <- c(10, 20, 30, 40, 50, 60, 70, 80)          
y <- x + 5 + rnorm(length(x), mean = 0, sd = .1)

x; y

## Paired t-test: very large t

t.test(x, y, paired = TRUE)

## Usual two-sample (pooled) t-test ignoring pairing: small t

t.test(x, y, paired = FALSE, var.equal = TRUE)

## 2.28  Burning times (minutes) for two flare types

type1 <- c(65, 81, 57, 66, 82,
           82, 67, 59, 75, 70)

type2 <- c(64, 71, 83, 59, 65,
           56, 69, 74, 82, 79)

## Basic summaries
n1  <- length(type1)
n2  <- length(type2)
m1  <- mean(type1)
m2  <- mean(type2)
s1  <- sd(type1)
s2  <- sd(type2)

n1; n2; m1; m2; s1; s2

## F-test for equal variances
var_test <- var.test(type1, type2, ratio = 1, alternative = "two.sided")
var_test

t_test <- t.test(type1, type2,
                 alternative = "two.sided",
                 var.equal   = TRUE)
t_test$statistic   # t
t_test$parameter   # df
t_test$p.value     # P-value

par(mfrow = c(1, 2))
qqnorm(type1, main = "Q-Q plot: Type 1"); qqline(type1, col = "red")
qqnorm(type2, main = "Q-Q plot: Type 2"); qqline(type2, col = "red")

shapiro.test(type1)
shapiro.test(type2)

## Pooled standard deviation from 2.28
sp2 <- ((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2)
sp  <- sqrt(sp2)

## 2.48: power if true mean difference is 2 minutes (delta = 2)
power_2 <- power.t.test(n = n1,
                         delta = 2,
                         sd = sp,
                         sig.level = 0.05,
                         type = "two.sample",
                         alternative = "two.sided")
power_2$power

n_req_1 <- power.t.test(n = NULL,
                         delta = 1,
                         sd = sp,
                         sig.level = 0.05,
                         power = 0.90,
                         type = "two.sample",
                         alternative = "two.sided")
n_req_1$n   # per group (round up)
