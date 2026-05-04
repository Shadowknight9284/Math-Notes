# ============================================================
# Question 5.7 — Two-way Factorial ANOVA (table reconstruction)
# HW: Stats 490 / HW4
# ============================================================

library(ggplot2)

out_dir <- "Stats 490/HW4/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------
# 1. Reproduce completed ANOVA table from given values
# ------------------------------------------------------------------
anova_df <- data.frame(
  Source  = c("A", "B", "Interaction (A:B)", "Error", "Total"),
  DF      = c(1,   3,   3,      8,       15),
  SS      = c(0.0002, 180.378, 8.479, 158.797, 347.654),
  MS      = c(0.0002,  60.126, 2.826,  19.850,      NA),
  F_value = c(0.00001,  3.029, 0.142,      NA,      NA),
  P_value = c(0.998,    0.093, 0.932,      NA,      NA)
)
print(anova_df)

# ------------------------------------------------------------------
# 2. Verify P-values using pf()
# ------------------------------------------------------------------
pA  <- pf(0.0002 / 19.850, df1 = 1, df2 = 8, lower.tail = FALSE)
pB  <- pf(60.126 / 19.850, df1 = 3, df2 = 8, lower.tail = FALSE)
pAB <- pf( 2.826 / 19.850, df1 = 3, df2 = 8, lower.tail = FALSE)

cat("P(A)  =", round(pA,  4), "\n")
cat("P(B)  =", round(pB,  4), "\n")
cat("P(AB) =", round(pAB, 4), "\n")

# ------------------------------------------------------------------
# 3. Save ANOVA table as a formatted PNG  (fixed: ggsave OUTSIDE ggplot)
# ------------------------------------------------------------------
tbl_label <- paste0(
  "Two-way ANOVA: y versus A, B\n\n",
  "Source            DF     SS        MS       F        P\n",
  "A                  1     0.0002    0.0002   0.00001  0.998\n",
  "B                  3   180.378    60.126    3.029    0.093\n",
  "A:B                3     8.479     2.826    0.142    0.932\n",
  "Error              8   158.797    19.850\n",
  "Total             15   347.654"
)

tbl_plot <- ggplot() +
  annotate("text",
           x      = 0.5,
           y      = 0.5,
           label  = tbl_label,
           family = "mono",
           size   = 3.5,
           hjust  = 0.5,
           vjust  = 0.5) +
  theme_void()

ggsave(
  filename = file.path(out_dir, "q57_anova_table.png"),
  plot     = tbl_plot,
  width    = 8,
  height   = 3,
  dpi      = 150
)

cat("Saved:", file.path(out_dir, "q57_anova_table.png"), "\n")


# ============================================================
# Question 5.9 — Two-way Factorial ANOVA
# Feed Rate × Depth of Cut on Surface Finish
# HW: Stats 490 / HW4
# ============================================================

library(dplyr)
library(ggplot2)
library(agricolae)

out_dir <- "Stats 490/HW4/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------
# 1. Enter data — 3 Feed Rates × 4 Depths × 3 replicates = 36 obs
#    Layout: rows = feed rate, cols = depth of cut
#    Reading cell values in order: depth 0.15, 0.18, 0.20, 0.25
# ------------------------------------------------------------------

Feed  <- factor(rep(c("0.20", "0.25", "0.30"), each = 12))
Depth <- factor(rep(rep(c("0.15", "0.18", "0.20", "0.25"), each = 3), times = 3))

y <- c(
  # Feed = 0.20
  74, 64, 60,   # Depth = 0.15
  79, 68, 73,   # Depth = 0.18
  82, 88, 92,   # Depth = 0.20
  99,104, 96,   # Depth = 0.25
  # Feed = 0.25
  86, 88, 99,   # Depth = 0.15
 104, 88,104,   # Depth = 0.18
 108, 95,108,   # Depth = 0.20
 110, 99,114,   # Depth = 0.25
  # Feed = 0.30
  98,102, 99,   # Depth = 0.15
  99, 95,104,   # Depth = 0.18
 110, 99,108,   # Depth = 0.20
 111,107,114    # Depth = 0.25
)

dat <- data.frame(Feed, Depth, y)
str(dat)

# ------------------------------------------------------------------
# 2. Fit two-way factorial model with interaction
# ------------------------------------------------------------------
fit <- aov(y ~ Feed * Depth, data = dat)
summary(fit)

# ------------------------------------------------------------------
# 3. Cell means and marginal means
# ------------------------------------------------------------------
# Marginal means by Feed Rate (for part c)
feed_means <- dat %>%
  group_by(Feed) %>%
  summarise(mean_y = mean(y), sd_y = sd(y), n = n())
print(feed_means)

# Marginal means by Depth
depth_means <- dat %>%
  group_by(Depth) %>%
  summarise(mean_y = mean(y), sd_y = sd(y), n = n())
print(depth_means)

# Cell means
cell_means <- dat %>%
  group_by(Feed, Depth) %>%
  summarise(mean_y = mean(y), sd_y = sd(y), .groups = "drop")
print(cell_means)

# ------------------------------------------------------------------
# 4. Multiple comparisons on Feed Rate (primary factor of interest)
# ------------------------------------------------------------------
# Fisher LSD
lsd_feed <- LSD.test(fit, "Feed", p.adj = "none")
print(lsd_feed)

# Pairwise t-test
pairwise.t.test(dat$y, dat$Feed, p.adjust.method = "none", pool.sd = TRUE)

# Tukey HSD
tukey_feed <- TukeyHSD(fit, "Feed")
print(tukey_feed)

# Also for Depth
tukey_depth <- TukeyHSD(fit, "Depth")
print(tukey_depth)

# ------------------------------------------------------------------
# 5. Diagnostic plots
# ------------------------------------------------------------------

# (a) Residuals vs Fitted
png(file.path(out_dir, "q59_resid_vs_fitted.png"), width = 700, height = 500)
plot(fit, which = 1, main = "Q5.9 — Residuals vs Fitted")
dev.off()

# (b) Normal Q-Q
png(file.path(out_dir, "q59_qq.png"), width = 700, height = 500)
plot(fit, which = 2, main = "Q5.9 — Normal Q-Q")
dev.off()

# (c) Boxplot by Feed Rate
png(file.path(out_dir, "q59_boxplot_feed.png"), width = 700, height = 500)
ggplot(dat, aes(x = Feed, y = y, fill = Feed)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Q5.9 — Surface Finish by Feed Rate",
       x = "Feed Rate (in/min)", y = "Surface Finish") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
dev.off()

# (d) Boxplot by Depth
png(file.path(out_dir, "q59_boxplot_depth.png"), width = 700, height = 500)
ggplot(dat, aes(x = Depth, y = y, fill = Depth)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Q5.9 — Surface Finish by Depth of Cut",
       x = "Depth of Cut (in)", y = "Surface Finish") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
dev.off()

# (e) Interaction plot
png(file.path(out_dir, "q59_interaction.png"), width = 700, height = 500)
interaction.plot(dat$Depth, dat$Feed, dat$y,
                 col = c("steelblue","tomato","forestgreen"),
                 lwd = 2, lty = 1,
                 xlab = "Depth of Cut (in)",
                 ylab = "Mean Surface Finish",
                 main = "Q5.9 — Interaction Plot (Feed × Depth)",
                 legend = TRUE)
dev.off()

# ------------------------------------------------------------------
# 6. ggplot2 mean ± 95% CI for Feed Rate
# ------------------------------------------------------------------
t_crit <- qt(0.975, df = df.residual(fit))

ci_feed <- dat %>%
  group_by(Feed) %>%
  summarise(
    mean_y = mean(y),
    se     = sd(y) / sqrt(n()),
    ci     = t_crit * se,
    .groups = "drop"
  )

p_ci <- ggplot(ci_feed, aes(x = Feed, y = mean_y, color = Feed)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = mean_y - ci, ymax = mean_y + ci), width = 0.2, linewidth = 1) +
  labs(title = "Q5.9 — Mean Surface Finish ± 95% CI by Feed Rate",
       x = "Feed Rate (in/min)", y = "Mean Surface Finish") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "q59_mean_ci_feed.png"),
       plot = p_ci, width = 7, height = 5, dpi = 150)

cat("All plots saved to", out_dir, "\n")

# ============================================================
# Question 5.10 — 95% CI for mean difference in surface finish
# Feed Rate 0.20 vs 0.25 in/min (from Problem 5.9 data)
# HW: Stats 490 / HW4
# ============================================================

# -- Values from Problem 5.9 ANOVA output --
ybar_020 <- 81.58333   # marginal mean, Feed = 0.20
ybar_025 <- 100.25000  # marginal mean, Feed = 0.25
n1       <- 12         # replicates per feed level (4 depths × 3 reps)
n2       <- 12
MSE      <- 37.19444   # MS(Error) from aov(y ~ Feed * Depth)
df_e     <- 24         # df(Error)

# -- Point estimate of the difference --
delta_hat <- ybar_020 - ybar_025
cat("Point estimate (0.20 - 0.25):", round(delta_hat, 4), "\n")

# -- Standard error --
SE <- sqrt(MSE * (1/n1 + 1/n2))
cat("Standard error:", round(SE, 4), "\n")

# -- t critical value --
t_crit <- qt(0.975, df = df_e)
cat("t(0.025, 24):", round(t_crit, 4), "\n")

# -- 95% CI --
CI_lower <- delta_hat - t_crit * SE
CI_upper <- delta_hat + t_crit * SE
cat("95% CI: (", round(CI_lower, 2), ",", round(CI_upper, 2), ")\n")

# ============================================================
# Question 5.17 — Tukey's Test on Pressure Factor
# Data from Problem 5.8: Temperature × Pressure factorial
# HW: Stats 490 / HW4
# ============================================================

library(dplyr)
library(ggplot2)
library(agricolae)

out_dir <- "Stats 490/HW4/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------
# 1. Enter data — 3 Temperatures × 3 Pressures × 2 replicates = 18
#    Reading row by row: Temp 150 (P200, P215, P230), etc.
# ------------------------------------------------------------------

Temp     <- factor(rep(c("150", "160", "170"), each = 6))
Pressure <- factor(rep(rep(c("200", "215", "230"), each = 2), times = 3))

y <- c(
  # Temp = 150
  90.4, 90.2,   # Pressure = 200
  90.7, 90.6,   # Pressure = 215
  90.2, 90.4,   # Pressure = 230
  # Temp = 160
  90.1, 90.3,   # Pressure = 200
  90.5, 90.6,   # Pressure = 215
  89.9, 90.1,   # Pressure = 230
  # Temp = 170
  90.5, 90.7,   # Pressure = 200
  90.8, 90.9,   # Pressure = 215
  90.4, 90.1    # Pressure = 230
)

dat <- data.frame(Temp, Pressure, y)
str(dat)

# ------------------------------------------------------------------
# 2. Fit two-way factorial model with interaction
# ------------------------------------------------------------------
fit <- aov(y ~ Temp * Pressure, data = dat)
summary(fit)

# ------------------------------------------------------------------
# 3. Marginal means by Pressure
# ------------------------------------------------------------------
press_means <- dat %>%
  group_by(Pressure) %>%
  summarise(mean_y = mean(y), sd_y = sd(y), n = n())
print(press_means)

# Marginal means by Temperature
temp_means <- dat %>%
  group_by(Temp) %>%
  summarise(mean_y = mean(y), sd_y = sd(y), n = n())
print(temp_means)

# ------------------------------------------------------------------
# 4. Tukey's test on Pressure (primary request)
# ------------------------------------------------------------------
tukey_press <- TukeyHSD(fit, "Pressure", conf.level = 0.95)
print(tukey_press)
plot(tukey_press)

# Also run for Temperature for completeness
tukey_temp <- TukeyHSD(fit, "Temp", conf.level = 0.95)
print(tukey_temp)

# Fisher LSD on Pressure for reference
lsd_press <- LSD.test(fit, "Pressure", p.adj = "none")
print(lsd_press)

# ------------------------------------------------------------------
# 5. Diagnostic plots
# ------------------------------------------------------------------

# (a) Residuals vs Fitted
png(file.path(out_dir, "q517_resid_vs_fitted.png"), width = 700, height = 500)
plot(fit, which = 1, main = "Q5.17 — Residuals vs Fitted")
dev.off()

# (b) Normal Q-Q
png(file.path(out_dir, "q517_qq.png"), width = 700, height = 500)
plot(fit, which = 2, main = "Q5.17 — Normal Q-Q")
dev.off()

# (c) Boxplot by Pressure
png(file.path(out_dir, "q517_boxplot_pressure.png"), width = 700, height = 500)
ggplot(dat, aes(x = Pressure, y = y, fill = Pressure)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Q5.17 — Response by Pressure",
       x = "Pressure (psig)", y = "Response") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
dev.off()

# (d) Boxplot by Temperature
png(file.path(out_dir, "q517_boxplot_temp.png"), width = 700, height = 500)
ggplot(dat, aes(x = Temp, y = y, fill = Temp)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Q5.17 — Response by Temperature",
       x = "Temperature (°C)", y = "Response") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
dev.off()

# (e) Interaction plot
png(file.path(out_dir, "q517_interaction.png"), width = 700, height = 500)
interaction.plot(dat$Pressure, dat$Temp, dat$y,
                 col  = c("steelblue","tomato","forestgreen"),
                 lwd  = 2, lty = 1,
                 xlab = "Pressure (psig)",
                 ylab = "Mean Response",
                 main = "Q5.17 — Interaction Plot (Temp × Pressure)")
dev.off()

# ------------------------------------------------------------------
# 6. ggplot2 mean ± 95% CI for Pressure
# ------------------------------------------------------------------
t_crit <- qt(0.975, df = df.residual(fit))

ci_press <- dat %>%
  group_by(Pressure) %>%
  summarise(
    mean_y = mean(y),
    se     = sd(y) / sqrt(n()),
    ci     = t_crit * se,
    .groups = "drop"
  )

p_ci <- ggplot(ci_press, aes(x = Pressure, y = mean_y, color = Pressure)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = mean_y - ci, ymax = mean_y + ci),
                width = 0.2, linewidth = 1) +
  labs(title = "Q5.17 — Mean Response ± 95% CI by Pressure",
       x = "Pressure (psig)", y = "Mean Response") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "q517_mean_ci_pressure.png"),
       plot = p_ci, width = 7, height = 5, dpi = 150)

cat("All plots saved to", out_dir, "\n")

# ============================================================
# Question 5.39 — 5.9 Data Re-analyzed as Factorial in Blocks
# Model: y ~ Feed * Depth + Block
# HW: Stats 490 / HW4
# ============================================================

library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW4/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------------------------
# 1. Build data — same 36 obs as 5.9, now with Block assigned
#    Block k = replicate k (k=1,2,3): first obs in each cell → Block 1,
#    second → Block 2, third → Block 3
# ------------------------------------------------------------------

Feed  <- factor(rep(c("0.20","0.25","0.30"), each = 12))
Depth <- factor(rep(rep(c("0.15","0.18","0.20","0.25"), each = 3), times = 3))
Block <- factor(rep(1:3, times = 12))   # cycles 1,2,3 within each cell

y <- c(
  # Feed = 0.20
  74, 64, 60,   # Depth = 0.15
  79, 68, 73,   # Depth = 0.18
  82, 88, 92,   # Depth = 0.20
  99,104, 96,   # Depth = 0.25
  # Feed = 0.25
  86, 88, 99,   # Depth = 0.15
 104, 88,104,   # Depth = 0.18
 108, 95,108,   # Depth = 0.20
 110, 99,114,   # Depth = 0.25
  # Feed = 0.30
  98,102, 99,   # Depth = 0.15
  99, 95,104,   # Depth = 0.18
 110, 99,108,   # Depth = 0.20
 111,107,114    # Depth = 0.25
)

dat_blk <- data.frame(Feed, Depth, Block, y)
str(dat_blk)

# ------------------------------------------------------------------
# 2. Fit factorial-in-blocks model: Feed * Depth + Block
#    Block is fixed (RCBD approach for variance component estimation)
# ------------------------------------------------------------------
fit_blk <- aov(y ~ Feed * Depth + Block, data = dat_blk)
summary(fit_blk)

# Also reprint original 5.9 model for comparison
fit_orig <- aov(y ~ Feed * Depth, data = dat_blk)
summary(fit_orig)

# ------------------------------------------------------------------
# 3. Variance component estimate for blocks
#    MS(Block) = sigma^2 + abn * sigma^2_block  (a=3, b=4, n=1)
#    => sigma^2_block = (MS_Block - MS_Error) / (a*b)
# ------------------------------------------------------------------
tab_blk  <- summary(fit_blk)[[1]]
MS_block <- tab_blk["Block",       "Mean Sq"]
MS_error <- tab_blk["Residuals",   "Mean Sq"]

a <- 3   # Feed levels
b <- 4   # Depth levels

sigma2_block     <- (MS_block - MS_error) / (a * b)
sigma2_block_est <- max(sigma2_block, 0)   # truncate at 0 if negative

cat("MS(Block) =", round(MS_block, 4), "\n")
cat("MS(Error) =", round(MS_error, 4), "\n")
cat("sigma^2_block (raw)    =", round(sigma2_block,     4), "\n")
cat("sigma^2_block (trunc)  =", round(sigma2_block_est, 4), "\n")

# Compare MS(Error) with and without blocking
MS_error_orig <- summary(fit_orig)[[1]]["Residuals", "Mean Sq"]
cat("\nMS(Error) WITHOUT blocking:", round(MS_error_orig, 4), "\n")
cat("MS(Error) WITH    blocking:", round(MS_error,      4), "\n")

# ------------------------------------------------------------------
# 4. Marginal means
# ------------------------------------------------------------------
feed_means <- dat_blk %>%
  group_by(Feed) %>%
  summarise(mean_y = mean(y), sd_y = sd(y), n = n())
print(feed_means)

block_means <- dat_blk %>%
  group_by(Block) %>%
  summarise(mean_y = mean(y), sd_y = sd(y), n = n())
print(block_means)

# ------------------------------------------------------------------
# 5. Diagnostic plots
# ------------------------------------------------------------------

# (a) Residuals vs Fitted
png(file.path(out_dir, "q539_resid_vs_fitted.png"), width = 700, height = 500)
plot(fit_blk, which = 1, main = "Q5.39 — Residuals vs Fitted (blocked)")
dev.off()

# (b) Normal Q-Q
png(file.path(out_dir, "q539_qq.png"), width = 700, height = 500)
plot(fit_blk, which = 2, main = "Q5.39 — Normal Q-Q (blocked)")
dev.off()

# (c) Boxplot by Block
png(file.path(out_dir, "q539_boxplot_block.png"), width = 700, height = 500)
ggplot(dat_blk, aes(x = Block, y = y, fill = Block)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Q5.39 — Surface Finish by Block (Replicate)",
       x = "Block", y = "Surface Finish") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
dev.off()

# (d) Boxplot by Feed
png(file.path(out_dir, "q539_boxplot_feed.png"), width = 700, height = 500)
ggplot(dat_blk, aes(x = Feed, y = y, fill = Feed)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Q5.39 — Surface Finish by Feed Rate",
       x = "Feed Rate (in/min)", y = "Surface Finish") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
dev.off()

# ------------------------------------------------------------------
# 6. ggplot2 mean ± 95% CI for Feed Rate
# ------------------------------------------------------------------
t_crit <- qt(0.975, df = df.residual(fit_blk))

ci_feed <- dat_blk %>%
  group_by(Feed) %>%
  summarise(
    mean_y = mean(y),
    se     = sd(y) / sqrt(n()),
    ci     = t_crit * se,
    .groups = "drop"
  )

p_ci <- ggplot(ci_feed, aes(x = Feed, y = mean_y, color = Feed)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = mean_y - ci, ymax = mean_y + ci),
                width = 0.2, linewidth = 1) +
  labs(title = "Q5.39 — Mean Surface Finish ± 95% CI by Feed Rate (blocked)",
       x = "Feed Rate (in/min)", y = "Mean Surface Finish") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "q539_mean_ci_feed.png"),
       plot = p_ci, width = 7, height = 5, dpi = 150)

cat("All plots saved to", out_dir, "\n")
