library(dplyr)
library(ggplot2)
library(agricolae)

out_dir <- "Stats 490/HW4/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)


# ---- 5.7 ----

anova_df <- data.frame(
  Source  = c("A", "B", "Interaction (A:B)", "Error", "Total"),
  DF      = c(1, 3, 3, 8, 15),
  SS      = c(0.0002, 180.378, 8.479, 158.797, 347.654),
  MS      = c(0.0002, 60.126, 2.826, 19.850, NA),
  F_value = c(0.00001, 3.029, 0.142, NA, NA),
  P_value = c(0.998, 0.093, 0.932, NA, NA)
)
print(anova_df)

pA  <- pf(0.0002 / 19.850, df1 = 1, df2 = 8, lower.tail = FALSE)
pB  <- pf(60.126 / 19.850, df1 = 3, df2 = 8, lower.tail = FALSE)
pAB <- pf(2.826  / 19.850, df1 = 3, df2 = 8, lower.tail = FALSE)
cat("P(A) =", round(pA, 4), " P(B) =", round(pB, 4), " P(AB) =", round(pAB, 4), "\n")


# ---- 5.9 ----

Feed  <- factor(rep(c("0.20", "0.25", "0.30"), each = 12))
Depth <- factor(rep(rep(c("0.15", "0.18", "0.20", "0.25"), each = 3), times = 3))

y <- c(
  74, 64, 60,  79, 68, 73,  82, 88, 92,  99, 104,  96,  # Feed 0.20
  86, 88, 99, 104, 88,104, 108, 95,108, 110,  99, 114,  # Feed 0.25
  98,102, 99,  99, 95,104, 110, 99,108, 111, 107, 114   # Feed 0.30
)

dat <- data.frame(Feed, Depth, y)

fit <- aov(y ~ Feed * Depth, data = dat)
summary(fit)

# marginal means
dat %>% group_by(Feed)  %>% summarise(mean_y = mean(y), sd_y = sd(y)) %>% print()
dat %>% group_by(Depth) %>% summarise(mean_y = mean(y), sd_y = sd(y)) %>% print()

TukeyHSD(fit, "Feed")
TukeyHSD(fit, "Depth")

# diagnostic plots
png(file.path(out_dir, "q59_resid_vs_fitted.png"), width = 700, height = 500)
plot(fit, which = 1); dev.off()

png(file.path(out_dir, "q59_qq.png"), width = 700, height = 500)
plot(fit, which = 2); dev.off()

png(file.path(out_dir, "q59_interaction.png"), width = 700, height = 500)
interaction.plot(dat$Depth, dat$Feed, dat$y,
                 col = c("steelblue","tomato","forestgreen"),
                 lwd = 2, lty = 1,
                 xlab = "Depth of Cut (in)", ylab = "Mean Surface Finish",
                 main = "Interaction Plot (Feed x Depth)")
dev.off()


# ---- 5.10 ----

ybar_020  <- mean(dat$y[dat$Feed == "0.20"])
ybar_025  <- mean(dat$y[dat$Feed == "0.25"])
MSE       <- summary(fit)[[1]]["Residuals", "Mean Sq"]
df_e      <- summary(fit)[[1]]["Residuals", "Df"]
n1 <- n2  <- 12

delta_hat <- ybar_020 - ybar_025
SE        <- sqrt(MSE * (1/n1 + 1/n2))
t_crit    <- qt(0.975, df = df_e)

cat("95% CI: (", round(delta_hat - t_crit * SE, 2), ",",
                 round(delta_hat + t_crit * SE, 2), ")\n")


# ---- 5.17 ----

Temp     <- factor(rep(c("150", "160", "170"), each = 6))
Pressure <- factor(rep(rep(c("200", "215", "230"), each = 2), times = 3))

y <- c(
  90.4, 90.2,  90.7, 90.6,  90.2, 90.4,  # Temp 150
  90.1, 90.3,  90.5, 90.6,  89.9, 90.1,  # Temp 160
  90.5, 90.7,  90.8, 90.9,  90.4, 90.1   # Temp 170
)

dat <- data.frame(Temp, Pressure, y)

fit <- aov(y ~ Temp * Pressure, data = dat)
summary(fit)

dat %>% group_by(Pressure) %>% summarise(mean_y = mean(y), sd_y = sd(y)) %>% print()

tukey_press <- TukeyHSD(fit, "Pressure", conf.level = 0.95)
print(tukey_press)
plot(tukey_press)

png(file.path(out_dir, "q517_resid_vs_fitted.png"), width = 700, height = 500)
plot(fit, which = 1); dev.off()

png(file.path(out_dir, "q517_qq.png"), width = 700, height = 500)
plot(fit, which = 2); dev.off()


# ---- 5.39 ----

Feed  <- factor(rep(c("0.20","0.25","0.30"), each = 12))
Depth <- factor(rep(rep(c("0.15","0.18","0.20","0.25"), each = 3), times = 3))
Block <- factor(rep(1:3, times = 12))

y <- c(
  74, 64, 60,  79, 68, 73,  82, 88, 92,  99, 104,  96,
  86, 88, 99, 104, 88,104, 108, 95,108, 110,  99, 114,
  98,102, 99,  99, 95,104, 110, 99,108, 111, 107, 114
)

dat_blk <- data.frame(Feed, Depth, Block, y)

fit_blk  <- aov(y ~ Feed * Depth + Block, data = dat_blk)
fit_orig <- aov(y ~ Feed * Depth,         data = dat_blk)
summary(fit_blk)

tab      <- summary(fit_blk)[[1]]
MS_block <- tab["Block",     "Mean Sq"]
MS_error <- tab["Residuals", "Mean Sq"]

sigma2_block <- (MS_block - MS_error) / (3 * 4)
cat("sigma^2_block =", round(max(sigma2_block, 0), 4), "\n")
cat("MSE without blocks:", round(summary(fit_orig)[[1]]["Residuals","Mean Sq"], 4),
    " with blocks:", round(MS_error, 4), "\n")

dat_blk %>% group_by(Block) %>% summarise(mean_y = mean(y)) %>% print()

png(file.path(out_dir, "q539_resid_vs_fitted.png"), width = 700, height = 500)
plot(fit_blk, which = 1); dev.off()

png(file.path(out_dir, "q539_qq.png"), width = 700, height = 500)
plot(fit_blk, which = 2); dev.off()

png(file.path(out_dir, "q539_boxplot_block.png"), width = 700, height = 500)
ggplot(dat_blk, aes(x = Block, y = y, fill = Block)) +
  geom_boxplot(alpha = 0.7) +
  labs(x = "Block", y = "Surface Finish") +
  theme_bw(base_size = 13) +
  theme(legend.position = "none")
dev.off()

cat("Done. Plots saved to", out_dir, "\n")











