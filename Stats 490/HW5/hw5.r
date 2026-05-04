# Problem 6.5 – 2^3 factorial with 3 replicates
library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW5/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Data entry -------------------------------------------------------------

A <- rep(c(-1, -1, -1, -1,  1,  1,  1,  1), each = 3)
B <- rep(c(-1,  1, -1,  1, -1,  1, -1,  1), each = 3)
C <- rep(c(-1, -1,  1,  1,  1,  1, -1, -1), each = 3)

y <- c(
  22, 31, 25,  # (1): A-, B-, C-
  32, 43, 29,  # a:  A+, B-, C-
  35, 34, 50,  # b:  A-, B+, C-
  55, 47, 46,  # ab: A+, B+, C-
  44, 45, 38,  # c:  A-, B-, C+
  40, 37, 36,  # ac: A+, B-, C+
  60, 50, 54,  # bc: A-, B+, C+
  39, 41, 47   # abc:A+, B+, C+
)

dat65 <- data.frame(
  A = A,
  B = B,
  C = C,
  y = y
)

str(dat65)
dat65 %>% head()
dat65 %>% group_by(A, B, C) %>%
  summarise(mean_y = mean(y), .groups = "drop")

# 2. Fit full 2^3 model ----------------------------------------------------

mod65 <- lm(y ~ A * B * C, data = dat65)
summary(mod65)

# 3. ANOVA for factor significance (part b) --------------------------------

anova(mod65)

# 4. Factor effect estimates (part a) --------------------------------------
# Effects are 2 * coefficient in coded units
coef_tab <- coef(mod65)
effects65 <- 2 * coef_tab[-1]  # drop intercept
effects65

# 5. Regression model (part c) ---------------------------------------------

coef_tab  # use these coefficients to write the coded regression equation


# 6. Residual analysis (part d) --------------------------------------------

resid65 <- resid(mod65)

# Residual vs fitted
p_resid <- ggplot(dat65, aes(x = fitted(mod65), y = resid65)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Fitted values", y = "Residuals",
       title = "6.5 Residuals vs Fitted")

ggsave(file.path(out_dir, "6.5_resid_vs_fitted.png"),
       p_resid, width = 5, height = 4, dpi = 300)

# Normal Q-Q plot
p_qq <- ggplot(dat65, aes(sample = resid65)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "6.5 Normal Q-Q Plot of Residuals")

ggsave(file.path(out_dir, "6.5_qq.png"),
       p_qq, width = 5, height = 4, dpi = 300)

# 7. Main-effect and interaction plots (part e) -----------------------------

p_me_A <- ggplot(dat65, aes(x = factor(A), y = y)) +
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun = mean, geom = "line", aes(group = 1)) +
  labs(x = "A (coded)", y = "Mean tool life",
       title = "6.5 Main Effect of A")

p_me_B <- ggplot(dat65, aes(x = factor(B), y = y)) +
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun = mean, geom = "line", aes(group = 1)) +
  labs(x = "B (coded)", y = "Mean tool life",
       title = "6.5 Main Effect of B")

p_me_C <- ggplot(dat65, aes(x = factor(C), y = y)) +
  stat_summary(fun = mean, geom = "point") +
  stat_summary(fun = mean, geom = "line", aes(group = 1)) +
  labs(x = "C (coded)", y = "Mean tool life",
       title = "6.5 Main Effect of C")

ggsave(file.path(out_dir, "6.5_main_A.png"),
       p_me_A, width = 4, height = 3, dpi = 300)
ggsave(file.path(out_dir, "6.5_main_B.png"),
       p_me_B, width = 4, height = 3, dpi = 300)
ggsave(file.path(out_dir, "6.5_main_C.png"),
       p_me_C, width = 4, height = 3, dpi = 300)

p_int_AB <- ggplot(dat65, aes(x = factor(A), y = y,
                              group = factor(B), linetype = factor(B))) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point") +
  labs(x = "A (coded)", y = "Mean tool life",
       linetype = "B (coded)",
       title = "6.5 Interaction A×B")

p_int_AC <- ggplot(dat65, aes(x = factor(A), y = y,
                              group = factor(C), linetype = factor(C))) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point") +
  labs(x = "A (coded)", y = "Mean tool life",
       linetype = "C (coded)",
       title = "6.5 Interaction A×C")

p_int_BC <- ggplot(dat65, aes(x = factor(B), y = y,
                              group = factor(C), linetype = factor(C))) +
  stat_summary(fun = mean, geom = "line") +
  stat_summary(fun = mean, geom = "point") +
  labs(x = "B (coded)", y = "Mean tool life",
       linetype = "C (coded)",
       title = "6.5 Interaction B×C")

ggsave(file.path(out_dir, "6.5_int_AB.png"),
       p_int_AB, width = 4, height = 3, dpi = 300)
ggsave(file.path(out_dir, "6.5_int_AC.png"),
       p_int_AC, width = 4, height = 3, dpi = 300)
ggsave(file.path(out_dir, "6.5_int_BC.png"),
       p_int_BC, width = 4, height = 3, dpi = 300)


# Problem 6.6 – response surfaces from 6.5 model

# Use the reduced model (A, AB, AC) from 6.5(c)
mod65_red <- lm(y ~ A + B + C + A:B + A:C, data = dat65)
summary(mod65_red)

# Make a grid for A and B at several levels, for fixed C
grid_fun <- function(C_level, n = 51) {
  expand.grid(
    A = seq(-1, 1, length.out = n),
    B = seq(-1, 1, length.out = n),
    C = C_level
  )
}

grid_Cm1 <- grid_fun(-1)  # C = -1
grid_C0  <- grid_fun(0)   # C = 0 (midpoint)
grid_C1  <- grid_fun(1)   # C = +1

grid_Cm1$yhat <- predict(mod65_red, newdata = grid_Cm1)
grid_C0$yhat  <- predict(mod65_red, newdata = grid_C0)
grid_C1$yhat  <- predict(mod65_red, newdata = grid_C1)

# Helper for response surface and contour plots
plot_surface <- function(grid, C_label) {
  ggplot(grid, aes(x = A, y = B, z = yhat)) +
    geom_contour_filled() +
    labs(title = paste("6.6 Response surface (C =", C_label, ")"),
         x = "A (coded cutting speed)",
         y = "B (coded tool geometry)",
         fill = "Predicted life")
}

plot_contour <- function(grid, C_label) {
  ggplot(grid, aes(x = A, y = B, z = yhat)) +
    geom_contour(color = "black") +
    labs(title = paste("6.6 Contour plot (C =", C_label, ")"),
         x = "A (coded cutting speed)",
         y = "B (coded tool geometry)")
}

p_surf_Cm1 <- plot_surface(grid_Cm1, "-1")
p_surf_C0  <- plot_surface(grid_C0,  "0")
p_surf_C1  <- plot_surface(grid_C1,  "+1")

p_cont_Cm1 <- plot_contour(grid_Cm1, "-1")
p_cont_C0  <- plot_contour(grid_C0,  "0")
p_cont_C1  <- plot_contour(grid_C1,  "+1")

# Save images with 6.6_ prefix
ggsave(file.path(out_dir, "6.6_surface_Cm1.png"),
       p_surf_Cm1, width = 5, height = 4, dpi = 300)
ggsave(file.path(out_dir, "6.6_surface_C0.png"),
       p_surf_C0, width = 5, height = 4, dpi = 300)
ggsave(file.path(out_dir, "6.6_surface_C1.png"),
       p_surf_C1, width = 5, height = 4, dpi = 300)

ggsave(file.path(out_dir, "6.6_contour_Cm1.png"),
       p_cont_Cm1, width = 5, height = 4, dpi = 300)
ggsave(file.path(out_dir, "6.6_contour_C0.png"),
       p_cont_C0, width = 5, height = 4, dpi = 300)
ggsave(file.path(out_dir, "6.6_contour_C1.png"),
       p_cont_C1, width = 5, height = 4, dpi = 300)

# Problem 6.7 – SEs and 95% CIs for factor effects in 6.5

# assumes mod65 from 6.5: lm(y ~ A * B * C, data = dat65)

# Extract coefficient estimates and standard errors
coef_summary <- summary(mod65)$coefficients
coef_summary

# Effect estimates are 2 * coefficient (excluding intercept)
effects <- 2 * coef_summary[-1, "Estimate"]
se_betas <- coef_summary[-1, "Std. Error"]

# Standard error of an effect = 2 * SE(beta)
se_effects <- 2 * se_betas

# 95% t critical value with df = residual df from mod65
df_resid <- df.residual(mod65)
t_crit <- qt(0.975, df = df_resid)

# Build table of effects, SEs, and CIs
effects_tab <- data.frame(
  Effect   = rownames(coef_summary)[-1],
  Estimate = as.numeric(effects),
  SE       = as.numeric(se_effects)
) %>%
  mutate(
    CI_lower = Estimate - t_crit * SE,
    CI_upper = Estimate + t_crit * SE
  )

effects_tab

# Problem 6.10 – replicate I + 4 center points

library(dplyr)
library(ggplot2)

# Factorial part: replicate I only (first column from 6.5)
A_8 <- c(-1,  1, -1,  1, -1,  1, -1,  1)
B_8 <- c(-1, -1,  1,  1, -1, -1,  1,  1)
C_8 <- c(-1, -1, -1, -1,  1,  1,  1,  1)

y_I <- c(22, 32, 35, 55, 44, 40, 60, 39)  # replicate I responses

dat10_fact <- data.frame(
  A = A_8,
  B = B_8,
  C = C_8,
  y = y_I,
  type = "factorial"
)

# Center points: A=B=C=0 with y = 36, 40, 43, 45
dat10_center <- data.frame(
  A = 0,
  B = 0,
  C = 0,
  y = c(36, 40, 43, 45),
  type = "center"
)

dat10 <- bind_rows(dat10_fact, dat10_center)
str(dat10)
dat10

# (a) Estimate effects using factorial part only -----------------------------

mod10_fact <- lm(y ~ A * B * C, data = dat10_fact)
summary(mod10_fact)

coef10 <- coef(mod10_fact)
effects10 <- 2 * coef10[-1]  # drop intercept
effects10

# (b) Curvature test using center points -----------------------------------

# Mean of factorial runs and mean of center points
ybar_fact   <- mean(dat10_fact$y)
ybar_center <- mean(dat10_center$y)
ybar_fact
ybar_center

# Pure quadratic curvature SS (Montgomery formula)
n_fact   <- nrow(dat10_fact)    # 8 factorial runs
n_center <- nrow(dat10_center)  # 4 centers

SS_PQ <- (n_fact * n_center * (ybar_fact - ybar_center)^2) /
         (n_fact + n_center)
SS_PQ

# Error MS from 6.5 (pooled three replicates)
MS_E_65 <- 30.1667  # from your 6.5 ANOVA
F_curv <- SS_PQ / MS_E_65
F_curv

# p-value with 1 and df_error=16 (from 6.5)
p_curv <- 1 - pf(F_curv, df1 = 1, df2 = 16)
p_curv

# (c) Regression model using replicate I only -------------------------------

# Use full 2^3 model on factorial I to get coefficients
mod10_reg <- lm(y ~ A * B * C, data = dat10_fact)
summary(mod10_reg)

coef10_reg <- coef(mod10_reg)
coef10_reg  # compare to coef(mod65) from 6.5

# (d) Residuals (note: saturated for factorial part) ------------------------

resid10 <- resid(mod10_reg)  # these are zero for saturated model
resid10

# Problem 6.11 – 2^4 factorial with 2 replicates

library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW5/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Data entry -------------------------------------------------------------
# Treatment combinations and responses from the table

A <- c(
  # left block ( (1), a, b, ab, c, ac, bc, abc ) replicate I and II
  -1, -1,  # (1)
  +1, +1,  # a
  -1, -1,  # b
  +1, +1,  # ab
  -1, -1,  # c
  +1, +1,  # ac
  -1, -1,  # bc
  +1, +1,  # abc
  # right block ( d, ad, bd, abd, cd, acd, bcd, abcd ) replicate I and II
  -1, -1,  # d  (A-)
  +1, +1,  # ad (A+)
  -1, -1,  # bd
  +1, +1,  # abd
  -1, -1,  # cd
  +1, +1,  # acd
  -1, -1,  # bcd
  +1, +1   # abcd
)

B <- c(
  -1, -1,  # (1)
  -1, -1,  # a
  +1, +1,  # b
  +1, +1,  # ab
  -1, -1,  # c
  -1, -1,  # ac
  +1, +1,  # bc
  +1, +1,  # abc
  -1, -1,  # d
  -1, -1,  # ad
  +1, +1,  # bd
  +1, +1,  # abd
  -1, -1,  # cd
  -1, -1,  # acd
  +1, +1,  # bcd
  +1, +1   # abcd
)

C <- c(
  -1, -1,  # (1)
  -1, -1,  # a
  -1, -1,  # b
  -1, -1,  # ab
  +1, +1,  # c
  +1, +1,  # ac
  +1, +1,  # bc
  +1, +1,  # abc
  -1, -1,  # d
  -1, -1,  # ad
  -1, -1,  # bd
  -1, -1,  # abd
  +1, +1,  # cd
  +1, +1,  # acd
  +1, +1,  # bcd
  +1, +1   # abcd
)

D <- c(
  -1, -1,  # (1)
  -1, -1,  # a
  -1, -1,  # b
  -1, -1,  # ab
  -1, -1,  # c
  -1, -1,  # ac
  -1, -1,  # bc
  -1, -1,  # abc
  +1, +1,  # d
  +1, +1,  # ad
  +1, +1,  # bd
  +1, +1,  # abd
  +1, +1,  # cd
  +1, +1,  # acd
  +1, +1,  # bcd
  +1, +1   # abcd
)

y <- c(
  90, 93,   # (1)
  74, 78,   # a
  81, 85,   # b
  83, 80,   # ab
  77, 78,   # c
  81, 80,   # ac
  88, 82,   # bc
  73, 70,   # abc
  98, 95,   # d
  72, 76,   # ad
  87, 83,   # bd
  85, 86,   # abd
  99, 90,   # cd
  79, 75,   # acd
  87, 84,   # bcd
  80, 80    # abcd
)

dat11 <- data.frame(A = A, B = B, C = C, D = D, y = y)
str(dat11)
dat11 %>% head()

# 2. Fit full 2^4 model -----------------------------------------------------

mod11 <- lm(y ~ A * B * C * D, data = dat11)
summary(mod11)

anova(mod11)  # for part (b)

# 3. Factor effect estimates (part a) ---------------------------------------

coef11 <- coef(mod11)
effects11 <- 2 * coef11[-1]  # drop intercept
effects11

# 4. Regression model (part c) ----------------------------------------------

coef11  # will be used to write regression equation in coded units

# 5. Residual analysis (part d) ---------------------------------------------

resid11 <- resid(mod11)

p_resid_11 <- ggplot(dat11, aes(x = fitted(mod11), y = resid11)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Fitted yield", y = "Residuals",
       title = "6.11 Residuals vs Fitted")

ggsave(file.path(out_dir, "6.11_resid_vs_fitted.png"),
       p_resid_11, width = 5, height = 4, dpi = 300)

p_qq_11 <- ggplot(dat11, aes(sample = resid11)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "6.11 Normal Q-Q Plot of Residuals")

ggsave(file.path(out_dir, "6.11_qq.png"),
       p_qq_11, width = 5, height = 4, dpi = 300)

shapiro.test(resid11)

# 6. Cube plots for part (e) -----------------------------------------------

# Means for A-B-C cube (averaging over D)
means_ABC <- dat11 %>%
  group_by(A, B, C) %>%
  summarise(mean_y = mean(y), .groups = "drop")
means_ABC

# Means for A-B-D cube (averaging over C)
means_ABD <- dat11 %>%
  group_by(A, B, D) %>%
  summarise(mean_y = mean(y), .groups = "drop")
means_ABD

# (You can sketch cube plots by hand from these tables; if you really want
# R-drawn cubes we can add them, but for HW usually the numeric means suffice.)

# Problem 6.21 – normal probability plot of effects, tentative model

library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW5/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

effects_21 <- data.frame(
  term = c("A","B","C","D",
           "AB","AC","AD","BC","BD","CD",
           "ABC","ABD","ACD","BCD","ABCD"),
  effect = c(76.95, -67.52, -7.84, -18.73,
             -51.32, 11.69, 9.78, 20.78, 14.74, 1.27,
             -2.82, -6.50, 10.20, -7.98, -6.25)
)

str(effects_21)
effects_21

# (a) Normal probability plot of effects -- with labels ---------------------

# Compute QQ coordinates first
qq_dat <- qqnorm(effects_21$effect, plot.it = FALSE)
qq_df <- data.frame(
  x = qq_dat$x,
  y = qq_dat$y,
  term = effects_21$term
)

p_norm <- ggplot(qq_df, aes(x = x, y = y, label = term)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_text(vjust = -0.5) +
  labs(title = "6.21 Normal Probability Plot of Effects",
       x = "Normal score",
       y = "Effect estimate")

ggsave(file.path(out_dir, "6.21_normal_plot_effects.png"),
       p_norm, width = 5, height = 4, dpi = 300)

# (b) Tentative model: pick "large" effects --------------------------------

# Simple rule: effects with |effect| > 15 look clearly off the line
effects_21 %>%
  mutate(abs_eff = abs(effect)) %>%
  arrange(desc(abs_eff))

