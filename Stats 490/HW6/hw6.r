# Problem 7.5 – 2^4 design in 2 blocks, ABCD confounded
library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW6/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Data: replicate I from 6.11 -------------------------------------------
# Treatment combinations and replicate I responses only

A <- c(
  -1,  1, -1,  1, -1,  1, -1,  1,  # (1), a, b, ab, c, ac, bc, abc
  -1,  1, -1,  1, -1,  1, -1,  1   # d, ad, bd, abd, cd, acd, bcd, abcd
)

B <- c(
  -1, -1,  1,  1, -1, -1,  1,  1,
  -1, -1,  1,  1, -1, -1,  1,  1
)

C <- c(
  -1, -1, -1, -1,  1,  1,  1,  1,
  -1, -1, -1, -1,  1,  1,  1,  1
)

D <- c(
  -1, -1, -1, -1, -1, -1, -1, -1,
   1,  1,  1,  1,  1,  1,  1,  1
)

y <- c(
  90, 74, 81, 83, 77, 81, 88, 73,  # replicate I
  98, 72, 87, 85, 99, 79, 87, 80
)

dat75 <- data.frame(A = A, B = B, C = C, D = D, y = y)

str(dat75)
dat75

# 2. Create blocks with ABCD confounded ------------------------------------
# Compute ABCD = A*B*C*D; use its sign to define 2 blocks
dat75 <- dat75 %>%
  mutate(ABCD = A * B * C * D,
         block = factor(if_else(ABCD == -1, "B1", "B2")))

dat75 %>% arrange(block)

# Quick check: 8 runs per block, mix of treatments in each
table(dat75$block)

# 3. Fit model with blocks and factorial terms -----------------------------

# Full 2^4 factorial model + block (block is a fixed effect)
mod75 <- lm(y ~ block + A * B * C * D, data = dat75)
anova(mod75)        # ANOVA with block
summary(mod75)      # coefficients

# 4. Normal plot of effects (excluding block) ------------------------------

# Extract factorial coefficients only (drop intercept and block)
coef75 <- coef(mod75)
coef_fact <- coef75[!grepl("^\\(Intercept\\)|^block", names(coef75))]
effects75 <- 2 * coef_fact  # effects in coded units

effects_df <- data.frame(
  term = names(effects75),
  effect = as.numeric(effects75)
)

effects_df

# QQ plot of effects to see which are large
qq_dat_75 <- qqnorm(effects_df$effect, plot.it = FALSE)
qq_df_75 <- data.frame(
  x = qq_dat_75$x,
  y = qq_dat_75$y,
  term = effects_df$term
)

p_norm_75 <- ggplot(qq_df_75, aes(x = x, y = y, label = term)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_text(vjust = -0.5) +
  labs(title = "7.5 Normal Plot of Factorial Effects (ABCD Confounded)",
       x = "Normal score",
       y = "Effect estimate")

ggsave(file.path(out_dir, "7.5_normal_plot_effects.png"),
       p_norm_75, width = 5, height = 4, dpi = 300)

# 5. Residual diagnostics ---------------------------------------------------

resid75 <- resid(mod75)

p_resid_75 <- ggplot(dat75, aes(x = fitted(mod75), y = resid75)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Fitted yield", y = "Residuals",
       title = "7.5 Residuals vs Fitted")

ggsave(file.path(out_dir, "7.5_resid_vs_fitted.png"),
       p_resid_75, width = 5, height = 4, dpi = 300)

p_qq_75 <- ggplot(dat75, aes(sample = resid75)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "7.5 Normal Q-Q Plot of Residuals")

ggsave(file.path(out_dir, "7.5_qq.png"),
       p_qq_75, width = 5, height = 4, dpi = 300)

shapiro.test(resid75)


# Problem 7.6 – 2^4 design in 4 blocks, ABD & ABC (and CD) confounded
library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW6/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Data: replicate I from 6.11 -------------------------------------------

A <- c(
  -1,  1, -1,  1, -1,  1, -1,  1,  # (1), a, b, ab, c, ac, bc, abc
  -1,  1, -1,  1, -1,  1, -1,  1   # d, ad, bd, abd, cd, acd, bcd, abcd
)

B <- c(
  -1, -1,  1,  1, -1, -1,  1,  1,
  -1, -1,  1,  1, -1, -1,  1,  1
)

C <- c(
  -1, -1, -1, -1,  1,  1,  1,  1,
  -1, -1, -1, -1,  1,  1,  1,  1
)

D <- c(
  -1, -1, -1, -1, -1, -1, -1, -1,
   1,  1,  1,  1,  1,  1,  1,  1
)

y <- c(
  90, 74, 81, 83, 77, 81, 88, 73,
  98, 72, 87, 85, 99, 79, 87, 80
)

dat76 <- data.frame(A = A, B = B, C = C, D = D, y = y)

# 2. Define blocks so ABD and ABC (and CD) are confounded -------------------

dat76 <- dat76 %>%
  mutate(
    ABD = A * B * D,
    ABC = A * B * C,
    CD  = C * D
  )

# Use (ABD, ABC) signs to define 4 blocks:
# (ABD, ABC) = (-1, -1), (-1, +1), (+1, -1), (+1, +1)
dat76 <- dat76 %>%
  mutate(
    block = case_when(
      ABD == -1 & ABC == -1 ~ "B1",
      ABD == -1 & ABC ==  1 ~ "B2",
      ABD ==  1 & ABC == -1 ~ "B3",
      ABD ==  1 & ABC ==  1 ~ "B4"
    ),
    block = factor(block)
  )

dat76 %>% arrange(block)
table(dat76$block)  # should be 4 runs in each block

# Check confounding: within each block, ABD, ABC, and CD are constant
dat76 %>% group_by(block) %>%
  summarise(
    ABD_vals = paste(unique(ABD), collapse = ","),
    ABC_vals = paste(unique(ABC), collapse = ","),
    CD_vals  = paste(unique(CD), collapse = ","),
    .groups = "drop"
  )

# 3. Fit a model with block and main effects + key interactions ------------
# We cannot estimate ABD, ABC, or CD separately from block.

mod76 <- lm(y ~ block + A + B + C + D + A:B + A:C + B:C + A:D + B:D, data = dat76)
anova(mod76)
summary(mod76)

# 4. Factor effect estimates (for estimable terms) --------------------------

coef76 <- coef(mod76)

# drop intercept and block terms
coef_fact_76 <- coef76[!grepl("^\\(Intercept\\)|^block", names(coef76))]
effects76 <- 2 * coef_fact_76

effects_df_76 <- data.frame(
  term = names(effects76),
  effect = as.numeric(effects76)
)
effects_df_76

# 5. Normal plot of effects -------------------------------------------------

qq_dat_76 <- qqnorm(effects_df_76$effect, plot.it = FALSE)
qq_df_76 <- data.frame(
  x = qq_dat_76$x,
  y = qq_dat_76$y,
  term = effects_df_76$term
)

p_norm_76 <- ggplot(qq_df_76, aes(x = x, y = y, label = term)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_text(vjust = -0.5) +
  labs(title = "7.6 Normal Plot of Effects (ABD, ABC, CD Confounded)",
       x = "Normal score",
       y = "Effect estimate")

ggsave(file.path(out_dir, "7.6_normal_plot_effects.png"),
       p_norm_76, width = 5, height = 4, dpi = 300)

# 6. Residual diagnostics ---------------------------------------------------

resid76 <- resid(mod76)

p_resid_76 <- ggplot(dat76, aes(x = fitted(mod76), y = resid76)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Fitted yield", y = "Residuals",
       title = "7.6 Residuals vs Fitted")

ggsave(file.path(out_dir, "7.6_resid_vs_fitted.png"),
       p_resid_76, width = 5, height = 4, dpi = 300)

p_qq_76 <- ggplot(dat76, aes(sample = resid76)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "7.6 Normal Q-Q Plot of Residuals")

ggsave(file.path(out_dir, "7.6_qq.png"),
       p_qq_76, width = 5, height = 4, dpi = 300)

shapiro.test(resid76)


# Problem 7.7 – 2^5 design in 2 blocks, ABCDE confounded
library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW6/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Data entry for 6.30 ----------------------------------------------------

# Coded factors: A,B,C,D,E in {-1, +1}
# Order: (1), a, b, ab, c, ac, bc, abc,
#        d, ad, bd, abd, cd, acd, bcd, abcd,
#        e, ae, be, abe, ce, ace, bce, abce,
#        de, ade, bde, abde, cde, acde, bcde, abcde

A <- c(
  -1,  1, -1,  1, -1,  1, -1,  1,
  -1,  1, -1,  1, -1,  1, -1,  1,
  -1,  1, -1,  1, -1,  1, -1,  1,
  -1,  1, -1,  1, -1,  1, -1,  1
)

B <- c(
  -1, -1,  1,  1, -1, -1,  1,  1,
  -1, -1,  1,  1, -1, -1,  1,  1,
  -1, -1,  1,  1, -1, -1,  1,  1,
  -1, -1,  1,  1, -1, -1,  1,  1
)

C <- c(
  -1, -1, -1, -1,  1,  1,  1,  1,
  -1, -1, -1, -1,  1,  1,  1,  1,
  -1, -1, -1, -1,  1,  1,  1,  1,
  -1, -1, -1, -1,  1,  1,  1,  1
)

D <- c(
  -1, -1, -1, -1, -1, -1, -1, -1,
   1,  1,  1,  1,  1,  1,  1,  1,
  -1, -1, -1, -1, -1, -1, -1, -1,
   1,  1,  1,  1,  1,  1,  1,  1
)

E <- c(
  -1, -1, -1, -1, -1, -1, -1, -1,
  -1, -1, -1, -1, -1, -1, -1, -1,
   1,  1,  1,  1,  1,  1,  1,  1,
   1,  1,  1,  1,  1,  1,  1,  1
)

y <- c(
  7,  9, 34, 55, 16, 20, 40, 60,   # no E
  8, 10, 32, 50, 18, 21, 44, 61,   # D = +1, E = -1
  8, 12, 35, 52, 15, 22, 45, 65,   # E = +1, D = -1
  6, 10, 30, 53, 15, 20, 41, 63    # D = +1, E = +1
)

dat77 <- data.frame(A = A, B = B, C = C, D = D, E = E, y = y)
str(dat77)
dat77[1:8, ]

# 2. Define 2 blocks with ABCDE confounded ----------------------------------

dat77 <- dat77 %>%
  mutate(
    ABCDE = A * B * C * D * E,
    block = factor(if_else(ABCDE == -1, "B1", "B2"))
  )

table(dat77$block)

# 3. Fit model with block + main effects + key interactions -----------------
# Full 2^5 is saturated; we use a reduced model guided by 6.30.

mod77 <- lm(y ~ block + A + B + C + D + E +
              A:B + A:C + A:D + A:E +
              B:C + B:D + B:E +
              C:D + C:E +
              D:E,
            data = dat77)

anova(mod77)
summary(mod77)

# 4. Factor effect estimates (for terms in the model) -----------------------

coef77 <- coef(mod77)
coef_fact_77 <- coef77[!grepl("^\\(Intercept\\)|^block", names(coef77))]
effects77 <- 2 * coef_fact_77

effects_df_77 <- data.frame(
  term = names(effects77),
  effect = as.numeric(effects77)
)
effects_df_77

# 5. Normal plot of effects -------------------------------------------------

qq_dat_77 <- qqnorm(effects_df_77$effect, plot.it = FALSE)
qq_df_77 <- data.frame(
  x = qq_dat_77$x,
  y = qq_dat_77$y,
  term = effects_df_77$term
)

p_norm_77 <- ggplot(qq_df_77, aes(x = x, y = y, label = term)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_text(vjust = -0.5, size = 3) +
  labs(title = "7.7 Normal Plot of Effects (ABCDE Confounded)",
       x = "Normal score",
       y = "Effect estimate")

ggsave(file.path(out_dir, "7.7_normal_plot_effects.png"),
       p_norm_77, width = 6, height = 4, dpi = 300)

# 6. Residual diagnostics ---------------------------------------------------

resid77 <- resid(mod77)

p_resid_77 <- ggplot(dat77, aes(x = fitted(mod77), y = resid77)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Fitted yield", y = "Residuals",
       title = "7.7 Residuals vs Fitted")

ggsave(file.path(out_dir, "7.7_resid_vs_fitted.png"),
       p_resid_77, width = 5, height = 4, dpi = 300)

p_qq_77_res <- ggplot(dat77, aes(sample = resid77)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "7.7 Normal Q-Q Plot of Residuals")

ggsave(file.path(out_dir, "7.7_qq.png"),
       p_qq_77_res, width = 5, height = 4, dpi = 300)

shapiro.test(resid77)


# Problem 8.4 – 2^(5-2) from 6.30 using data from 6.28 (Table P6.6)

library(dplyr)
library(ggplot2)

out_dir <- "Stats 490/HW6/img"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 1. Data from Table P6.6 (Problem 6.28) -----------------------------------
# Coded A,B,C and two replicates of number of orders

A <- c(-1,  1, -1,  1, -1,  1, -1,  1)
B <- c(-1, -1,  1,  1, -1, -1,  1,  1)
C <- c(-1, -1, -1, -1,  1,  1,  1,  1)

y1 <- c(50, 44, 46, 42, 49, 48, 47, 56)  # replicate 1
y2 <- c(54, 42, 48, 43, 46, 45, 48, 54)  # replicate 2

dat84_raw <- data.frame(A = A, B = B, C = C, y1 = y1, y2 = y2)

# Use run means as single response for the 2^(5-2) fraction
dat84 <- dat84_raw %>%
  mutate(y = (y1 + y2) / 2) %>%
  select(A, B, C, y)

dat84

# 2. Define 2^(5-2) structure: generators D = AB, E = AC --------------------

dat84 <- dat84 %>%
  mutate(
    D = A * B,  # generator
    E = A * C   # generator
  )

dat84

# Defining relation and alias structure (symbolic)
# I = ABD (since D = AB)
# I = ACE (since E = AC)
# Multiply: (ABD)(ACE) = A^2 B C D E = BCDE, so I = ABD = ACE = BCDE.

# 3. Fit factorial model in A,B,C,D,E (on fraction) -------------------------

# Full 2^5 is not estimable in 8 runs; we fit up to two-factor interactions
mod84 <- lm(y ~ A + B + C + D + E +
              A:B + A:C + A:D + A:E +
              B:C + B:D + B:E +
              C:D + C:E +
              D:E,
            data = dat84)

summary(mod84)
anova(mod84)

# 4. Effect estimates from this fraction -----------------------------------

coef84 <- coef(mod84)
coef_fact_84 <- coef84[!grepl("^\\(Intercept\\)", names(coef84))]
effects84 <- 2 * coef_fact_84  # effects corresponding to columns

effects_df_84 <- data.frame(
  term = names(effects84),
  effect = as.numeric(effects84)
)
effects_df_84

# 5. Normal plot of effects -------------------------------------------------

qq_dat_84 <- qqnorm(effects_df_84$effect, plot.it = FALSE)
qq_df_84 <- data.frame(
  x = qq_dat_84$x,
  y = qq_dat_84$y,
  term = effects_df_84$term
)

p_norm_84 <- ggplot(qq_df_84, aes(x = x, y = y, label = term)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  geom_text(vjust = -0.5, size = 3) +
  labs(title = "8.4 Normal Plot of Effects (2^(5-2) fraction)",
       x = "Normal score",
       y = "Effect estimate")

ggsave(file.path(out_dir, "8.4_normal_plot_effects.png"),
       p_norm_84, width = 6, height = 4, dpi = 300)

