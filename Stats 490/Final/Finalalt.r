# Question 1

effects <- c(
  A = 76.95,
  B = -67.52,
  C = -7.84,
  D = -18.73,
  AB = -51.32,
  AC = 11.69,
  AD = 9.78,
  BC = 20.78,
  BD = 14.74,
  CD = 1.27,
  ABC = -2.82,
  ABD = -6.50,
  ACD = 10.20,
  BCD = -7.98,
  ABCD = -6.25
)

k <- 4
total_SS <- 58665

SS <- 2^(k - 2) * effects^2
pct_SS <- 100 * SS / total_SS
betas <- effects / 2

effect_table <- data.frame(
  Effect = names(effects),
  Effect_Estimate = as.numeric(effects),
  SS = as.numeric(SS),
  Percent_Total_SS = as.numeric(pct_SS),
  Regression_Coefficient = as.numeric(betas)
)

effect_table_sorted <- effect_table[order(-effect_table$Percent_Total_SS), ]
print(effect_table_sorted, row.names = FALSE)

top_effects <- head(effect_table_sorted, 3)
sum(top_effects$Percent_Total_SS)


# Question 2

## Q2: Two-factor ANOVA with 3 replications

## Given information
a <- 2          # levels of factor A, because df_A = 1
b <- 3          # levels of factor B, because df_B = 2
r <- 3          # replications

SS_A <- 50
SS_B <- 80
SS_AB <- 30
SS_total <- 172

## Part (a): Unblocked ANOVA table

df_A <- a - 1
df_B <- b - 1
df_AB <- df_A * df_B
df_total <- a * b * r - 1
df_error <- df_total - df_A - df_B - df_AB

SS_error <- SS_total - SS_A - SS_B - SS_AB

MS_A <- SS_A / df_A
MS_B <- SS_B / df_B
MS_AB <- SS_AB / df_AB
MS_error <- SS_error / df_error

F_A <- MS_A / MS_error
F_B <- MS_B / MS_error
F_AB <- MS_AB / MS_error

anova_unblocked <- data.frame(
  Source = c("A", "B", "AB", "Error", "Total"),
  df = c(df_A, df_B, df_AB, df_error, df_total),
  SS = c(SS_A, SS_B, SS_AB, SS_error, SS_total),
  MS = c(MS_A, MS_B, MS_AB, MS_error, NA),
  F = c(F_A, F_B, F_AB, NA, NA)
)

cat("\nQ2(a): Unblocked ANOVA Table\n")
print(anova_unblocked, row.names = FALSE)


## Part (b): Revised ANOVA table if replications are blocks

block_totals <- c(10, 12, 14)

G <- sum(block_totals)
N <- a * b * r
obs_per_block <- a * b

SS_blocks <- sum(block_totals^2 / obs_per_block) - G^2 / N
df_blocks <- r - 1
MS_blocks <- SS_blocks / df_blocks

SS_error_blocked <- SS_error - SS_blocks
df_error_blocked <- df_error - df_blocks
MS_error_blocked <- SS_error_blocked / df_error_blocked

F_blocks <- MS_blocks / MS_error_blocked
F_A_blocked <- MS_A / MS_error_blocked
F_B_blocked <- MS_B / MS_error_blocked
F_AB_blocked <- MS_AB / MS_error_blocked

s_hat <- sqrt(MS_error_blocked)

anova_blocked <- data.frame(
  Source = c("Blocks", "A", "B", "AB", "Error", "Total"),
  df = c(df_blocks, df_A, df_B, df_AB, df_error_blocked, df_total),
  SS = c(SS_blocks, SS_A, SS_B, SS_AB, SS_error_blocked, SS_total),
  MS = c(MS_blocks, MS_A, MS_B, MS_AB, MS_error_blocked, NA),
  F = c(F_blocks, F_A_blocked, F_B_blocked, F_AB_blocked, NA, NA)
)

cat("\nQ2(b): Blocked ANOVA Table\n")
print(anova_blocked, row.names = FALSE)

cat("\nEstimate of standard deviation:\n")
print(s_hat)


## Part (c): Optional p-values for significance comments

p_A <- pf(F_A_blocked, df_A, df_error_blocked, lower.tail = FALSE)
p_B <- pf(F_B_blocked, df_B, df_error_blocked, lower.tail = FALSE)
p_AB <- pf(F_AB_blocked, df_AB, df_error_blocked, lower.tail = FALSE)

p_values <- data.frame(
  Effect = c("A", "B", "AB"),
  F_value = c(F_A_blocked, F_B_blocked, F_AB_blocked),
  df1 = c(df_A, df_B, df_AB),
  df2 = c(df_error_blocked, df_error_blocked, df_error_blocked),
  p_value = c(p_A, p_B, p_AB)
)

cat("\nQ2(c): Significance p-values using blocked error\n")
print(p_values, row.names = FALSE)


## Q4: 2^4 factorial with center points

## Original factorial ANOVA sum of squares from the problem
effect_SS <- c(
  A = 1870.56,
  B = 39.06,
  C = 390.06,
  D = 855.56,
  AB = 0.06,
  AC = 1314.06,
  BC = 22.56,
  AD = 1105.56,
  BD = 0.56,
  CD = 5.06,
  ABC = 14.06,
  ABD = 68.06,
  ACD = 10.56,
  BCD = 27.56,
  ABCD = 7.56
)

## Original intercept from regression output
factorial_mean <- 70.0625

## Center point observations
center_points <- c(73, 75, 66, 99)

## Number of factorial and center observations
n_f <- 16
n_c <- length(center_points)

## Center point mean
center_mean <- mean(center_points)

## Pure error from replicated center points
SS_pure_error <- sum((center_points - center_mean)^2)
df_pure_error <- n_c - 1
MS_pure_error <- SS_pure_error / df_pure_error

## Curvature sum of squares
SS_curvature <- (n_f * n_c) / (n_f + n_c) * (factorial_mean - center_mean)^2
df_curvature <- 1
MS_curvature <- SS_curvature / df_curvature

## Residual/error after adding center points
SS_residual_new <- SS_pure_error + SS_curvature
df_residual_new <- df_pure_error + df_curvature
MS_residual_new <- SS_residual_new / df_residual_new

## Curvature F-test
F_curvature <- MS_curvature / MS_pure_error
p_curvature <- pf(F_curvature, df_curvature, df_pure_error, lower.tail = FALSE)

## Factorial model SS and new total SS
SS_model <- sum(effect_SS)
df_model <- length(effect_SS)

SS_total_new <- SS_model + SS_residual_new
df_total_new <- n_f + n_c - 1

## Modified ANOVA table
anova_modified <- data.frame(
  Source = c(names(effect_SS), "Curvature", "Pure Error", "Residual", "Total"),
  df = c(rep(1, length(effect_SS)), df_curvature, df_pure_error, df_residual_new, df_total_new),
  SS = c(effect_SS, SS_curvature, SS_pure_error, SS_residual_new, SS_total_new),
  MS = c(effect_SS, MS_curvature, MS_pure_error, MS_residual_new, NA),
  F = c(rep(NA, length(effect_SS)), F_curvature, NA, NA, NA)
)

cat("\nQ4 Modified ANOVA Table\n")
print(anova_modified, row.names = FALSE)

cat("\nImportant quantities:\n")
cat("Factorial mean =", factorial_mean, "\n")
cat("Center mean =", center_mean, "\n")
cat("SS curvature =", SS_curvature, "\n")
cat("SS pure error =", SS_pure_error, "\n")
cat("MS pure error =", MS_pure_error, "\n")
cat("F curvature =", F_curvature, "\n")
cat("p-value curvature =", p_curvature, "\n")

## Q5: Blocking in a 2^6 factorial experiment

factors <- c("A", "B", "C", "D", "E", "F")

runs <- c("(1)", "b", "abd", "abcdef", "bdf")

## Block generators
gen1 <- c("A", "B", "C", "D")  # ABCD
gen2 <- c("C", "D", "E", "F")  # CDEF

## Function to get signs for a run
get_signs <- function(run) {
  if (run == "(1)") {
    high <- character(0)
  } else {
    high <- toupper(strsplit(run, "")[[1]])
  }
  
  signs <- ifelse(factors %in% high, 1, -1)
  names(signs) <- factors
  return(signs)
}

## Function to compute sign of a word
word_sign <- function(signs, word) {
  prod(signs[word])
}

results <- data.frame(
  Run = runs,
  ABCD = NA,
  CDEF = NA,
  Principal_Block = NA
)

for (i in seq_along(runs)) {
  signs <- get_signs(runs[i])
  s1 <- word_sign(signs, gen1)
  s2 <- word_sign(signs, gen2)
  
  results$ABCD[i] <- s1
  results$CDEF[i] <- s2
  results$Principal_Block[i] <- (s1 == 1 & s2 == 1)
}

print(results)

## Defining relation and resolution
defining_words <- c("ABCD", "CDEF", "ABEF")
word_lengths <- nchar(defining_words)
resolution <- min(word_lengths)

cat("\nDefining relation: I =", paste(defining_words, collapse = " = "), "\n")
cat("Resolution =", resolution, "\n")

## Q8: Defining relations and resolution

multiply_words <- function(w1, w2) {
  letters <- strsplit(paste0(w1, w2), "")[[1]]
  tab <- table(letters)
  remaining <- names(tab[tab %% 2 == 1])
  paste(sort(remaining), collapse = "")
}

resolution <- function(words) {
  min(nchar(words))
}

designs <- list(
  A = c("ABCP", "BCDQ"),
  B = c("ABCEP", "BCDFQ"),
  C = c("ABCDEP", "ABDFQ")
)

for (d in names(designs)) {
  word1 <- designs[[d]][1]
  word2 <- designs[[d]][2]
  word3 <- multiply_words(word1, word2)
  defining_relation <- c(word1, word2, word3)
  
  cat("\nDesign", d, "\n")
  cat("Defining relation: I =", paste(defining_relation, collapse = " = "), "\n")
  cat("Resolution:", resolution(defining_relation), "\n")
}

## Q9: Same block as acd when ABCDE is confounded with blocks

factors <- c("A", "B", "C", "D", "E")
target <- "acd"
candidates <- c("(1)", "a", "ad", "bcd", "be", "abe")

get_signs <- function(run) {
  if (run == "(1)") {
    high <- character(0)
  } else {
    high <- toupper(strsplit(run, "")[[1]])
  }
  
  signs <- ifelse(factors %in% high, 1, -1)
  names(signs) <- factors
  signs
}

word_sign <- function(run, word = factors) {
  signs <- get_signs(run)
  prod(signs[word])
}

target_sign <- word_sign(target)

results <- data.frame(
  Run = candidates,
  ABCDE_sign = sapply(candidates, word_sign),
  Same_Block_As_acd = sapply(candidates, word_sign) == target_sign
)

cat("Target run:", target, "\n")
cat("ABCDE sign for target:", target_sign, "\n\n")
print(results, row.names = FALSE)
