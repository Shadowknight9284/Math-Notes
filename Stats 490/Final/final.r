
options(digits = 6)

section <- function(title) {
  cat("\n", paste(rep("=", 72), collapse = ""), "\n", sep = "")
  cat(title, "\n")
  cat(paste(rep("=", 72), collapse = ""), "\n", sep = "")
}

signs_for_run <- function(run, factors) {
  if (run == "(1)") {
    high <- character(0)
  } else {
    high <- toupper(strsplit(run, "")[[1]])
  }
  signs <- ifelse(factors %in% high, 1, -1)
  names(signs) <- factors
  signs
}

word_sign_from_run <- function(run, word, factors) {
  signs <- signs_for_run(run, factors)
  prod(signs[word])
}

multiply_words <- function(w1, w2) {
  letters <- strsplit(paste0(w1, w2), "")[[1]]
  counts <- table(letters)
  remaining <- names(counts[counts %% 2 == 1])
  paste(sort(remaining), collapse = "")
}

design_resolution <- function(words) {
  min(nchar(words))
}

## -------------------------------------------------------------------------
## Question 1
## -------------------------------------------------------------------------

section("Question 1: 2^4 factorial effect contributions")

effects_q1 <- c(
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

k_q1 <- 4
total_ss_q1 <- 58665

ss_q1 <- 2^(k_q1 - 2) * effects_q1^2
pct_q1 <- 100 * ss_q1 / total_ss_q1
beta_q1 <- effects_q1 / 2

q1_table <- data.frame(
  Effect = names(effects_q1),
  Effect_Estimate = as.numeric(effects_q1),
  SS = as.numeric(ss_q1),
  Percent_Total_SS = as.numeric(pct_q1),
  Regression_Coefficient = as.numeric(beta_q1)
)

q1_table <- q1_table[order(-q1_table$Percent_Total_SS), ]
print(q1_table, row.names = FALSE)

q1_top <- head(q1_table, 3)
cat("\nTop effects:\n")
print(q1_top, row.names = FALSE)
cat("\nCombined percent for top three effects:",
    sum(q1_top$Percent_Total_SS), "\n")

## -------------------------------------------------------------------------
## Question 2
## -------------------------------------------------------------------------

section("Question 2: Two-factor ANOVA with and without blocks")

a <- 2
b <- 3
r <- 3

ss_a <- 50
ss_b <- 80
ss_ab <- 30
ss_total <- 172

df_a <- a - 1
df_b <- b - 1
df_ab <- df_a * df_b
df_total <- a * b * r - 1
df_error <- df_total - df_a - df_b - df_ab

ss_error <- ss_total - ss_a - ss_b - ss_ab

ms_a <- ss_a / df_a
ms_b <- ss_b / df_b
ms_ab <- ss_ab / df_ab
ms_error <- ss_error / df_error

f_a <- ms_a / ms_error
f_b <- ms_b / ms_error
f_ab <- ms_ab / ms_error

anova_q2_unblocked <- data.frame(
  Source = c("A", "B", "AB", "Error", "Total"),
  df = c(df_a, df_b, df_ab, df_error, df_total),
  SS = c(ss_a, ss_b, ss_ab, ss_error, ss_total),
  MS = c(ms_a, ms_b, ms_ab, ms_error, NA),
  F = c(f_a, f_b, f_ab, NA, NA)
)

cat("\nQ2(a): Unblocked ANOVA table\n")
print(anova_q2_unblocked, row.names = FALSE)

block_totals <- c(10, 12, 14)
grand_total <- sum(block_totals)
n_total <- a * b * r
obs_per_block <- a * b

ss_blocks <- sum(block_totals^2 / obs_per_block) - grand_total^2 / n_total
df_blocks <- r - 1
ms_blocks <- ss_blocks / df_blocks

ss_error_blocked <- ss_error - ss_blocks
df_error_blocked <- df_error - df_blocks
ms_error_blocked <- ss_error_blocked / df_error_blocked

f_blocks <- ms_blocks / ms_error_blocked
f_a_blocked <- ms_a / ms_error_blocked
f_b_blocked <- ms_b / ms_error_blocked
f_ab_blocked <- ms_ab / ms_error_blocked

s_hat_q2 <- sqrt(ms_error_blocked)

anova_q2_blocked <- data.frame(
  Source = c("Blocks", "A", "B", "AB", "Error", "Total"),
  df = c(df_blocks, df_a, df_b, df_ab, df_error_blocked, df_total),
  SS = c(ss_blocks, ss_a, ss_b, ss_ab, ss_error_blocked, ss_total),
  MS = c(ms_blocks, ms_a, ms_b, ms_ab, ms_error_blocked, NA),
  F = c(f_blocks, f_a_blocked, f_b_blocked, f_ab_blocked, NA, NA)
)

cat("\nQ2(b): Blocked ANOVA table\n")
print(anova_q2_blocked, row.names = FALSE)

cat("\nEstimated standard deviation:", s_hat_q2, "\n")

p_values_q2 <- data.frame(
  Effect = c("Blocks", "A", "B", "AB"),
  F_value = c(f_blocks, f_a_blocked, f_b_blocked, f_ab_blocked),
  df1 = c(df_blocks, df_a, df_b, df_ab),
  df2 = rep(df_error_blocked, 4),
  p_value = c(
    pf(f_blocks, df_blocks, df_error_blocked, lower.tail = FALSE),
    pf(f_a_blocked, df_a, df_error_blocked, lower.tail = FALSE),
    pf(f_b_blocked, df_b, df_error_blocked, lower.tail = FALSE),
    pf(f_ab_blocked, df_ab, df_error_blocked, lower.tail = FALSE)
  )
)

cat("\nQ2(c): p-values using blocked error\n")
print(p_values_q2, row.names = FALSE)

## -------------------------------------------------------------------------
## Question 3
## -------------------------------------------------------------------------

section("Question 3: Standard error formula")

cat("SE(effect) = sqrt(4 * MSE / (n * 2^k))\n")
cat("Equivalent: SE(effect) = sqrt(MSE / (n * 2^(k - 2)))\n")
cat("Main effects and interactions have the same SE in a balanced 2^k design.\n")

## -------------------------------------------------------------------------
## Question 4
## -------------------------------------------------------------------------

section("Question 4: Center points, pure error, and curvature")

ss_terms_q4 <- c(
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

factorial_mean <- 70.0625
center_points <- c(73, 75, 66, 99)

n_factorial <- 16
n_center <- length(center_points)
center_mean <- mean(center_points)

ss_factorial <- sum(ss_terms_q4)
df_factorial <- length(ss_terms_q4)
ms_factorial <- ss_factorial / df_factorial

ss_pure_error <- sum((center_points - center_mean)^2)
df_pure_error <- n_center - 1
ms_pure_error <- ss_pure_error / df_pure_error

ss_curvature <- (n_factorial * n_center) /
  (n_factorial + n_center) * (factorial_mean - center_mean)^2
df_curvature <- 1
ms_curvature <- ss_curvature

f_curvature <- ms_curvature / ms_pure_error
p_curvature <- pf(
  f_curvature,
  df1 = df_curvature,
  df2 = df_pure_error,
  lower.tail = FALSE
)

ss_total_q4 <- ss_factorial + ss_curvature + ss_pure_error
df_total_q4 <- df_factorial + df_curvature + df_pure_error

anova_q4 <- data.frame(
  Source = c("Factorial model", "Curvature", "Pure Error", "Total"),
  df = c(df_factorial, df_curvature, df_pure_error, df_total_q4),
  SS = c(ss_factorial, ss_curvature, ss_pure_error, ss_total_q4),
  MS = c(ms_factorial, ms_curvature, ms_pure_error, NA),
  F = c(NA, f_curvature, NA, NA)
)

print(anova_q4, row.names = FALSE)

cat("\nFactorial mean:", factorial_mean, "\n")
cat("Center mean:", center_mean, "\n")
cat("Curvature p-value:", p_curvature, "\n")

## -------------------------------------------------------------------------
## Question 5
## -------------------------------------------------------------------------

section("Question 5: Blocking in a 2^6 factorial experiment")

factors_q5 <- c("A", "B", "C", "D", "E", "F")
runs_q5 <- c("(1)", "b", "abd", "abcdef", "bdf")

gen1_q5 <- c("A", "B", "C", "D")
gen2_q5 <- c("C", "D", "E", "F")

q5_results <- data.frame(
  Run = runs_q5,
  ABCD = sapply(runs_q5, word_sign_from_run, word = gen1_q5,
                factors = factors_q5),
  CDEF = sapply(runs_q5, word_sign_from_run, word = gen2_q5,
                factors = factors_q5)
)
q5_results$Principal_Block <- q5_results$ABCD == 1 & q5_results$CDEF == 1

print(q5_results, row.names = FALSE)

defining_words_q5 <- c("ABCD", "CDEF", "ABEF")
cat("\nBlock defining relation: I =",
    paste(defining_words_q5, collapse = " = "), "\n")
cat("Block alias relation: Block =",
    paste(defining_words_q5, collapse = " = "), "\n")
cat("Resolution if used as a 2^(6-2) fraction:",
    design_resolution(defining_words_q5), "\n")

## -------------------------------------------------------------------------
## Question 6
## -------------------------------------------------------------------------

section("Question 6: Conceptual items")

cat("No numerical computation needed for Q6.\n")
cat("Key checks: D-optimal design, resolution III/IV/V, 2^(5-1) Resolution V,\n")
cat("and location versus dispersion.\n")

## -------------------------------------------------------------------------
## Question 7
## -------------------------------------------------------------------------

section("Question 7: True/false checks")

q7_checks <- data.frame(
  Statement = 1:4,
  Answer = c("False", "False as written", "False", "False"),
  Key_Reason = c(
    "A 2^(5-2) design has alias chains of size 2^2 = 4.",
    "Half-normal plots use ordered absolute effects, not raw signed effects.",
    "A two-block contrast can be computed from block averages.",
    "Four blocks require two independent generators and confound 3 effects."
  )
)
print(q7_checks, row.names = FALSE)

## -------------------------------------------------------------------------
## Question 8
## -------------------------------------------------------------------------

section("Question 8: Fractional factorial defining relations")

designs_q8 <- list(
  A = c("ABCP", "BCDQ"),
  B = c("ABCEP", "BCDFQ"),
  C = c("ABCDEP", "ABDFQ")
)

q8_results <- data.frame(
  Design = character(0),
  Fraction = character(0),
  Defining_Relation = character(0),
  Resolution = integer(0)
)

for (d in names(designs_q8)) {
  word1 <- designs_q8[[d]][1]
  word2 <- designs_q8[[d]][2]
  word3 <- multiply_words(word1, word2)
  words <- c(word1, word2, word3)
  q8_results <- rbind(
    q8_results,
    data.frame(
      Design = d,
      Fraction = "1/4",
      Defining_Relation = paste("I =", paste(words, collapse = " = ")),
      Resolution = design_resolution(words)
    )
  )
}

print(q8_results, row.names = FALSE)

## -------------------------------------------------------------------------
## Question 9
## -------------------------------------------------------------------------

section("Question 9: Same block as acd when ABCDE is confounded")

factors_q9 <- c("A", "B", "C", "D", "E")
target_q9 <- "acd"
candidates_q9 <- c("(1)", "a", "ad", "bcd", "be", "abe")
word_q9 <- factors_q9

target_sign_q9 <- word_sign_from_run(target_q9, word_q9, factors_q9)

q9_results <- data.frame(
  Run = candidates_q9,
  ABCDE_sign = sapply(candidates_q9, word_sign_from_run,
                      word = word_q9, factors = factors_q9)
)
q9_results$Same_Block_As_acd <- q9_results$ABCDE_sign == target_sign_q9

cat("Target run:", target_q9, "\n")
cat("ABCDE sign for target:", target_sign_q9, "\n\n")
print(q9_results, row.names = FALSE)



