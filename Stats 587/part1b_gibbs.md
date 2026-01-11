# Part (b): Gibbs Sampler Algorithm

## Overview

The Gibbs sampler is an MCMC algorithm that alternately samples from the full conditional distributions derived in Part (a). For this hierarchical model, we iteratively sample from:

- $\theta_i \mid \beta, Y_i \sim \Gamma(Y_i + 1, \, t_i + \beta)$
- $\beta \mid \boldsymbol{\theta} \sim \Gamma\left(\frac{51}{5}, \, 1 + \sum_{i=1}^{10} \theta_i\right)$

---

## Gibbs Sampling Algorithm

### Algorithm Steps

**Initialization:**
1. Set $\beta^{(0)}$ (e.g., $\beta^{(0)} = 1$)
2. Initialize $\theta_i^{(0)}$ for $i = 1, \ldots, 10$ (e.g., $\theta_i^{(0)} = Y_i / t_i$)

**For each iteration $m = 1, 2, \ldots, M$:**

**Step 1: Update $\theta_i$ for each pump**

For $i = 1, 2, \ldots, 10$:
$$\theta_i^{(m)} \sim \Gamma\left(Y_i + 1, \, t_i + \beta^{(m-1)}\right)$$

**Step 2: Update $\beta$**

$$\beta^{(m)} \sim \Gamma\left(\frac{51}{5}, \, 1 + \sum_{i=1}^{10} \theta_i^{(m)}\right)$$

---

## Pseudocode

```
Algorithm: Gibbs Sampler

Input: Y, t, M (number of iterations), B (burn-in)

Initialize:
  beta[0] = 1.0
  theta[0, :] = Y / t

For m = 1 to M:
    For i = 1 to 10:
        shape = Y[i] + 1
        rate = t[i] + beta[m-1]
        theta[m, i] ~ Gamma(shape, rate)
    
    shape_beta = 51/5
    rate_beta = 1 + sum(theta[m, :])
    beta[m] ~ Gamma(shape_beta, rate_beta)

Return: theta[B+1:M], beta[B+1:M]
```

---

## Convergence Recommendations

- **Burn-in:** Remove first 1,000 iterations
- **Total iterations:** Run 10,000-50,000 iterations
- **Check trace plots** for convergence and mixing
- **Plot autocorrelation** to assess correlation between samples
