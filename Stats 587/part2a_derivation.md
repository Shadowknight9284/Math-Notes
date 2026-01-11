# Part (a): Full Conditional Distributions

## Given Model

- $Y \mid \lambda, \beta \sim \text{Poisson}(\lambda)$
- $\lambda \mid \beta \sim \text{Gamma}(2, \beta)$
- $\beta \sim \text{Gamma}(a, b)$ where $a, b$ are fixed constants

---

## Derivation of Full Conditional for $\lambda \mid \beta, Y$

Using the proportionality trick:

$$p(\lambda \mid \beta, Y) \propto p(Y \mid \lambda) \cdot p(\lambda \mid \beta)$$

$$\propto \left[ e^{-\lambda} \cdot \lambda^Y \right] \cdot \left[ \beta^2 \cdot \lambda^{2-1} \cdot e^{-\beta\lambda} \right]$$

$$\propto e^{-\lambda} \cdot \lambda^Y \cdot \lambda \cdot e^{-\beta\lambda}$$

$$\propto \lambda^{Y+1} \cdot e^{-(1+\beta)\lambda}$$

This is the kernel of a Gamma distribution.

$$\boxed{\lambda \mid \beta, Y \sim \text{Gamma}(Y + 2, \, 1 + \beta)}$$

---

## Derivation of Full Conditional for $\beta \mid \lambda, Y$

Using the proportionality trick:

$$p(\beta \mid \lambda, Y) \propto p(\lambda \mid \beta) \cdot p(\beta)$$

Note: $Y$ does not depend on $\beta$ directly given $\lambda$.

$$\propto \left[ \beta^2 \cdot \lambda \cdot e^{-\beta\lambda} \right] \cdot \left[ b^a \cdot \beta^{a-1} \cdot e^{-b\beta} \right]$$

$$\propto \beta^2 \cdot e^{-\beta\lambda} \cdot \beta^{a-1} \cdot e^{-b\beta}$$

$$\propto \beta^{a+1} \cdot e^{-\beta(b+\lambda)}$$

This is the kernel of a Gamma distribution.

$$\boxed{\beta \mid \lambda \sim \text{Gamma}(a + 2, \, b + \lambda)}$$

---

## Summary of Full Conditional Distributions

| Parameter | Full Conditional Distribution |
|-----------|-------------------------------|
| $\lambda$ | $\text{Gamma}(Y + 2, \, 1 + \beta)$ |
| $\beta$ | $\text{Gamma}(a + 2, \, b + \lambda)$ |

**Note:** The Gamma distribution is parameterized as $\text{Gamma}(\text{shape}, \text{rate})$.
