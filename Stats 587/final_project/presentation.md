# NFL Bayesian MCMC – Presentation Outline

## 1. Introduction (1–2 slides)
- **Problem motivation**
  - NFL outcomes are noisy; we want to estimate team strength and home-field advantage and use them for playoff predictions.
- **Goals of the project**
  - Build and compare three Bayesian models of team ability.
  - Quantify home-field advantage.
  - Forecast 2024–25 Super Wild Card games with uncertainty.
- **Data source**
  - 2024 NFL regular season play-by-play / game-level dataset (272 games, 32 teams, weeks 1–18).

---

## 2. Data and Exploratory Analysis (2–3 slides)
- **Dataset description**
  - Variables used: game_id, week, home_team, away_team, point_diff (home − away).
  - Sanity checks: 272 games, 32 teams, weeks 1–18, no missing critical fields.
- **Distribution of point differentials**
  - Histogram: mean ≈ 1.9, SD ≈ 14.
  - Q–Q plot vs normal: central region near-line, heavier tails in extremes.
- **Home field advantage and outcomes**
  - Home win rate ≈ 53%.
  - HFA by week: mean > 0 with week-to-week noise.
  - Outcomes pie chart: split of home vs away wins.
- **Team performance validation**
  - Team win percentages plot.
  - Average margin vs win percentage (r ≈ 0.91): validates margin as ability proxy.

Explain: this EDA motivates a normal likelihood with team effects and a modest home-field advantage.

---

## 3. Modeling Framework (1–2 slides)
- **Three nested models**
  1. Static model: θ_i constant over season.
  2. Independent time-varying model: α_{i,t} independent across t.
  3. Hierarchical AR(1): α_{i,t} with AR(1) dependence and team means μ_i.
- **Likelihood**
  - For game g with home i, away j in week t:  
    \( y_g \sim N(\text{team effect} + \beta, \sigma^2) \).
- **Priors (high level)**
  - Weakly informative normals on abilities and β.
  - AR(1) hyperpriors: μ_i, φ ~ Beta, σ_team half-normal.

Explain each model’s purpose: static as baseline, independent as over-flexible check, AR(1) as final model with temporal structure.

---

## 4. Static Model (θ) – Results (2 slides)
- **Model definition**
  - \( y_g = \theta_i - \theta_j + \beta + \varepsilon_g \), ε ~ N(0, σ²).
- **MCMC implementation**
  - Metropolis–Hastings with tuned step sizes.
  - Convergence diagnostics: trace plots for θ_{ARI}, β, σ.
- **Key posterior estimates**
  - β ≈ 1.84 (95% CI ~ [0.4, 3.2]).
  - σ ≈ 12.0.
- **Team rankings**
  - Bar chart of θ_i (DET, BAL, BUF, GB, PHI at top; CAR, CLE, NE at bottom).
  - Interpretation: season-long strength ranking.

Explain why this is a good baseline and how σ < raw SD shows variance explained by θ.

---

## 5. Independent Time-Varying Model (α) – Results (2 slides)
- **Model definition**
  - \( y_g = \alpha_{i,t} - \alpha_{j,t} + \beta + \varepsilon_g \), α_{i,t} ~ N(0, 2²) i.i.d.
- **MCMC diagnostics**
  - Trace plots for β, σ, and α_{ARI,week1}.
  - Posterior histograms for β, σ.
- **Parameter behavior**
  - β ≈ 2.0, similar to static model.
  - σ ≈ 14.1, closer to raw SD.
- **Week 18 rankings**
  - α_{i,18} bar chart: noisy ranking dominated by single games.

Explain: this model is intentionally over-flexible; illustrates the danger of not pooling over time and motivates AR(1).

---

## 6. Hierarchical AR(1) Model – Results (3 slides)
- **Model definition**
  - AR(1): \( \alpha_{i,t} \mid \alpha_{i,t-1}, \mu_i, \phi, \sigma_{\text{team}} \).
  - Team means μ_i give long-run ability; φ controls persistence.
- **Computation**
  - Optimized vectorized MH with team-block α updates.
  - Runtime vs naive implementation.
- **Posterior summaries**
  - β ≈ 1.97, σ ≈ 14.6.
  - φ ≈ 0.74 (moderate persistence).
  - σ_team ≈ 0.27 (smooth evolution).
  - Example μ_i posteriors for ARI and WAS.
- **Diagnostics**
  - Trace plots for β, φ, σ_team, σ.
  - Comment on mixing and how burn-in was chosen.
- **Rankings**
  - Week 18 α bar chart: HOU, SF, CHI, BAL, PHI, NYG, GB at the top; MIN, CAR, CLE at bottom.
  - Interpret shift relative to static θ (captures recent form).

---

## 7. Model Comparison (2 slides)
- **Shared parameters β and σ**
  - Table and overlapping histograms:
    - All models agree on β ≈ 2.
    - σ: static (≈12) < independent (≈14) < AR(1) (≈14.6).
- **Ranking comparison**
  - Top-10 table across models.
  - Side-by-side top-16 bar charts (static vs independent vs AR(1)).
- **Model complexity**
  - Parameter counts: 34 vs 578 vs 612.
- **Takeaways**
  - Static best explains variance but no time dynamics.
  - Independent is over-flexible and noisy.
  - AR(1) balances flexibility, interpretability, and prediction.

---

## 8. Posterior Predictive Checks & Playoff Predictions (2–3 slides)
- **Posterior predictive checks**
  - (If added) Compare simulated y_rep distribution vs observed histogram / QQ plot.
  - Discuss calibration of tails and center.
- **Playoff predictions**
  - Table of win probabilities and predicted spreads for Super Wild Card games.
  - Highlight strongest favorite (HOU vs CLE) and closest matchups (DAL–GB, DET–LA).
  - Show predictive density plots for 2–3 games (e.g., BUF–PIT, KC–MIA, HOU–CLE).
- **Interpretation**
  - Emphasize model uncertainty: wide point-spread intervals, but meaningful win probabilities.

---

## 9. Limitations and Future Work (1 slide)
- Data limited to one season; results may not generalize across eras.
- Normal likelihood ignores discrete scoring and heavy tails of blowouts.
- No explicit covariates (injuries, rest days, weather).
- Possible extensions:
  - Hierarchical priors across seasons.
  - Non–Gaussian error models (e.g., t-distribution).
  - Incorporate betting spreads or EPA-based features.

---

## 10. Conclusion (1 slide)
- Recap:
  - EDA validated modeling assumptions (home-field advantage, approximate normality).
  - Three Bayesian models fit and compared; AR(1) chosen as final.
  - Generated realistic playoff predictions with quantified uncertainty.
- Final message:
  - Bayesian hierarchical time-series modeling gives interpretable, flexible estimates of team strength and uncertainty that align well with football intuition.

