# NFL Bayesian MCMC Final Project - Complete Checklist

**Project Due:** December 16, 2025 (Report) | December 9, 2025 (Presentation)  
**Current Date:** December 2, 2025  
**Days Remaining:** 7 days to completion

---

## PHASE 1: DATA PREPARATION & EXPLORATION
**Estimated Time: 3-4 hours | Status: [ ] NOT STARTED**

### 1.1 Data Loading & Inspection
- [ ] Load `NFL_DataScrap(2024-2025).csv` using pandas
- [ ] Display shape: should be (272 games, 60+ columns)
- [ ] List all column names and data types
- [ ] Check for missing values (NaN count per column)
- [ ] Display first 5 rows
- [ ] Display summary statistics (describe())

### 1.2 Data Quality Checks
- [ ] Verify all 272 regular season games present (weeks 1-18)
- [ ] Confirm all 32 NFL teams appear in home_team and away_team
- [ ] Check point_diff range: should be mostly -50 to +50
- [ ] Verify no duplicate games (check game_id uniqueness)
- [ ] Verify home_win, away_wins, tie_flag sum to 1 for each row
- [ ] Check all weeks are 1-18, no missing weeks

### 1.3 Extract Game Outcomes
- [ ] Create DataFrame with columns: `game_id`, `week`, `home_team`, `away_team`, `point_diff`
- [ ] Create team list: sorted list of 32 unique teams
- [ ] Create team_to_index mapping: dict {team: index 0-31}
- [ ] Create index_to_team mapping (reverse)
- [ ] Verify total games = 272

### 1.4 Exploratory Visualizations
- [ ] Histogram of point_diff: check distribution
  - Should be roughly centered at 0-3 (home field advantage)
  - Should be approximately normal with SD ≈ 13-14 points
- [ ] Boxplot of point_diff by week: check for week effects
- [ ] Count of wins by team: bar chart of total wins per team
- [ ] Home field advantage: compute mean point_diff for home team across all games
- [ ] Scatter: team record vs. average point_diff (correlation check)

### 1.5 Compute Summary Statistics
- [ ] Mean point differential (should be ≈ 2-3 points for home team)
- [ ] Standard deviation of point differential (should be ≈ 13 points)
- [ ] Min/max point differential
- [ ] Count ties (should be 0 or very few in NFL)
- [ ] Home team win rate (should be ≈ 58%)

### 1.6 Data Validation for MCMC
- [ ] No NaN values in: game_id, week, home_team, away_team, point_diff
- [ ] All teams have ≥ 1 home game and ≥ 1 away game
- [ ] All weeks represented for each team (or document which teams have byes)
- [ ] point_diff is numeric (int or float)

**DELIVERABLE:** Cleaned dataset + summary statistics notebook + one-page data report

---

## PHASE 2: MCMC IMPLEMENTATION - PART A: SETUP & FUNCTIONS
**Estimated Time: 4-5 hours | Status: [ ] NOT STARTED**

### 2.1 Initialize Model Parameters
- [ ] Create parameter initialization function
  - [ ] `alpha` (32 teams × 18 weeks): initialize to 0
  - [ ] `beta_home` (32 teams): initialize to 3 (typical home advantage)
  - [ ] `mu_team` (32 teams): initialize to 0
  - [ ] `mu_home` (scalar): initialize to 3
  - [ ] `phi` (scalar): initialize to 0.9
  - [ ] `sigma` (scalar): initialize to 13
  - [ ] `sigma_team` (scalar): initialize to 2
- [ ] Print all initial values to verify

### 2.2 Likelihood Functions
- [ ] Code `log_likelihood_all_games()` function
  - Input: Y (outcomes), alpha, beta_home, sigma, home_indices, away_indices
  - Compute: mu = alpha[home_idx] - alpha[away_idx] + beta_home[home_idx]
  - Return: sum of log(N(Y | mu, sigma^2))
  - [ ] Test on toy data: verify likelihood increases when parameters get closer to observed Y
- [ ] Code `log_likelihood_single_param()` helper (for single parameter updates)
  - Input: parameter value, all other parameters, game data
  - Return: log-likelihood for just that parameter change

### 2.3 Prior Functions (Individual)
- [ ] Code `log_prior_alpha_onestep()`: AR(1) prior for alpha[i,t]
  - Input: alpha_it, mu_i, phi, alpha_i(t-1), sigma_team
  - Formula: log N(mu_i + phi*(alpha_i(t-1) - mu_i), sigma_team^2)
  - Edge case: t=0, use log N(mu_i, sigma_team^2)
  - [ ] Test: verify prior is high when alpha_it ≈ mu_i + phi*(previous - mu_i)

- [ ] Code `log_prior_beta_home()`: Normal prior on beta_home
  - Input: beta_home[i], mu_home
  - Formula: log N(mu_home, 2^2)
  - [ ] Test: verify prior is symmetric around mu_home

- [ ] Code `log_prior_mu_team()`: Normal prior on mu_team
  - Input: mu_team (32-vector)
  - Formula: sum of log N(0, 10^2)
  - [ ] Test: verify sum is correct

- [ ] Code `log_prior_mu_home()`: Normal prior on mu_home
  - Input: mu_home (scalar)
  - Formula: log N(0, 10^2)

- [ ] Code `log_prior_phi()`: Uniform prior on phi
  - Input: phi (scalar)
  - Formula: 0 if 0 < phi < 1, else -inf
  - [ ] Test: verify returns 0 for phi ∈ (0,1), -inf otherwise

- [ ] Code `log_prior_sigma()`: Inverse-gamma or flat prior
  - Input: sigma
  - Choose: inverse-gamma(1, 1) or flat log(sigma) penalty
  - [ ] Document choice and justification

- [ ] Code `log_prior_sigma_team()`: Same as sigma

### 2.4 Posterior Functions
- [ ] Code `log_posterior_alpha_onestep()`: likelihood + prior for alpha[i,t]
  - Input: alpha_it, all other params, games involving team i at week t
  - Return: log_likelihood_contribution + log_prior_contribution
  - [ ] Test on known data point

- [ ] Code `log_posterior_beta_home()`: likelihood + prior for beta_home[i]
  - Input: beta_home_i, all other params, games with team i at home
  - [ ] Test

- [ ] Code `log_posterior_phi()`: likelihood + prior for phi
  - Note: phi affects alpha evolution, so need to sum over all alpha constraints
  - [ ] Complex function - code carefully

- [ ] Code `log_posterior_sigma()`: likelihood + prior for sigma
  - [ ] Test

### 2.5 Helper Functions
- [ ] Code `get_games_for_team_week()`: filter games to team i week t
  - Return: list of (home_idx, away_idx, outcome) tuples

- [ ] Code `get_home_games_for_team()`: filter to all home games for team i
  - Return: game indices

- [ ] Code `get_away_games_for_team()`: filter to all away games for team i
  - Return: game indices

- [ ] Code `compute_team_ability_mean()`: compute mean ability for each team
  - Input: alpha (all weeks)
  - Output: mean across weeks (should reflect "typical" strength)

**DELIVERABLE:** Tested likelihood and prior functions, standalone Python script

---

## PHASE 2B: MCMC IMPLEMENTATION - PART B: SAMPLERS
**Estimated Time: 5-6 hours | Status: [ ] NOT STARTED**

### 2.6 Random-Walk Metropolis-Hastings (General)
- [ ] Code `metropolis_hastings_step()` function
  - Input: current_value, log_posterior_func, jump_sd, args
  - Propose: new_value ~ N(current_value, jump_sd^2)
  - Compute: log_ratio = log_posterior(new) - log_posterior(current)
  - Accept: with probability min(1, exp(log_ratio))
  - Return: (new_value or current_value, acceptance_indicator)
  - [ ] Test: verify acceptance rate ≈ 0.5 with well-tuned jump_sd

### 2.7 Sample alpha (Team Abilities)
- [ ] Code `sample_alpha_all()` function
  - Loop over all 32 teams × 18 weeks = 576 parameters
  - For each alpha[i,t]:
    - [ ] Call metropolis_hastings_step() with log_posterior_alpha_onestep
    - [ ] Track acceptance (yes/no)
  - Return: updated alpha matrix, acceptance_rate
  - [ ] Test: run for 10 iterations, check acceptance rate 30-60%
  - [ ] Optimization: consider blocking (e.g., sample all weeks for team i together)

### 2.8 Sample beta_home (Home Field Advantage)
- [ ] Code `sample_beta_home_all()` function
  - Loop over all 32 teams
  - For each beta_home[i]:
    - [ ] Call metropolis_hastings_step() with log_posterior_beta_home
    - [ ] Track acceptance
  - Return: updated beta_home vector, acceptance_rate
  - [ ] Test: run for 10 iterations, check acceptance rate

### 2.9 Sample mu_team (Mean Team Abilities)
- [ ] Code `sample_mu_team_all()` function
  - Can use direct sampling if using conjugate prior
  - OR use Metropolis-Hastings
  - [ ] Implement direct sampling (easier)
    - mu_team[i] | alpha depends on alpha[i, :] and prior
    - posterior = N(posterior_mean, posterior_sd)
    - Direct sample from N(posterior_mean, posterior_sd)
  - Return: updated mu_team vector

### 2.10 Sample mu_home (Mean Home Field Advantage)
- [ ] Code `sample_mu_home()` function
  - Condition on all beta_home[i]
  - Use direct sampling if conjugate, else MH
  - [ ] Implement direct sampling
  - Return: updated mu_home scalar

### 2.11 Sample phi (AR(1) Coefficient)
- [ ] Code `sample_phi()` function
  - This is tricky: phi affects likelihood through alpha constraints
  - Recommend: logit transform to ensure bounds (0,1)
  - phi_logit = logit(phi) = log(phi/(1-phi))
  - Propose: phi_logit_new ~ N(phi_logit, jump_sd^2)
  - Transform back: phi_new = expit(phi_logit_new)
  - Accept/reject using MH
  - [ ] Code transformation functions: logit(), expit()
  - [ ] Test: verify phi stays in (0, 1)
  - Return: updated phi scalar, acceptance_indicator

### 2.12 Sample sigma (Residual Standard Deviation)
- [ ] Code `sample_sigma()` function
  - Option 1: MH on log(sigma) to ensure sigma > 0
  - Option 2: Inverse-gamma direct sampling if using that prior
  - [ ] Implement log(sigma) MH
    - Propose: log_sigma_new ~ N(log_sigma_old, jump_sd^2)
    - Compute sigma_new = exp(log_sigma_new)
    - Accept/reject using MH
  - Return: updated sigma scalar, acceptance_indicator

### 2.13 Sample sigma_team (Team Ability SD)
- [ ] Code `sample_sigma_team()` function
  - Similar to sigma (use log-scale MH)
  - Return: updated sigma_team scalar, acceptance_indicator

**DELIVERABLE:** All sampler functions, integration tests

---

## PHASE 2C: MCMC IMPLEMENTATION - PART C: MAIN LOOP
**Estimated Time: 3-4 hours | Status: [ ] NOT STARTED**

### 2.14 MCMC Main Loop (Tuning Phase)
- [ ] Code `run_mcmc_tuning()` function
  - Input: n_tuning_iter = 1000, initial jump_sd dict
  - Loop for each iteration:
    - [ ] Call all 7 sampler functions in order
    - [ ] Track acceptance rates for each parameter
  - Adaptive tuning: after every 100 iterations
    - [ ] If acceptance < 30%, increase jump_sd by 10%
    - [ ] If acceptance > 60%, decrease jump_sd by 10%
    - [ ] Keep jump_sd within reasonable bounds (e.g., 0.01 to 5.0)
  - [ ] Print progress every 100 iterations
  - Return: tuned jump_sd, final parameter state, acceptance history

### 2.15 MCMC Main Loop (Sampling Phase)
- [ ] Code `run_mcmc_sampling()` function
  - Input: n_iter = 2000, current state, tuned jump_sd
  - Loop for each iteration:
    - [ ] Call all 7 sampler functions (no more tuning)
    - [ ] Store samples for all parameters EVERY iteration
    - [ ] Track acceptance rates (for diagnostics)
  - [ ] Print progress every 500 iterations
  - Return: samples dict with keys ['alpha', 'beta_home', 'mu_team', 'mu_home', 'phi', 'sigma', 'sigma_team']
    - Each entry: list of length n_iter (or numpy array)

### 2.16 Main MCMC Runner
- [ ] Code `run_mcmc()` wrapper function
  - Input: game_data, n_tuning=1000, n_sampling=2000
  - [ ] Initialize parameters
  - [ ] Run tuning phase
  - [ ] Reset tuned jump_sd from tuning
  - [ ] Run sampling phase
  - [ ] Save samples to pickle file: `mcmc_samples.pkl`
  - [ ] Return: samples, jump_sd_final, acceptance_rates

### 2.17 Integration Test
- [ ] Run MCMC for 100 iterations on full data
  - [ ] Should complete without errors
  - [ ] Check shapes: alpha (100, 32, 18), beta_home (100, 32), etc.
  - [ ] Check values: alpha ∈ [-10, 10], beta_home ∈ [-2, 8], phi ∈ (0,1), sigma > 0
  - [ ] Print first 10 alpha samples for team 0
  - [ ] Print first 10 beta_home samples
  - Expected runtime: 2-5 minutes for 100 iterations

**DELIVERABLE:** Full MCMC code, tested on small sample

---

## PHASE 3: CONVERGENCE DIAGNOSTICS & TUNING
**Estimated Time: 2-3 hours | Status: [ ] NOT STARTED**

### 3.1 Run Full MCMC (Tuning + Sampling)
- [ ] Execute: `samples, jump_sd, acceptance = run_mcmc(game_data, n_tuning=1000, n_sampling=2000)`
- [ ] Expected runtime: 30-60 minutes (depending on optimization)
- [ ] Monitor: print every 500 iterations to track progress
- [ ] Save intermediate results

### 3.2 Acceptance Rate Analysis
- [ ] Extract acceptance rates for each parameter
- [ ] Compute mean acceptance for each parameter (from tuning phase and sampling phase)
  - [ ] alpha: target ≈ 40-50%
  - [ ] beta_home: target ≈ 40-50%
  - [ ] mu_team: if MH, target ≈ 40-50%
  - [ ] phi: target ≈ 30-60%
  - [ ] sigma: target ≈ 30-60%
- [ ] Create bar chart: acceptance_rate vs. parameter
- [ ] Document: which parameters have poor mixing (accept rate < 20% or > 80%)?

### 3.3 Trace Plots
- [ ] Create trace plots for key parameters:
  - [ ] phi (should show strong mixing, not stuck)
  - [ ] sigma (should show good mixing)
  - [ ] mu_home (should show good mixing)
  - [ ] One representative alpha[0, 0] (team 0, week 0)
  - [ ] One representative beta_home[0] (team 0)
- [ ] Check visual pattern:
  - [ ] No trending or structural patterns
  - [ ] Random walk-like behavior
  - [ ] No stuck values
- [ ] Save trace plots as PNG

### 3.4 Autocorrelation Analysis
- [ ] Code `compute_autocorrelation()` function
  - Input: samples (1D array of iterations)
  - Output: autocorrelation at lags 1, 5, 10, 20, 50
- [ ] Compute ACF for key parameters:
  - [ ] phi: should drop to near 0 by lag 20
  - [ ] sigma: should drop to near 0 by lag 20
  - [ ] mu_home: similar
- [ ] Interpretation: high ACF → high autocorrelation → poor mixing → effective sample size is small

### 3.5 Gelman-Rubin Convergence Diagnostic (Rhat)
- [ ] Run MCMC TWICE with different starting points
  - [ ] Run 1: chain_1 (n_sampling=2000)
  - [ ] Run 2: chain_2 (n_sampling=2000, overdispersed starting point)
  - Overdispersed: e.g., phi_chain2 = 0.5 (instead of 0.9), alpha_chain2 ~ N(5, 1)
- [ ] Code `compute_rhat()` function
  - Input: list of chains (each chain is array of shape [n_iter, n_params])
  - Compute: between-chain variance B
  - Compute: within-chain variance W
  - Return: Rhat = sqrt((n_iter - 1)/n_iter + B/(n_iter*W)) / sqrt(W) for each parameter
- [ ] Interpret: Rhat < 1.1 suggests convergence
- [ ] Create table of Rhat values for all scalar parameters:
  - [ ] phi
  - [ ] sigma
  - [ ] sigma_team
  - [ ] mu_home
  - [ ] Mean of each mu_team[i]
  - [ ] Mean of each beta_home[i]
- [ ] Document: any parameters with Rhat > 1.1 → needs more iterations or model adjustment

### 3.6 Posterior Predictive Checks
- [ ] For 10 random games from 2024 season:
  - [ ] Observed margin Y
  - [ ] Posterior predictive mean and SD
  - [ ] Is observed Y within posterior predictive 95% CI?
  - [ ] Expected: ≈ 95% of observed Y within CI
- [ ] Create scatter: observed vs. predicted mean
  - Should have tight correlation

**DELIVERABLE:** Convergence report with figures (Rhat, trace plots, ACF), diagnose any issues

---

## PHASE 4: POSTERIOR INFERENCE & SUMMARIES
**Estimated Time: 2-3 hours | Status: [ ] NOT STARTED**

### 4.1 Posterior Summaries (Scalar Parameters)
- [ ] Extract posterior samples for: phi, sigma, sigma_team, mu_home
- [ ] For each:
  - [ ] Compute mean, median, SD
  - [ ] Compute 95% credible interval (2.5%, 97.5% quantiles)
  - [ ] Create trace plot + density plot (side by side)
- [ ] Create table:

| Parameter | Mean | Median | SD | 95% CI Lower | 95% CI Upper |
|-----------|------|--------|-----|-----|-----|
| phi | | | | | |
| sigma | | | | | |
| sigma_team | | | | | |
| mu_home | | | | | |

- [ ] Interpretation:
  - phi: How much does team ability persist week-to-week? (should be high ≈ 0.9+)
  - sigma: Residual uncertainty in margin (should be ≈ 13-14 points)
  - mu_home: Average home field advantage (should be ≈ 2-3 points)

### 4.2 Posterior Summaries (Team Abilities)
- [ ] Extract posterior samples for mu_team (32 teams)
- [ ] For each team:
  - [ ] Compute mean, 95% CI
- [ ] Create table: sorted by posterior mean (strongest to weakest)
  - [ ] Verify: does ranking match 2024 NFL standings? (e.g., Chiefs high, Panthers low)
- [ ] Visualization: 
  - [ ] Forest plot: posterior mean ± 95% CI for each team
  - [ ] Rank by posterior mean
  - [ ] Color code: top 8 (playoff teams) vs. others

### 4.3 Posterior Summaries (Home Field Advantage by Team)
- [ ] Extract posterior samples for beta_home (32 teams)
- [ ] For each team:
  - [ ] Compute mean, 95% CI
- [ ] Interpretation: Do some teams have larger home advantage?
- [ ] Visualization:
  - [ ] Bar chart: beta_home[i] for each team, sorted
  - [ ] Overlay: posterior mean ± SD

### 4.4 Team Ability Over Time (alpha)
- [ ] Extract posterior samples for alpha (32 teams × 18 weeks)
- [ ] For selected teams (e.g., top 8):
  - [ ] Compute posterior mean across iterations
  - [ ] Plot: week (x-axis) vs. posterior mean ability (y-axis)
  - [ ] Overlay: 95% credible band (2.5%, 97.5% quantiles by week)
- [ ] Interpretation: How did top teams' strength evolve? Any injuries/slumps visible?

### 4.5 Correlation of Posterior Means
- [ ] Compute posterior mean team ability mu_team[i] for all teams
- [ ] Compute posterior mean home advantage beta_home[i] for all teams
- [ ] Correlation: are strong teams also high home advantage teams? (should be weakly correlated or uncorrelated)
- [ ] Scatter: mu_team vs. beta_home

**DELIVERABLE:** Comprehensive posterior summary tables and visualizations

---

## PHASE 5: PLAYOFF PREDICTIONS
**Estimated Time: 2-3 hours | Status: [ ] NOT STARTED**

### 5.1 Identify Playoff Matchups
- [ ] Load `nfl-playoff-bracket-25-01-27.pdf`
- [ ] Manually identify 6 first-round matchups (as of January 10, 2025):
  - AFC Wild Card 1: ?
  - AFC Wild Card 2: ?
  - AFC Wild Card 3: ?
  - NFC Wild Card 1: ?
  - NFC Wild Card 2: ?
  - NFC Wild Card 3: ?
- [ ] Determine home/away for each
- [ ] Note: Higher seed is (typically) at home

### 5.2 Posterior Predictive Sampling
- [ ] Code `generate_posterior_predictive_sample()` function
  - Input: home_team, away_team, mcmc_samples
  - For s = 1, ..., n_samples:
    - [ ] Draw alpha[s] for both teams (final week 18)
    - [ ] Draw beta_home[s] for home team
    - [ ] Draw sigma[s]
    - [ ] Compute mu[s] = alpha_home[s] - alpha_away[s] + beta_home[s]
    - [ ] Sample Y[s] ~ N(mu[s], sigma[s]^2)
  - [ ] Return: Y samples (length n_samples, e.g., 2000)

### 5.3 Playoff Game Predictions (All 6 Games)
- [ ] For each of the 6 playoff matchups:
  - [ ] Call `generate_posterior_predictive_sample(home, away, samples)`
  - [ ] Compute:
    - [ ] P(home wins) = mean(Y > 0)
    - [ ] P(away wins) = mean(Y < 0)
    - [ ] P(tie) = mean(Y ≈ 0) [usually negligible]
    - [ ] E[Y | data] = mean(Y)
    - [ ] SD[Y | data] = sd(Y)
    - [ ] 95% credible interval for margin: (q_2.5, q_97.5)
    - [ ] Modal prediction: sign(E[Y])
  - [ ] Store in results dict

### 5.4 Predictions Table
- [ ] Create comprehensive table:

| Game | Matchup | Home Team Prob | Away Team Prob | Expected Margin | 95% CI | Modal Winner |
|------|---------|---|---|---|---|---|
| 1 | X vs. Y | | | | | |
| 2 | A vs. B | | | | | |
| ... | | | | | | |

- [ ] Round probabilities to 3 decimals
- [ ] Expected margin to 1 decimal

### 5.5 Visualizations for Predictions
- [ ] For each game: Create side-by-side plot
  - Left: Posterior predictive density (Y samples)
  - Overlay: vertical line at Y=0 (break-even)
  - Shade regions: Y > 0 (home wins), Y < 0 (away wins)
  - Right: Table with predictions (prob, margin, CI)
  
- [ ] Alternatively: Single multi-panel figure with all 6 games

### 5.6 Comparison to Betting Markets (Optional)
- [ ] If available: obtain Vegas spread & over/under for playoff games
- [ ] Compare model predictions to Vegas
  - [ ] Are model probabilities higher/lower than betting line implies?
  - [ ] Any noteworthy disagreements?

**DELIVERABLE:** 6 playoff game predictions with full posterior predictive analysis

---

## PHASE 6: SENSITIVITY ANALYSIS
**Estimated Time: 1-2 hours | Status: [ ] NOT STARTED**

### 6.1 Prior Sensitivity - Phi
- [ ] Rerun MCMC with different phi prior:
  - [ ] Original: phi ~ U(0, 1)
  - [ ] Alternative 1: phi ~ Beta(5, 2) [concentrated at 0.7]
  - [ ] Alternative 2: phi ~ Beta(10, 5) [concentrated at 0.67]
- [ ] Compare:
  - [ ] Posterior mean phi
  - [ ] Playoff predictions (do they change significantly?)

### 6.2 Prior Sensitivity - mu_home
- [ ] Rerun with different mu_home prior:
  - [ ] Original: mu_home ~ N(0, 10^2)
  - [ ] Alternative 1: mu_home ~ N(2, 1^2) [informative at 2 points]
  - [ ] Alternative 2: mu_home ~ N(3, 0.5^2) [very informative at 3 points]
- [ ] Compare:
  - [ ] Posterior mean mu_home
  - [ ] Playoff predictions

### 6.3 Prior Sensitivity - sigma_team
- [ ] Rerun with different sigma_team prior:
  - [ ] Original: sigma_team ~ inv-gamma(1,1)
  - [ ] Alternative: sigma_team ~ inv-gamma(2, 1)
- [ ] Compare results

### 6.4 Sensitivity Report
- [ ] Create table: how much do predictions change?
  - [ ] Game 1: P(home wins) under 3 different priors for phi
  - [ ] Expected margins under 3 different mu_home priors
  - [ ] Max deviation from base predictions
- [ ] Interpretation: are predictions robust or sensitive to priors?

**DELIVERABLE:** Sensitivity analysis report + table

---

## PHASE 7: MODEL VALIDATION & DIAGNOSTICS
**Estimated Time: 1-2 hours | Status: [ ] NOT STARTED**

### 7.1 Posterior Predictive Check (Detailed)
- [ ] For all 272 games in 2024 season:
  - [ ] Generate posterior predictive Y_pred[i] for game i
  - [ ] Observed margin Y_obs[i]
- [ ] Check: proportion of Y_obs in 95% predictive interval
  - [ ] Should be ≈ 95%
  - [ ] If < 90%: model is overconfident
  - [ ] If > 98%: model is underconfident
- [ ] Visualization:
  - [ ] Scatter: Y_obs vs. Y_pred mean
  - [ ] Add 45-degree line
  - [ ] Add ±1 SD band around line

### 7.2 Coverage Analysis
- [ ] Compute empirical coverage: % of observed Y in 68%, 95% PPC intervals
- [ ] Expected: 68% of Y should fall in 68% interval, 95% in 95% interval
- [ ] If coverage is off: may indicate model misspecification

### 7.3 Residual Analysis
- [ ] Compute residuals: residual[i] = Y_obs[i] - Y_pred[i]
- [ ] Visualizations:
  - [ ] Histogram of residuals: should be ~ N(0, 1) after standardization
  - [ ] Q-Q plot: residuals vs. normal distribution
  - [ ] Residuals vs. predicted Y_pred: should have no pattern

### 7.4 Check: Home Field Advantage Over Time
- [ ] Compute average point_diff by week
- [ ] Is home field advantage consistent? Or does it vary?
- [ ] Plot: week (x) vs. avg point_diff (y)
  - Should hover around 2-3 points

**DELIVERABLE:** Model validation report

---

## PHASE 8: FINAL REPORT PREPARATION
**Estimated Time: 4-5 hours | Status: [ ] NOT STARTED**

### 8.1 Report Structure
Create a single PDF report with sections:

**Section 1: Executive Summary (1 page)**
- [ ] What was done: Bayesian hierarchical model for NFL
- [ ] Key findings: team rankings, home field advantage, playoff predictions
- [ ] Confidence level in predictions

**Section 2: Data & Methods (2-3 pages)**
- [ ] Data source: NFL 2024 regular season, 272 games
- [ ] Data quality checks: completeness, distributions
- [ ] Model specification: write out full posterior
  - Include: likelihood, priors, hyperparameters
- [ ] MCMC algorithm: Metropolis-Hastings, 1000 tuning + 2000 sampling iterations
- [ ] Diagnostics: Rhat, acceptance rates, trace plots

**Section 3: Results (4-5 pages)**
- [ ] 3.1 Posterior Parameter Estimates
  - Table: phi, sigma, mu_home with 95% CI
  - Interpretation: what do these mean?
- [ ] 3.2 Team Rankings
  - Forest plot: team abilities with 95% CI
  - Table: top 8, bottom 8 teams
- [ ] 3.3 Home Field Advantage
  - Table: beta_home for each team
  - Summary: mean home advantage across teams
- [ ] 3.4 Team Ability Evolution
  - Spaghetti plot or line plot: alpha[i,t] over weeks for selected teams
  - Any notable trends?

**Section 4: Playoff Predictions (1-2 pages)**
- [ ] Table: 6 matchups with probabilities and margins
- [ ] Individual plots/summaries for each game (or multi-panel figure)
- [ ] Interpretation: which games most uncertain? Why?

**Section 5: Sensitivity Analysis (1 page)**
- [ ] How robust are predictions to prior choices?
- [ ] Table: predictions under different priors

**Section 6: Model Validation (1 page)**
- [ ] Posterior predictive checks
- [ ] Coverage analysis
- [ ] Residual diagnostics

**Section 7: Discussion & Limitations (1-2 pages)**
- [ ] Strengths: hierarchical model captures partial pooling, time-varying abilities, etc.
- [ ] Limitations:
  - [ ] Playoffs ≠ regular season (different intensity, stakes, preparation)
  - [ ] Injuries/roster changes not modeled
  - [ ] No weather, travel, scheduling effects
  - [ ] Normal likelihood assumes symmetric errors (may not hold)
- [ ] Future improvements: incorporate playoff history, injury reports, etc.

**Section 8: Conclusion (0.5 page)**
- [ ] Summary of findings
- [ ] Key takeaway: confident in playoff predictions? Which games uncertain?

### 8.2 Report Writing
- [ ] Write each section in clear, technical language
- [ ] Every figure/table must have caption and explanation in text
- [ ] No orphan results: don't include plot without explanation
- [ ] Cite equations, explain notation
- [ ] Proofread for clarity and accuracy

### 8.3 Figures & Tables
- [ ] All figures should be high-resolution (dpi ≥ 300)
- [ ] All tables formatted nicely (use pandas to_latex() or similar)
- [ ] Consistent font, sizing, colors
- [ ] Figure captions: 1-2 sentences explaining what plot shows
- [ ] Color scheme: consider colorblind-friendly palette

### 8.4 Code Appendix
- [ ] Attach full Python code as appendix or supplementary file
- [ ] Organize code into sections with comments
- [ ] Include: data loading, model definition, MCMC, predictions, diagnostics
- [ ] Ensure code is reproducible (random seed set, paths documented)

**DELIVERABLE:** 8-10 page PDF report with figures, tables, and interpretation

---

## PHASE 9: PRESENTATION PREPARATION
**Estimated Time: 2-3 hours | Status: [ ] NOT STARTED**

### 9.1 Slide Deck (10-15 minutes)
Create presentation slides:

**Slide 1: Title & Overview**
- [ ] Title: NFL Playoff Predictions via Bayesian Hierarchical MCMC
- [ ] Your name, date, class

**Slide 2: Motivation & Problem**
- [ ] Why predict NFL playoff outcomes?
- [ ] Challenges: team strength evolves, home field advantage varies

**Slide 3: Data Overview**
- [ ] 272 games, 32 teams, 2024 regular season
- [ ] Key variables: margin, EPA, yards, etc.
- [ ] Distribution plot: point_diff

**Slide 4: Model Specification (Non-Technical)**
- [ ] Margin of victory explained by: team strengths + home advantage
- [ ] Team strengths evolve week-to-week (AR(1))
- [ ] Hierarchical: teams share info through common priors

**Slide 5: Model Specification (Technical)**
- [ ] Write equation: Y_ijt ~ N(alpha_it - alpha_jt + beta_i^home, sigma^2)
- [ ] Priors diagram/table

**Slide 6: MCMC Algorithm**
- [ ] Metropolis-Hastings sampler
- [ ] 1000 tuning + 2000 sampling iterations
- [ ] Convergence diagnostics: Rhat < 1.1, trace plots look good

**Slide 7: Key Findings - Convergence**
- [ ] Show trace plot (one convincing example)
- [ ] Show table of Rhat values (all < 1.1)
- [ ] Text: "Model converged successfully"

**Slide 8: Key Findings - Parameters**
- [ ] Table: phi ≈ 0.93 (high week-to-week persistence)
- [ ] mu_home ≈ 2.5 (home advantage)
- [ ] sigma ≈ 13 (residual SD)

**Slide 9: Team Rankings**
- [ ] Forest plot (or screenshot of forest): top 8 teams + 95% CI
- [ ] Text: "Which teams are strongest in playoffs?"

**Slide 10: Playoff Predictions**
- [ ] Highlight 1-2 most interesting matchups
- [ ] Example: "Team A 65% to beat Team B"
- [ ] Map margin + credible interval

**Slide 11: All 6 Predictions (Compact)**
- [ ] Table: 6 matchups, P(home wins), expected margin
- [ ] Visual: color-code by certainty (green = confident, yellow = uncertain)

**Slide 12: Model Validation**
- [ ] Posterior predictive check: "95% of observed margins in 95% CI"
- [ ] Show scatter plot: observed vs. predicted

**Slide 13: Limitations & Robustness**
- [ ] Playoffs ≠ regular season
- [ ] Sensitivity to priors: predictions stable (show comparison table)

**Slide 14: Conclusion**
- [ ] "Bayesian hierarchical model provides coherent uncertainty quantification"
- [ ] "Key insight: [something interesting from your analysis]"
- [ ] "Predictions ready for playoff games!"

### 9.2 Presentation Notes
- [ ] Write speaker notes for each slide (1-2 sentences per slide)
- [ ] Plan pacing: ~1 min per slide = 14 min total
- [ ] Practice presentation: time yourself, refine pacing

### 9.3 Backup Slides (Optional)
- [ ] 1-2 backup slides for Q&A (e.g., detailed math, alt. model specifications)

**DELIVERABLE:** 14-slide presentation + speaker notes (PDF + PPTX)

---

## PHASE 10: FINAL CHECKS & SUBMISSION
**Estimated Time: 1-2 hours | Status: [ ] NOT STARTED**

### 10.1 Report Checklist
- [ ] All sections complete
- [ ] All figures have captions and text explanation
- [ ] All tables labeled and interpreted
- [ ] Equations numbered and referenced
- [ ] References to Kenny Shirley's slides
- [ ] No typos, grammar checked
- [ ] PDFgenerated properly (equations render, figures visible)
- [ ] File size reasonable (< 50 MB)
- [ ] File name: `FSRM587_FinalProject_[YourName].pdf`

### 10.2 Code Checklist
- [ ] All code runs without errors (tested)
- [ ] Reproducible: set random seed, document paths
- [ ] Comments explaining each section
- [ ] Variable names descriptive
- [ ] No hardcoded values (use parameters instead)
- [ ] Output saved to pkl/csv for reference
- [ ] File name: `FSRM587_FinalProject_Code_[YourName].py` or `.ipynb`

### 10.3 Presentation Checklist
- [ ] All slides render properly
- [ ] Fonts readable on projector
- [ ] Figures visible and clear
- [ ] File name: `FSRM587_FinalProject_Presentation_[YourName].pptx`
- [ ] Tested on presentation computer

### 10.4 Canvas Submission
- [ ] Create folder: `FSRM587_FinalProject_[YourName]`
  - [ ] `Report.pdf` (main report)
  - [ ] `Code.py` or `Code.ipynb` (source code)
  - [ ] `Presentation.pptx` (slides)
  - [ ] `README.txt` (instructions to run code)
- [ ] Upload to Canvas by **11:59pm December 16**
- [ ] Upload presentation slides by **2:00pm December 9** (in-class presentation)

### 10.5 In-Class Presentation
- [ ] Print backup copies of slides (just in case)
- [ ] Test presenter remote/laptop with projector
- [ ] Arrive 10 min early for setup
- [ ] Practice deep breaths—you've got this! 🏈

**DELIVERABLE:** Polished submission package

---

## SUMMARY TIMELINE

| Phase | Task | Deadline | Status |
|-------|------|----------|--------|
| 1 | Data prep | Dec 3 | [ ] |
| 2A | Functions | Dec 4 | [ ] |
| 2B | Samplers | Dec 5 | [ ] |
| 2C | Main loop | Dec 5 | [ ] |
| 3 | Diagnostics | Dec 6 | [ ] |
| 4 | Inference | Dec 6 | [ ] |
| 5 | Predictions | Dec 7 | [ ] |
| 6 | Sensitivity | Dec 7 | [ ] |
| 7 | Validation | Dec 8 | [ ] |
| 8 | Report | Dec 8-9 | [ ] |
| 9 | Presentation | Dec 9 | [ ] |
| 10 | Submit | Dec 16 | [ ] |

---

## RESOURCES

- **Bayesian Reference:** Gelman et al. (2013) "Bayesian Data Analysis" - hierarchical models (Ch. 5)
- **MCMC Reference:** Chib & Greenberg (1995) "Understanding the Metropolis-Hastings Algorithm"
- **Kenny Shirley Slides:** `Bayesian-Analysis-NFL.pdf` in Canvas
- **Data Script:** `NFL_DataScrap-2024-2025.ipynb` in Canvas
- **Convergence Diagnostics:** Gelman & Rubin (1992), Brooks & Gelman (1998)

---

**Good luck! You've got 2 weeks to produce professional-grade Bayesian inference. Let's build this step by step.** 🚀
