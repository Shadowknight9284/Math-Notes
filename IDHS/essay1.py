# Simulation of a Wisdom of the Crowd scenario

import numpy as np

N = 300  # Number of individuals
true_val = 10 # True value to be estimated
np.random.seed(0)  # For reproducibility
std_dev = true_val / 10  # Standard deviation of individual estimates

def simulate_wisdom_of_crowd(N, true_val, std_dev):
    # Generate individual estimates with some noise
    estimates = np.random.normal(loc=true_val, scale=std_dev, size=N)
    uniform_estimates = np.random.uniform(low=0, high=1, size=N)
    # Calculate X_t = U_t * E_t + (1 - U_t) * X_{t-1} ,X_0 = E_0
    X = np.zeros(N)
    X[0] = estimates[0]
    for t in range(1, N):
        X[t] = uniform_estimates[t] * estimates[t] + (1 - uniform_estimates[t]) * X[t-1]
    mean_estimate = np.mean(estimates)
    mean_X = np.mean(X)
    error_estimate = abs(mean_estimate - true_val)
    error_X = abs(mean_X - true_val)
    std_dev_estimate = np.std(estimates)
    std_dev_X = np.std(X)
    var_estimate = np.var(estimates)
    var_X = np.var(X)
    return (mean_estimate, error_estimate, std_dev_estimate, var_estimate,
            mean_X, error_X, std_dev_X, var_X)

average_results = np.zeros(8)  # To accumulate results

for _ in range(10000): # Run multiple simulations to average results
    mean_estimate, error_estimate, std_dev_estimate, var_estimate, mean_X, error_X, std_dev_X, var_X = simulate_wisdom_of_crowd(N, true_val, std_dev)
    average_results += np.array([mean_estimate, error_estimate, std_dev_estimate, var_estimate,
                                 mean_X, error_X, std_dev_X, var_X])
    
average_results /= 10000  # Average over simulations
mean_estimate, error_estimate, std_dev_estimate, var_estimate, mean_X, error_X, std_dev_X, var_X = average_results


print("Results after averaging over 10,000 simulations:")
print(f"Mean of estimates: {mean_estimate}, Error: {error_estimate}, Std Dev: {std_dev_estimate}, Variance: {var_estimate}")
print(f"Mean of X: {mean_X}, Error: {error_X}, Std Dev: {std_dev_X}, Variance: {var_X}")
