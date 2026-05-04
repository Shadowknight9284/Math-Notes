import numpy as np
import matplotlib.pyplot as plt

# State encoding and decoding
states = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

def gibbs_step(state, beta, rng):
    """One random-scan Gibbs step for the 2-spin Ising model."""
    sigma1, sigma2 = state
    rho = np.tanh(beta)
    # choose which spin to update
    if rng.random() < 0.5:
        # update sigma1 given sigma2
        p_plus = 0.5 * (1 + rho * sigma2)   # P(sigma1 = +1 | sigma2)
        sigma1 = 1 if rng.random() < p_plus else -1
    else:
        # update sigma2 given sigma1
        p_plus = 0.5 * (1 + rho * sigma1)   # P(sigma2 = +1 | sigma1)
        sigma2 = 1 if rng.random() < p_plus else -1
    return (sigma1, sigma2)

def run_chain(beta, T=100, x0=(1, 1), seed=0):
    rng = np.random.default_rng(seed)
    sigma1_vals = np.zeros(T+1, dtype=int)
    states_list = [x0]
    sigma1_vals[0] = x0[0]
    x = x0
    for t in range(T):
        x = gibbs_step(x, beta, rng)
        states_list.append(x)
        sigma1_vals[t+1] = x[0]
    return np.array(states_list), sigma1_vals

# Example: run for several betas and plot
betas = [0.1, 1.0, 2.0]
T = 100

fig, axes = plt.subplots(len(betas), 2, figsize=(10, 3 * len(betas)), sharex='col')

for i, beta in enumerate(betas):
    _, sigma1 = run_chain(beta, T=T, x0=(1, 1), seed=123 + i)

    # Sample path
    ax_path = axes[i, 0]
    ax_path.step(range(T+1), sigma1, where='post')
    ax_path.set_ylim(-1.2, 1.2)
    ax_path.set_xlabel('t')
    ax_path.set_ylabel(r'$\sigma_1^{(t)}$')
    ax_path.set_title(f'Sample path, beta = {beta}')

    # Histogram
    ax_hist = axes[i, 1]
    ax_hist.hist(sigma1, bins=[-1.5, -0.5, 0.5, 1.5], rwidth=0.8)
    ax_hist.set_xticks([-1, 1])
    ax_hist.set_xlabel(r'$\sigma_1$')
    ax_hist.set_ylabel('count')
    ax_hist.set_title(f'Histogram, beta = {beta}')

plt.tight_layout()
plt.show()