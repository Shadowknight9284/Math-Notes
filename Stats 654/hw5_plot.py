import numpy as np
import matplotlib.pyplot as plt

np.random.seed(654)
n = 100000
X = np.random.normal(size=n)
Y = np.random.normal(size=n)

rhos = [1, 0, -1]

import os
os.makedirs('output', exist_ok=True)

for r in rhos:
    Z = r * X + np.sqrt(1 - r**2) * Y

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    axes[0].scatter(X[::50], Z[::50], s=5, alpha=0.5)
    axes[0].set_title(f"Scatterplot X vs Z, rho={r}")
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Z')

    axes[1].hist(X, bins=50, density=True, alpha=0.5, label='X')
    axes[1].hist(Z, bins=50, density=True, alpha=0.5, label='Z')
    axes[1].set_title(f"Histograms, rho={r}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f'output/gauss_coupling_rho_{r}.png', dpi=200)
    plt.close(fig)