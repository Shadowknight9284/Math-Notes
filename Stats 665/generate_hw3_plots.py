#!/usr/bin/env python3
"""
Exercise 16: Spectral Density Plots
Generates all plots for homework submission
Saves to hw3img/ folder
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('hw3img', exist_ok=True)

# Frequency range
omega = np.linspace(-np.pi, np.pi, 1000)

# Configure plotting style
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def save_plot(filename, title_text):
    """Helper to save plots with consistent formatting."""
    plt.xlabel('Frequency $\\lambda$ (rad)')
    plt.ylabel('Spectral Density $f(\\lambda)$')
    plt.title(title_text)
    plt.xlim([-np.pi, np.pi])
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
               ['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
    plt.tight_layout()
    plt.savefig(f'hw3img/{filename}', dpi=300, bbox_inches='tight')
    print(f'  ✓ Saved: hw3img/{filename}')
    plt.close()

print("="*60)
print("Generating Exercise 16 Spectral Density Plots")
print("="*60)

# ============================================================
# Part (a): MA(1) Processes
# ============================================================
print("\nPart (a): MA(1) Processes")
print("-" * 60)

# MA(1) with θ = 0.9, σ² = 2
f_ma_pos = (2 / (2*np.pi)) * (1 + 0.81 + 1.8*np.cos(omega))
plt.figure()
plt.plot(omega, f_ma_pos, 'b-', linewidth=2)
save_plot('ma1_theta_pos09.png', 
          'MA(1): $X_t = Z_t + 0.9Z_{t-1}$, $\\{Z_t\\} \\sim WN(0,2)$')

# MA(1) with θ = -0.9, σ² = 2
f_ma_neg = (2 / (2*np.pi)) * (1 + 0.81 - 1.8*np.cos(omega))
plt.figure()
plt.plot(omega, f_ma_neg, 'r-', linewidth=2)
save_plot('ma1_theta_neg09.png', 
          'MA(1): $X_t = Z_t - 0.9Z_{t-1}$, $\\{Z_t\\} \\sim WN(0,2)$')

# ============================================================
# Part (b): AR(1) Processes
# ============================================================
print("\nPart (b): AR(1) Processes")
print("-" * 60)

# AR(1) with φ = 0.9, σ² = 3
f_ar_pos = (3 / (2*np.pi)) / (1 + 0.81 - 1.8*np.cos(omega))
plt.figure()
plt.plot(omega, f_ar_pos, 'b-', linewidth=2)
save_plot('ar1_phi_pos09.png', 
          'AR(1): $X_t = 0.9X_{t-1} + Z_t$, $\\{Z_t\\} \\sim WN(0,3)$')

# AR(1) with φ = -0.9, σ² = 3
f_ar_neg = (3 / (2*np.pi)) / (1 + 0.81 + 1.8*np.cos(omega))
plt.figure()
plt.plot(omega, f_ar_neg, 'r-', linewidth=2)
save_plot('ar1_phi_neg09.png', 
          'AR(1): $X_t = -0.9X_{t-1} + Z_t$, $\\{Z_t\\} \\sim WN(0,3)$')

# ============================================================
# Part (c): Problem 7 Processes
# ============================================================
print("\nPart (c): Problem 7 Processes")
print("-" * 60)

sigma_sq = 4  # a_t ~ N(0,4)

# (i) AR(3): r_t = 0.3 + 0.8*r_{t-1} - 0.5*r_{t-2} - 0.2*r_{t-3} + a_t
print("  Computing AR(3)...")
phi = [0.8, -0.5, -0.2]
ar_poly = np.ones_like(omega, dtype=complex)
for k, ph in enumerate(phi, start=1):
    ar_poly -= ph * np.exp(-1j * k * omega)
f_ar3 = (sigma_sq / (2*np.pi)) / np.abs(ar_poly)**2

plt.figure()
plt.plot(omega, f_ar3, 'darkblue', linewidth=2)
save_plot('ar3_problem7.png', 
          'AR(3): $r_t = 0.3 + 0.8r_{t-1} - 0.5r_{t-2} - 0.2r_{t-3} + a_t$')

# (ii) MA(3): r_t = 0.3 + a_t + 0.8*a_{t-1} - 0.5*a_{t-2} - 0.2*a_{t-3}
print("  Computing MA(3)...")
theta = [0.8, -0.5, -0.2]
ma_poly = np.ones_like(omega, dtype=complex)
for k, th in enumerate(theta, start=1):
    ma_poly += th * np.exp(-1j * k * omega)
f_ma3 = (sigma_sq / (2*np.pi)) * np.abs(ma_poly)**2

plt.figure()
plt.plot(omega, f_ma3, 'darkgreen', linewidth=2)
save_plot('ma3_problem7.png', 
          'MA(3): $r_t = 0.3 + a_t + 0.8a_{t-1} - 0.5a_{t-2} - 0.2a_{t-3}$')

# (iii) ARMA(3,2): Combined AR(3) and MA(2)
print("  Computing ARMA(3,2)...")
phi_arma = [0.8, -0.5, -0.2]
theta_arma = [0.5, 0.3]

ar_poly_arma = np.ones_like(omega, dtype=complex)
for k, ph in enumerate(phi_arma, start=1):
    ar_poly_arma -= ph * np.exp(-1j * k * omega)

ma_poly_arma = np.ones_like(omega, dtype=complex)
for k, th in enumerate(theta_arma, start=1):
    ma_poly_arma += th * np.exp(-1j * k * omega)

f_arma32 = (sigma_sq / (2*np.pi)) * np.abs(ma_poly_arma)**2 / np.abs(ar_poly_arma)**2

plt.figure()
plt.plot(omega, f_arma32, 'darkred', linewidth=2)
save_plot('arma32_problem7.png', 
          'ARMA(3,2): $r_t = 0.3 + 0.8r_{t-1} - 0.5r_{t-2} - 0.2r_{t-3} + a_t + 0.5a_{t-1} + 0.3a_{t-2}$')

# ============================================================
# Create comparison plots
# ============================================================
print("\nCreating comparison plots...")
print("-" * 60)

# MA(1) comparison
plt.figure(figsize=(10, 6))
plt.plot(omega, f_ma_pos, 'b-', linewidth=2, label='$\\theta = +0.9$')
plt.plot(omega, f_ma_neg, 'r--', linewidth=2, label='$\\theta = -0.9$')
plt.xlabel('Frequency $\\lambda$ (rad)')
plt.ylabel('Spectral Density $f(\\lambda)$')
plt.title('MA(1) Comparison: Effect of Sign')
plt.xlim([-np.pi, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           ['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hw3img/ma1_comparison.png', dpi=300, bbox_inches='tight')
print('  ✓ Saved: hw3img/ma1_comparison.png')
plt.close()

# AR(1) comparison
plt.figure(figsize=(10, 6))
plt.plot(omega, f_ar_pos, 'b-', linewidth=2, label='$\\phi = +0.9$')
plt.plot(omega, f_ar_neg, 'r--', linewidth=2, label='$\\phi = -0.9$')
plt.xlabel('Frequency $\\lambda$ (rad)')
plt.ylabel('Spectral Density $f(\\lambda)$')
plt.title('AR(1) Comparison: Effect of Sign')
plt.xlim([-np.pi, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           ['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hw3img/ar1_comparison.png', dpi=300, bbox_inches='tight')
print('  ✓ Saved: hw3img/ar1_comparison.png')
plt.close()

# Problem 7 comparison
plt.figure(figsize=(10, 6))
plt.plot(omega, f_ar3, 'darkblue', linewidth=2, label='AR(3)')
plt.plot(omega, f_ma3, 'darkgreen', linewidth=2, label='MA(3)')
plt.plot(omega, f_arma32, 'darkred', linewidth=2, label='ARMA(3,2)')
plt.xlabel('Frequency $\\lambda$ (rad)')
plt.ylabel('Spectral Density $f(\\lambda)$')
plt.title('Problem 7: Comparison of All Three Processes')
plt.xlim([-np.pi, np.pi])
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           ['$-\\pi$', '$-\\pi/2$', '0', '$\\pi/2$', '$\\pi$'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hw3img/problem7_comparison.png', dpi=300, bbox_inches='tight')
print('  ✓ Saved: hw3img/problem7_comparison.png')
plt.close()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("COMPLETE! Generated 10 plots:")
print("="*60)
print("\nPart (a) - MA(1):")
print("  1. ma1_theta_pos09.png")
print("  2. ma1_theta_neg09.png")
print("  3. ma1_comparison.png")
print("\nPart (b) - AR(1):")
print("  4. ar1_phi_pos09.png")
print("  5. ar1_phi_neg09.png")
print("  6. ar1_comparison.png")
print("\nPart (c) - Problem 7:")
print("  7. ar3_problem7.png")
print("  8. ma3_problem7.png")
print("  9. arma32_problem7.png")
print("  10. problem7_comparison.png")
print("\nAll plots saved to: hw3img/")
print("="*60)
