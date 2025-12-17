#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:44:41 2025

@author: phillycheese
"""

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

# Read the CSV data
data = pd.read_csv('50Ah_NMC_1RC_params.csv')


LKF_A = np.mean(data['R0 [Ohm]'])

LKF_B = np.mean(data['R1 [Ohm]'])

LKF_C= np.mean(data['C1 [F]'])

def fit_and_plot_second_order_polynomials(data):
    """
    Fit second-order polynomials to R0, R1, and C1 vs SOC and plot them.
    
    Parameters:
    -----------
    data : pandas DataFrame
        DataFrame containing 'SOC', 'R0 [Ohm]', 'R1 [Ohm]', 'C1 [F]' columns
    """
    
    # Extract SOC values
    soc = data['SOC'].values
    
    # Create a dense SOC array for smooth polynomial curves
    soc_smooth = np.linspace(0, 1, 100)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # 1. Fit and plot R0
    R0 = data['R0 [Ohm]'].values
    R0_coeffs = np.polyfit(soc, R0, 2)
    R0_fit = np.polyval(R0_coeffs, soc_smooth)
    
    axes[0].scatter(soc, R0, color='blue', s=50, label='Original Data', zorder=3)
    axes[0].plot(soc_smooth, R0_fit, 'r-', linewidth=2, label=f'Polynomial Fit: R0 = {R0_coeffs[0]:.2e}soc² + {R0_coeffs[1]:.2e}soc + {R0_coeffs[2]:.2e}')
    axes[0].set_xlabel('State of Charge (SOC)')
    axes[0].set_ylabel('R0 [Ohm]')
    axes[0].set_title('Ohmic Resistance (R0) vs SOC with 2nd Order Polynomial Fit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    
    # 2. Fit and plot R1
    R1 = data['R1 [Ohm]'].values
    R1_coeffs = np.polyfit(soc, R1, 2)
    R1_fit = np.polyval(R1_coeffs, soc_smooth)
    
    axes[1].scatter(soc, R1, color='red', s=50, label='Original Data', zorder=3)
    axes[1].plot(soc_smooth, R1_fit, 'b-', linewidth=2, label=f'Polynomial Fit: R1 = {R1_coeffs[0]:.3f}soc² + {R1_coeffs[1]:.3f}soc + {R1_coeffs[2]:.3f}')
    axes[1].set_xlabel('State of Charge (SOC)')
    axes[1].set_ylabel('R1 [Ohm]')
    axes[1].set_title('Polarization Resistance (R1) vs SOC with 2nd Order Polynomial Fit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    
    # 3. Fit and plot C1
    C1 = data['C1 [F]'].values
    C1_coeffs = np.polyfit(soc, C1, 2)
    C1_fit = np.polyval(C1_coeffs, soc_smooth)
    
    axes[2].scatter(soc, C1, color='green', s=50, label='Original Data', zorder=3)
    axes[2].plot(soc_smooth, C1_fit, 'm-', linewidth=2, label=f'Polynomial Fit: C1 = {C1_coeffs[0]:.3e}soc² + {C1_coeffs[1]:.3e}soc + {C1_coeffs[2]:.3e}')
    axes[2].set_xlabel('State of Charge (SOC)')
    axes[2].set_ylabel('C1 [F]')
    axes[2].set_title('Polarization Capacitance (C1) vs SOC with 2nd Order Polynomial Fit')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Return the coefficients
    coefficients = {
        'R0': R0_coeffs.tolist(),
        'R1': R1_coeffs.tolist(),
        'C1': C1_coeffs.tolist()
    }
    
    return coefficients
bruh = fit_and_plot_second_order_polynomials(data)
  

def plot_discrete_coeffs_from_polynomials(bruh, T=1, eta=1, Q=50):
    """
    Plot discrete-time filter coefficients as functions of SOC.
    
    Parameters:
    -----------
    bruh : dict
        Dictionary containing polynomial coefficients for R0, R1, C1
        Format: {'R0': [a2, a1, a0], 'R1': [a2, a1, a0], 'C1': [a2, a1, a0]}
    T : float
        Sampling time in seconds
    eta : float
        Coulombic efficiency (typically 1 for discharge, <1 for charge)
    Q : float
        Battery capacity in Ah
    """
    
    # Extract coefficients from the dictionary
    a2_R0, a1_R0, a0_R0 = bruh['R0']
    a2_R1, a1_R1, a0_R1 = bruh['R1']
    a2_C, a1_C, a0_C = bruh['C1']
    
    # Create SOC array from 0 to 1
    soc = np.linspace(0, 1, 1000)
    
    # Initialize arrays for storing coefficients
    alpha_d = np.zeros_like(soc)
    beta_d1 = np.zeros_like(soc)
    beta_d0 = np.zeros_like(soc)
    
    # Calculate coefficients for each SOC value
    for i, z in enumerate(soc):
        # 1. Calculate battery parameters from polynomials
        R = a0_R0 + a1_R0*z + a2_R0*(z**2)
        R_1 = a0_R1 + a1_R1*z + a2_R1*(z**2)
        C = a0_C + a1_C*z + a2_C*(z**2)
        
        # 2. Calculate continuous-time parameters
        b_1 = R * R_1 * C
        b_0 = R + R_1
        a_0 = R_1 * C
        
        # 3. Apply bilinear transform
        beta_1 = 2*b_1/T + b_0
        beta_0 = b_0 - 2*b_1/T
        alpha_1 = 2*a_0/T + 1
        alpha_0 = 1 - 2*a_0/T
        
        # 4. Calculate discrete-time filter coefficients
        alpha_d[i] = -alpha_0/alpha_1
        beta_d1[i] = beta_1/alpha_1
        beta_d0[i] = beta_0/alpha_1
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot α_d = -α₀/α₁
    axes[0].plot(soc, alpha_d, 'b-', linewidth=2)
    axes[0].set_xlabel('State of Charge (SOC)')
    axes[0].set_ylabel(r'$\alpha_d = -\frac{\alpha_0}{\alpha_1}$')
    axes[0].set_title(r'Discrete-time Coefficient $\alpha_d$ vs SOC')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    
    # Plot β_d1 = β₁/α₁
    axes[1].plot(soc, beta_d1, 'r-', linewidth=2)
    axes[1].set_xlabel('State of Charge (SOC)')
    axes[1].set_ylabel(r'$\beta_{d1} = \frac{\beta_1}{\alpha_1}$')
    axes[1].set_title(r'Discrete-time Coefficient $\beta_{d1}$ vs SOC')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    
    # Plot β_d0 = β₀/α₁
    axes[2].plot(soc, beta_d0, 'g-', linewidth=2)
    axes[2].set_xlabel('State of Charge (SOC)')
    axes[2].set_ylabel(r'$\beta_{d0} = \frac{\beta_0}{\alpha_1}$')
    axes[2].set_title(r'Discrete-time Coefficient $\beta_{d0}$ vs SOC')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Also print coefficient ranges for reference
    print(f"α_d range: [{alpha_d.min():.4f}, {alpha_d.max():.4f}]")
    print(f"β_d1 range: [{beta_d1.min():.4f}, {beta_d1.max():.4f}]")
    print(f"β_d0 range: [{beta_d0.min():.4f}, {beta_d0.max():.4f}]")
    
    return soc, alpha_d, beta_d1, beta_d0

# Run the function with your coefficients
soc_vals, alpha_d_vals, beta_d1_vals, beta_d0_vals = plot_discrete_coeffs_from_polynomials(bruh, T=1)
