#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 15:34:25 2025

@author: phillycheese
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from OCV_GLM  import ocv_func

from true_parameters import bruh

#variabler
T=1
eta= 0.9898
Q = 51.4*3600

#################################
#                               #
#   PLOTTING CONFIGURATION      #
#                               #
#################################

# Parameter plot y-axis limits
# Set to None for automatic scaling, or specify [min, max] for each parameter
plot_limits = {
    'R0': None,  # Automatic scaling for R0
    'R1': None,      # Automatic scaling for R1  
    'C1': None,        # Automatic scaling for C1
    'error_R0': None,  # Automatic scaling for R0 error
    'error_R1': None,  # Automatic scaling for R1 error
    'error_C1': None   # Automatic scaling for C1 error
}
# Example: If you want to fix the y-axis for specific parameters:
# plot_limits = {
#     'R0': [0.01, 0.03],      # Fix R0 between 0.01 and 0.03 Ohm
#     'R1': [0.005, 0.015],    # Fix R1 between 0.005 and 0.015 Ohm
#     'C1': [2000, 6000],      # Fix C1 between 2000 and 6000 F
#     'error_R0': [-0.01, 0.01],    # Fix R0 error between -0.01 and 0.01
#     'error_R1': [-0.005, 0.005],  # Fix R1 error between -0.005 and 0.005
#     'error_C1': [-1000, 1000]     # Fix C1 error between -1000 and 1000
# }



#################################
#                               #
#           inputs              #
#                               #
#################################

#data indlæsning
df = pd.read_csv('OCV_curve.csv')

# Extract SOC and OCV data
df_ocv1 = pd.read_csv('OCV_curve.csv')
df_ocv2 = pd.read_csv('50Ah_NMC_ocv_curve.csv')

soc_data = df['SOC'].values
soc_data = np.round(soc_data, 3)

average_curve = pd.DataFrame()
average_curve['SOC'] = soc_data #samme soc akse

average_curve['OCV_avg [V]'] = (df_ocv1['OCV [V]'] + df_ocv2['OCV [V]']) / 2

ocv_data = average_curve['OCV_avg [V]']

df = pd.read_csv('50Ah_NMC_1RC_params.csv')

df = pd.read_csv('udds.csv')

#inputs = df['Normalized current (A)'].tolist()

inputs = [x *200 for x in (-df['Normalized current (A)']).tolist()] 

#inputs = np.full(4000, -0.12)

#inputs = np.zeros(10000)
#pulse_regions = [(0, 300), (5000, 5300)]
#for start, end in pulse_regions:
#    inputs[start:end] = -0.1

# Generate sine wave
#n_samples = 10000
#time = np.linspace(0, 10, n_samples)
#frequency = 0.5  # Hz
#amplitude = 0.0001

# Sine wave input
#inputs = amplitude * np.sin(2 * np.pi * frequency * time)



#################################
#                               #
#       define functions        #
#                               #
#################################




def Couloumb_Counting_true(input_, initial_x, OCV_data, SOC_data):
    maxIter = len(input_)
    x = initial_x # [V_RRC[0]  z[0]]^T
    xstore = np.zeros((len(x), maxIter)) # matix med len[x] rows og maxIter koloner
    parastore = np.zeros((3, maxIter))
    xstore[:,0] = x.T[0]
    output_true  = np.zeros(maxIter) # "true" output i kalmann filter
    for k in range(1, maxIter):
        
            
        u = np.array([[input_[k]], [input_[k-1]]]) 
        
        soc = x[1, 0]
        #R, R_1, C = get_battery_params(soc)
      
        # For R0: coefficients are [a2, a1, a0] from polyfit
        a2_R0, a1_R0, a0_R0 = bruh['R0']  # This part is correct
        R = a0_R0 + a1_R0*soc + a2_R0*(soc**2)  # Fixed this line

        # For R1: same pattern
        a2_R1, a1_R1, a0_R1 = bruh['R1']
        R_1 = a0_R1 + a1_R1*soc + a2_R1*(soc**2)  # Fixed

        # For C1: same pattern
        a2_C, a1_C, a0_C = bruh['C1']  # Fixed variable names
        C = a0_C + a1_C*soc + a2_C*(soc**2)  # Fixed
        
      
        if k == 1:
            print(soc)
    
        
        b_1= R*R_1*C
        b_0 = R+R_1
        a_0 = R_1*C
            

        beta_1 = 2*b_1/T + b_0 
        beta_0 = b_0 - 2*b_1/T
        alph_1 = 2*a_0/T + 1
        alph_0 = 1 - 2*a_0/T 
        
        
        parastore[0, k] = -1*alph_0/alph_1     
        parastore[1, k] = beta_1/alph_1
        parastore[2, k] = beta_0/alph_1

        
        if k == 1:
            print(-1* alph_0/alph_1, beta_1/alph_1, beta_0/alph_1)

        
        A = np.array([[-1 * alph_0/alph_1, 0], [0, 1]])
        if input_[k] > 0:
            B = np.array([[ beta_1/alph_1, beta_0/alph_1], [0,  T / Q ]])
        else:
            B = np.array([[ beta_1/alph_1, beta_0/alph_1], [0,  eta*T / Q ]])

        
        x = np.matmul(A, x) + np.matmul(B, u)
        
        #output_true[k] = x[0, 0]  + np.array(OCV_data[SOC_data == np.round(x[1, 0], 3)])[0] 
        
        output_true[k] = x[0, 0]  + ocv_func(x[1, 0])
        xstore[:,k] = x.T[0]
    return maxIter, xstore, output_true, parastore


def RLS(inputs, measured_voltage, initial_x, initial_sigmax, sigmav, lam):
    maxIter = len(inputs)
    xhat = initial_x
    SigmaX = initial_sigmax
    xhatstore = np.zeros((len(xhat), maxIter))
    xhatstore[:,0] = xhat 
    esti = np.zeros(maxIter)
    
    print(xhat.shape)
    
    for k in range(1, maxIter):
        
        H = np.array([measured_voltage[k-1], inputs[k], inputs[k-1], 1])

        
        denominator = H.T @ SigmaX @ H + lam * sigmav
        L = (SigmaX @ H) / denominator

        # H er defineret som en vektor her, hvor imod at i bogen er det en row vector 
        
        xhat = xhat + L * ( measured_voltage[k] - np.dot(H, xhat))
        
        esti[k]= np.dot(H, xhat)
        
        xhatstore[:,k] = xhat.T
        
        if k== 1:
            print( f' dif = {measured_voltage[k] - np.dot(H, xhat)}')
            #print(L.shape)
            #print(xhat)
            print(xhatstore)
        SigmaX = (1/lam) * ((np.eye(4) - np.outer(L, H)) @ SigmaX @ (np.eye(4) - np.outer(L, H)).T) + sigmav * np.outer(L, L) # np.outer(L, H) = L H^T (matrix outcome)
        # sigmav * np.outer(L, L) = L * V * L^T = matrix outcome    
        
    return xhatstore, esti



#########################################################
#                                                       #
#          simulation og parameter for simulation       #
#                                                       #
#########################################################


xtrue = np.array([[0],  # intial state, start spænding 0, start soc 0.7
                     [0.7]])

# Convert to numpy array for easier manipulation
inputs_clean = np.array(inputs)

measure_error=0.0001

noise = np.random.normal(0, measure_error, len(inputs_clean))
inputs_noise = inputs_clean + noise
    

MaxIter, xstore, y_NoNoise, Parameters = Couloumb_Counting_true(inputs_noise, xtrue, ocv_data, soc_data)


measure_error_2=0.0001
noise = np.random.normal(0, measure_error_2, len(xstore[0, :]))

y_Noise = y_NoNoise + noise
#bruge outout med støj som y_k i RLS, 

inital_soc = 0.7 # inital soc fra xtrue fra CC

#vi finder intial, /theta_0, ved at bruge intial soc, til at bestemme parameterne og v_oc

#iR, iR_1, iC = get_battery_params(inital_soc)
#print(get_battery_params(inital_soc))
#b_1= iR*iR_1*iC
#b_0 = iR+iR_1
#a_0 = iR_1*iC

#initial_x = np.array([-1, 4.8, 4.8, 13 ])


soc = inital_soc
#R, R_1, C = get_battery_params(soc)

# For R0: coefficients are [a2, a1, a0] from polyfit
a2_R0, a1_R0, a0_R0 = bruh['R0']  # This part is correct
R = a0_R0 + a1_R0*soc + a2_R0*(soc**2)  # Fixed this line

# For R1: same pattern
a2_R1, a1_R1, a0_R1 = bruh['R1']
R_1 = a0_R1 + a1_R1*soc + a2_R1*(soc**2)  # Fixed

# For C1: same pattern
a2_C, a1_C, a0_C = bruh['C1']  # Fixed variable names
C = a0_C + a1_C*soc + a2_C*(soc**2)  # Fixed


b_1= R*R_1*C
b_0 = R+R_1
a_0 = R_1*C
    

beta_1 = 2*b_1/T + b_0 
beta_0 = b_0 - 2*b_1/T
alph_1 = 2*a_0/T + 1
alph_0 = 1 - 2*a_0/T 
 

initial_x =   np.array([-1 * alph_0 / alph_1, beta_1 / alph_1, beta_0 / alph_1, (alph_0 / alph_1)*ocv_func(soc)+ ocv_func(soc)])


print(initial_x)

#initial_sigmax=np.eye(4)*1

initial_sigmax = 0.000001*np.diag([1, 1, 1, 1]) 

lamlam= 0.98
Pe, esti= RLS(inputs_clean, y_Noise, initial_x, initial_sigmax, sigmav=measure_error_2, lam=lamlam) 

#Parammeter_estimations = transform_to_R0_R1_C_columns(Pe, 1)

Parammeter_estimations = Pe

########################
#                      #
#        Plots         #
#                      #
########################


plt.figure(figsize=(18, 12))

plt.ylabel('Normalized current * 1000')
plt.title('Input med noise')
plt.legend()
plt.grid(True)

# First figure: Original data
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(inputs_noise, label = '200*Normalized Currents [A]')
#plt.plot(xstore[0, :], label='KALMAN FILTER Y HAT')
plt.legend(loc='upper right')
plt.xlabel('k')
plt.ylabel('Input[A]')
plt.title(label='True System')
plt.grid(True)

# Plot second row of xstore (V_R1 - voltage across R1)
plt.subplot(3, 1, 2)
plt.plot(xstore[1, :] , label='True SOC', color='red')
plt.xlabel('k')
plt.ylabel('SOC')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(y_Noise, label='System Output Voltage[V]', color='green')
plt.xlabel('k')
plt.ylabel('Voltage[V]')
plt.legend(loc='upper right')
plt.grid(True)




plt.tight_layout()
plt.savefig('RLS_ipnutogsoc.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
# Plot the 3 parameters with true values, estimations, and percentage errors
fig = plt.figure(figsize=(12, 10))

# Create a common x-axis for all plots
iterations = np.arange(1, MaxIter)

# Calculate PERCENTAGE errors for each parameter (reusing your calculation)
epsilon = 1e-10
error_R0_pct = 100 * (Parammeter_estimations[0, :MaxIter-1] - Parameters[0, 1:]) / (np.abs(Parameters[0, 1:]) + epsilon)
error_R1_pct = 100 * (Parammeter_estimations[1, :MaxIter-1] - Parameters[1, 1:]) / (np.abs(Parameters[1, 1:]) + epsilon)
error_C_pct = 100 * (Parammeter_estimations[2, :MaxIter-1] - Parameters[2, 1:]) / (np.abs(Parameters[2, 1:]) + epsilon)

# Format noise values for display (using your variables from earlier)
sigma_v = measure_error  # Process noise standard deviation
sigma_w = measure_error_2  # Measurement noise standard deviation

# Format with scientific notation if small, otherwise regular format
def format_noise(value):
    if value < 0.001:
        return f"{value:.2e}"
    else:
        return f"{value:.4f}"

sigma_v_str = format_noise(sigma_v)
sigma_w_str = format_noise(sigma_w)

# Create ONE title for the whole plot with forgetting factor and noise information
main_title = f'Battery Parameters: True vs Estimated with Percentage Error (Red)\nForgetting Factor λ = {lamlam}, Process Noise σ_v = {sigma_v_str}, Measurement Noise σ_w = {sigma_w_str}'

# Plot θ₁ (-alpha_0/alpha_1) - True vs Estimated with Percentage Error on secondary y-axis
plt.subplot(3, 1, 1)
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot true parameter with DASHED line
line1 = ax1.plot(Parameters[0, 1:], label='True θ₁', color='blue', linewidth=1.5, linestyle='--')
# Plot estimated parameter with SOLID line (same color and thickness)
line2 = ax1.plot(Parammeter_estimations[0, 0:MaxIter-1], label='Estimated θ₁', color='blue', linewidth=1.5, linestyle='-')
# Plot percentage error in RED on right axis
line3 = ax2.plot(iterations, error_R0_pct, label='Percentage Error', color='red', alpha=0.7, linewidth=1)

# Apply y-axis limits if specified
if plot_limits['R0'] is not None:
    ax1.set_ylim(plot_limits['R0'])

ax1.set_ylabel('θ₁', color='black', fontsize=12)
ax2.set_ylabel('Error (%)', color='red', fontsize=10)
ax1.tick_params(axis='y', labelcolor='black')
ax2.tick_params(axis='y', labelcolor='red')
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right')
ax1.grid(True, alpha=0.3)

# Plot θ₂ (beta_1/alpha_1) - True vs Estimated with Percentage Error on secondary y-axis
plt.subplot(3, 1, 2)
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot true parameter with DASHED line
line1 = ax1.plot(Parameters[1, 1:], label='True θ₂', color='orange', linewidth=1.5, linestyle='--')
# Plot estimated parameter with SOLID line (same color and thickness)
line2 = ax1.plot(Parammeter_estimations[1, 0:MaxIter-1], label='Estimated θ₂', color='orange', linewidth=1.5, linestyle='-')
# Plot percentage error in RED on right axis
line3 = ax2.plot(iterations, error_R1_pct, label='Percentage Error', color='red', alpha=0.7, linewidth=1)
ax1.set_ylim([4.4,4.9])
# Apply y-axis limits if specified
if plot_limits['R1'] is not None:
    ax1.set_ylim(plot_limits['R1'])
    

ax1.set_ylabel('θ₂', color='black', fontsize=12)
ax2.set_ylabel('Error (%)', color='red', fontsize=10)
ax1.tick_params(axis='y', labelcolor='black')
ax2.tick_params(axis='y', labelcolor='red')
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right')
ax1.grid(True, alpha=0.3)

# Plot θ₃ (beta_0/alpha_1) - True vs Estimated with Percentage Error on secondary y-axis
plt.subplot(3, 1, 3)
ax1 = plt.gca()
ax2 = ax1.twinx()

# Plot true parameter with DASHED line
line1 = ax1.plot(Parameters[2, 1:], label='True θ₃', color='green', linewidth=1.5, linestyle='--')
# Plot estimated parameter with SOLID line (same color and thickness)
line2 = ax1.plot(Parammeter_estimations[2, 0:MaxIter-1], label='Estimated θ₃', color='green', linewidth=1.5, linestyle='-')
# Plot percentage error in RED on right axis
line3 = ax2.plot(iterations, error_C_pct, label='Percentage Error', color='red', alpha=0.7, linewidth=1)

# Apply y-axis limits if specified
if plot_limits['C1'] is not None:
    ax1.set_ylim(plot_limits['C1'])

ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('θ₃', color='black', fontsize=12)
ax2.set_ylabel('Error (%)', color='red', fontsize=10)
ax1.tick_params(axis='y', labelcolor='black')
ax2.tick_params(axis='y', labelcolor='red')
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right')
ax1.grid(True, alpha=0.3)

# Add ONE title for the entire figure
fig.suptitle(main_title, fontsize=14, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
plt.savefig('RLS_ESTI.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()