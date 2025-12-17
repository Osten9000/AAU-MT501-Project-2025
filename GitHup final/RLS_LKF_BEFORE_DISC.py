#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 09:56:40 2025

@author: phillycheese
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from OCV_GLM  import a, b, ocv_func

from true_parameters import bruh

C_m = np.array([[1, b]])

#variabler
T=1
eta= 0.9898
Q = 51.4*3600

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

df = pd.read_csv('udds.csv')

inputs = [x *1 for x in (-df['Normalized current (A)']).tolist()] 



def Couloumb_Counting_true(input_, initial_x):
    maxIter = len(input_)
    x = initial_x # [V_RRC[0]  z[0]]^T
    xstore = np.zeros((len(x), maxIter)) # matix med len[x] rows og maxIter koloner
    xstore[:,0] = x.T[0]
    output_true  = np.zeros(maxIter) # "true" output i kalmann filter
    for k in range(1, maxIter):
        u = np.array([[input_[k]], [input_[k-1]]]) 
        soc = x[1, 0]
      
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
        
        
        A = np.array([[-1 * alph_0/alph_1, 0], [0, 1]])
        if input_[k] > 0:
            B = np.array([[ beta_1/alph_1, beta_0/alph_1], [0,   T / Q ]])
        else:
            B = np.array([[ beta_1/alph_1, beta_0/alph_1], [0,  eta*T / Q ]])

        if k==2:
            print(A,B)
        x = np.matmul(A, x) + np.matmul(B, u)
        
        #output_true[k] = x[0, 0]  + np.array(ocv_data[soc_data == np.round(x[1, 0], 3)])[0]
        
        output_true[k] = x[0, 0]  + ocv_func(x[1,0])
        
        xstore[:,k] = x.T[0]
    return maxIter, xstore, output_true



def Kalman_Filter_RLS(input_, measured_voltage, initial_x, initial_SigmaX, SigmaW, SigmaV, initial_theta, initial_sigmax_R, sigmav_R, xtore,lam):
    maxIter = len(input_)
    xhat = initial_x
    yhatstore = np.zeros(maxIter)
    SigmaX = initial_SigmaX
    
    xhatstore = np.zeros((len(xhat), maxIter))
    xhatstore[:,0] = xhat.T[0]
    
    SigmaXstore = np.zeros((len(xhat)**2, maxIter))
    SigmaXstore[:,0] = SigmaX.flatten()
    
    
    thetahat= initial_theta
    thetahat_store = np.zeros((len(thetahat), maxIter))
    thetahat_store[:,0] = initial_theta
    SigmaX_R = initial_sigmax_R
    
    for k in range(1, maxIter):
        
        u = np.array([[input_[k]], [input_[k-1]]]) 
        
        theta_1, theta_2, theta_3, dummy = thetahat
        #lave A_k-1 og b_k-1
        
        A = np.array([[theta_1, 0], [0, 1]])
        B = np.array([[ theta_2, theta_3], [0,  eta*T / Q ]])
            
        #1a
        xhat = np.matmul(A, xhat) + np.matmul(B, u)
        
            
        SigmaX = np.matmul(np.matmul(A, SigmaX),A.T)  + SigmaW 
        
        
        #1c
        yhat = np.matmul(C_m, xhat) + a 

        
        yhatstore[k]=yhat
        
        
        #2a gain matrix fårst skal SigmaY regnes
        SigmaY = np.matmul(np.matmul(C_m, SigmaX), C_m.T) + SigmaV  #SigmaV = covarians matrixen af v-N(0,SigmaV), siden det er en skalar er covarians matrixen bare varains
        #Det for oven bliver in skalar fordi C_m^T er en vektor 
        
        L = np.matmul(SigmaX, C_m.T)/SigmaY  #(SigmaX *C^T) * SigmaY^-1
        
        #2b
        ytrue = measured_voltage[k]
        xhat  = xhat + L*(ytrue-yhat)
        
            
        # 2c
        SigmaX  = SigmaX  - SigmaY * np.matmul(L, L.T) # L * SigmaY * L^T = SigmaY L*T

        
        xhatstore[:,k] = xhat.T[0]
        SigmaXstore[:,k] = SigmaX.flatten()
        
        H = np.array([measured_voltage[k-1], inputs[k], inputs[k-1], 1])
        
        
        denominator = H.T @ SigmaX_R @ H + lam * sigmav_R
        L = (SigmaX_R @ H) / denominator

        # H er defineret som en vektor her, hvor imod at i bogen er det en row vector 
        
        thetahat = thetahat + L * ( measured_voltage[k] - np.dot(H, thetahat))
    
        thetahat_store[:,k] = thetahat

        #if k== 2:
        #   print(thetahat)
         #   print(thetahat_store)
        
        SigmaX_R = (1/lam) * ((np.eye(4) - np.outer(L, H)) @ SigmaX_R @ (np.eye(4) - np.outer(L, H)).T) + sigmav_R * np.outer(L, L) # np.outer(L, H) = L H^T (matrix outcome)
        # sigmav * np.outer(L, L) = L * V * L^T = matrix outcome  
        
    return xhatstore, SigmaXstore, yhatstore, thetahat_store



#########################################################
#                                                       #
#          simulation og parameter for simulation       #
#                                                       #
#########################################################

#først simuleres det true system

xtrue = np.array([[0],  # intial state, start spænding 0, start soc 0.7
                     [0.7]])

# Convert to numpy array for easier manipulation
inputs_clean = np.array(inputs)

measure_error=0.001# process error

noise = np.random.normal(0, measure_error, len(inputs_clean))
inputs_noise = inputs_clean + noise
    

MaxIter, xstore, y_NoNoise = Couloumb_Counting_true(inputs_noise, xtrue)

# "true system" y_NoNoise= measured voltage uden støj
# "true system" xstore[0:] = V_RRC, xstore[1:] = SOC

measure_error_2=0.001 #measurement error på målt output

noise = np.random.normal(0, measure_error_2, len(xstore[0, :]))

y_Noise = y_NoNoise + noise#ssumen measured data

#simulering ar algorithm

initial_SigmaX= 1*np.diag([1, 1])

#initial_SigmaX = np.ones((2,2))

SigmaW = np.eye(2) 
SigmaV = 1
soc=0.7

a2_R0, a1_R0, a0_R0 = bruh['R0']  
R = a0_R0 + a1_R0*soc + a2_R0*(soc**2)  

# For R1: same pattern
a2_R1, a1_R1, a0_R1 = bruh['R1']
R_1 = a0_R1 + a1_R1*soc + a2_R1*(soc**2)  

# For C1: same pattern
a2_C, a1_C, a0_C = bruh['C1']
C = a0_C + a1_C*soc + a2_C*(soc**2)  


#initial_theta= np.array([-1, 4.8, 4.8])

b_1= R*R_1*C
b_0 = R+R_1
a_0 = R_1*C
    

beta_1 = 2*b_1/T + b_0 
beta_0 = b_0 - 2*b_1/T
alph_1 = 2*a_0/T + 1
alph_0 = 1 - 2*a_0/T 


initial_theta= np.array([-alph_0/alph_1, beta_1/alph_1, beta_0/alph_1, ((alph_0/alph_1) *ocv_func(soc) + ocv_func(soc))])

#initial_theta = [-1, 4.8, 4.8]
#initial_sigmax_R=np.eye(4)*10000

initial_sigmax_R=0.0001*np.diag([1, 1, 1, 1]) 

sigmaV_R=measure_error_2
lamlam = 0.67
xhatstore, SigmaXstore , yhatstore, pe= Kalman_Filter_RLS(inputs, y_Noise, xtrue, initial_SigmaX, SigmaW, SigmaV, initial_theta, initial_sigmax_R, sigmaV_R, xstore, lamlam)




#plots
fig = plt.figure(figsize=(15, 10))
plt.plot(xhatstore[1, :], label='Estimated SOC (Kalman Filter)')
plt.plot(xstore[1, :], label='True SOC')
plt.ylabel('SOC')
plt.title(f'Kalman Filter SOC Estimation\nσ_w² = {measure_error}, σ_v² = {measure_error_2}, Forgetting Factor (λ): {lamlam}')
plt.legend()
plt.grid(True)
plt.show()


fig = plt.figure(figsize=(15, 10))
plt.plot(y_Noise)
plt.show()

# Create a figure with 2 rows (one for SOC, one for histogram and residuals)
fig = plt.figure(figsize=(15, 6))

# Calculate percentage deviation (in %)
residuals_percent = 100 * (xhatstore[1, 100:] - xstore[1, 100:])  # Convert to percentage


# Top plot - SOC estimation
ax1 = plt.subplot(2, 2, (1, 2))  # Top row spanning 2 columns
ax1.plot(xhatstore[1, :], label='Estimated SOC (Kalman Filter)')
ax1.plot(xstore[1, :], label='True SOC')
ax1.set_ylabel('SOC')
ax1.set_title(f'Kalman Filter SOC Estimation\nσ_w² = {measure_error}, σ_v² = {measure_error_2}, Forgetting Factor (λ): {lamlam}')
ax1.legend()
ax1.grid(True)


# Bottom left plot - Histogram of percentage deviations
ax2 = plt.subplot(2, 2, 3)
n, bins, patches = ax2.hist(residuals_percent, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')

# Add mean and std lines
mean_residual_percent = np.mean(residuals_percent)
std_residual_percent = np.std(residuals_percent)
ax2.axvline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('SOC Estimation Error (%)')
ax2.set_ylabel('Density')
ax2.set_title('Distribution of SOC Estimation Errors')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom right plot - Percentage deviation over time
ax3 = plt.subplot(2, 2, 4)
ax3.plot(residuals_percent, color='red', alpha=0.7, linewidth=0.5)
ax3.set_xlabel('Time step')
ax3.set_ylabel('SOC Error (%)')
ax3.set_title('SOC Estimation Errors Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Save as high-resolution PDF
plt.savefig('kalman_filter_soc_estimation.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.show()


# Calculate Durbin-Watson statistic using percentage deviations
def durbin_watson_test(residuals):
    e = residuals
    numerator = np.sum((e[1:]*e[:-1]))
    denominator = np.sum(e**2)
    d = numerator / denominator
    return d

residuals_percent = xhatstore[1, 400:-200] - xstore[1, 400:-200]
d_stat = durbin_watson_test(residuals_percent)
print(f"\nDurbin-Watson Statistic: {d_stat:.4f}")

# Interpretation
print("\nInterpretation of Durbin-Watson statistic:")
print(f"  d = {d_stat:.4f}")
if d_stat < 1.5:
    print("  → Positive autocorrelation (errors are correlated)")
elif d_stat > 2.5:
    print("  → Negative autocorrelation (errors are correlated)")
else:
    print("  → No significant autocorrelation (errors are independent)")
    