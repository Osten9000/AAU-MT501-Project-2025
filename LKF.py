#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 21:28:23 2025

@author: phillycheese
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
from OCV_GLM import a, b, ocv_func


from true_parameters import LKF_A, LKF_B, LKF_C

#################################
#                               #
#   constans and state model    #
#                               #
#################################

C = LKF_C #F
Q = 51.4*3600 # collumbs = aH * 3600
R_1 = LKF_B #ohm
R = LKF_A #ohm
T=1 # sampling time
eta= 0.9898

b_1= R*R_1*C
b_0 = R+R_1
a_0 = R_1*C

beta_1 = 2*b_1/T + b_0 
beta_0 = b_0 - 2*b_1/T
alph_1 = 2*a_0/T + 1
alph_0 = 1 - 2*a_0/T 

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


#State space model matrixes x_k = Ax_k + Bu_k

A = np.array([[-1 * alph_0/alph_1, 0], [0, 1]])

B = np.array([[ beta_1/alph_1, beta_0/alph_1], [eta*T / Q ,  0]])

# output space model equation, ved at bruge linear model (a,b), y=Cx + Du

C = np.array([[1, b]])

#################################
#                               #
#           inputs              #
#                               #
#################################

df = pd.read_csv('udds.csv')

#inputs = df['Normalized current (A)'].tolist()

inputs = [x * 200 for x in (-df['Normalized current (A)']).tolist()] #inputdata and change fortegn

#inputs = np.full(len(inputs), -0.30)

speed = df['Speed (km/h)'].tolist()


#################################
#                               #
#       define functions        #
#                               #
#################################
#definer funktioner, CC og linear kalmann filter

"""
Vi bruger CC til at berenge det "rigtige output", vi bruger cc til at regne soc,
og aflæser OCV fra kurven(ikke linear),
herefter bruger vi V(k) = V_RRC(k,i) + OCV(z[k]) som det rigtige output i LKF
"""
def Couloumb_Counting(input_, initial_x, OCV_data, SOC_data):
    maxIter = len(input_)
    x = initial_x # [V_RRC[0]  z[0]]^T
    #print(x.shape)
    xstore = np.zeros((len(x), maxIter)) # matix med len[x] rows og maxIter koloner
    xstore[:,0] = x.T[0]
    output_true  = np.zeros(maxIter) # "true" output i kalmann filter
    for k in range(1, maxIter):
        u = np.array([[input_[k]], [input_[k-1]]]) 
        
        x = np.matmul(A, x) + np.matmul(B, u)
        output_true[k] = x[0, 0]  + np.array(OCV_data[SOC_data == np.round(x[1, 0], 3)])[0] # vi bruger den ikke lineariseret output equation i state modellen og finder soc ved at kigge i dataen 
        if k==20:
            print(np.array(OCV_data[SOC_data == np.round(x[1, 0], 3)])[0] )
            print(x[0, 0])
        #output_true[k] = x[0, 0]  + ocv_func(x[1, 0])
        xstore[:,k] = x.T[0]
        
    return maxIter, xstore, output_true

#SigmaN, paremeteren covariacne matrixen af w, white gaussion noise, dim=(2,2). 

#SigmaV, parameteren varians for v, white gaussian noise dim=(1,1)

def Kalman_Filter(input_, measured_voltage, initial_x, initial_SigmaX, SigmaW, SigmaV):
    maxIter = len(input_)
    xhat = initial_x
    yhatstore = np.zeros(maxIter)
    SigmaX = initial_SigmaX
    xhatstore = np.zeros((len(xhat), maxIter))
    xhatstore[:,0] = xhat.T[0]
    SigmaXstore = np.zeros((len(xhat)**2, maxIter))
    SigmaXstore[:,0] = SigmaX.flatten()
    
    for k in range(1, maxIter):
        
        #input vector in state equatio
        u = np.array([[input_[k]], [input_[k-1]]]) 
        
        #1a
        xhat = np.matmul(A, xhat) + np.matmul(B, u)
        
        #1b
        SigmaX = np.matmul(np.matmul(A, SigmaX),A.T)  + SigmaW #E[ww^T] = SigmaV 

        #1c
        yhat = np.matmul(C, xhat) + a 
        
        yhatstore[k]=yhat
        
        #2a gain matrix fårst skal SigmaY regnes
        SigmaY = np.matmul(np.matmul(C, SigmaX), C.T) + SigmaV  #SigmaV = covarians matrixen af v-N(0,SigmaV), siden det er en skalar er covarians matrixen bare varains
        #Det for oven bliver in skalar fordi C^T er en vektor 
        
        L = np.matmul(SigmaX, C.T)/SigmaY  #(SigmaX *C^T) * SigmaY^-1
        
        if k==2:
            print(xhat)
        
        #2b
        ytrue = measured_voltage[k]
        xhat += L*(ytrue - yhat)
        # 2c
        SigmaX  = SigmaX  - SigmaY * np.matmul(L, L.T) # L * SigmaY * L^T = SigmaY L*T

        # [Store information for evaluation/plotting purposes]
        xhatstore[:,k] = xhat.T
        SigmaXstore[:,k] = SigmaX.flatten()
        
    return xhatstore, SigmaXstore, yhatstore



#########################################################
#                                                       #
#          simulation og parameter for simulation       #
#                                                       #
#########################################################
np.random.seed(67) 
    
xtrue = np.array([[0],  # intial state, start spænding 0, start soc 0.7
                     [0.7]])

# Convert to numpy array for easier manipulation
inputs_clean = np.array(inputs)

measure_error=0.000

noise = np.random.normal(0, measure_error, len(inputs_clean))

inputs_noise = inputs_clean + noise


    
maxIter, xstore, y_NoNoise = Couloumb_Counting(inputs_noise, xtrue, ocv_data, soc_data)

#x_1_1=xtrue[1,0] + np.random.normal(0,0.)

xtrue = np.array([[0],  # intial state, start spænding 0, start soc 0.7
                     [0.7]])
measure_error_2=0.000

noise = np.random.normal(0, measure_error_2, len(y_NoNoise ))

y_Noise = y_NoNoise + noise
#y_Noise = y_NoNoise 

initial_SigmaX= np.diag([1, 1])

#initial_SigmaX = np.ones((2,2))


SigmaW = np.eye(2) * 1
SigmaV = np.array([[1]])  

xhatstore, SigmaXstore , yhatstore= Kalman_Filter(inputs_clean, y_Noise, xtrue, initial_SigmaX, SigmaW, SigmaV)

########################
#                      #
#        Plots         #
#                      #
########################

# Calculate percentage deviation (in %)
residuals_percent = 100 * (xhatstore[1, :] - xstore[1, :])  # Convert to percentage

# Create a figure with 2 rows (one for SOC, one for histogram and residuals)
fig = plt.figure(figsize=(15, 10))

# Top plot - SOC estimation
ax1 = plt.subplot(2, 2, (1, 2))  # Top row spanning 2 columns
ax1.plot(xhatstore[1, :], label='Estimated SOC (Kalman Filter)')
ax1.plot(xstore[1, :], label='True SOC')
ax1.set_ylim([0.4, 1])
ax1.set_ylabel('SOC')
ax1.set_title(f'Kalman Filter SOC Estimation\nσ_w² = {measure_error}, σ_v² = {measure_error_2}')
ax1.legend()
ax1.grid(True)

# Calculate mean and std for residuals in percentage
mean_residual_percent = np.mean(residuals_percent)
std_residual_percent = np.std(residuals_percent)

# Bottom left plot - Histogram of percentage deviations
ax2 = plt.subplot(2, 2, 3)
n, bins, patches = ax2.hist(residuals_percent, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')

# Add mean and std lines
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
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save as high-resolution PDF
plt.savefig('4.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.show()

# Calculate and print statistics in percentage
print(f"\nSOC Error Statistics (in %):")
print(f"  Mean: {mean_residual_percent:.6f}%")
print(f"  Std:  {std_residual_percent:.6f}%")
print(f"  Min:  {np.min(residuals_percent):.6f}%")
print(f"  Max:  {np.max(residuals_percent):.6f}%")
print(f"  RMSE: {np.sqrt(np.mean(residuals_percent**2)):.6f}%")

# Calculate Durbin-Watson statistic using percentage deviations
def durbin_watson_test(residuals):
    e = residuals
    numerator = np.sum((e[1:]*e[:-1]))
    denominator = np.sum(e**2)
    d = numerator / denominator
    return d

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
    