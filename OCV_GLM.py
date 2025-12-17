import pandas as pd, matplotlib.pyplot as plt, numpy as np, scipy.stats as stats

# Read the CSV file
df = pd.read_csv('OCV_curve.csv')
#df = pd.read_csv('50Ah_NMC_ocv_curve.csv')

plott = True

soc = df['SOC']
ocv = df['OCV [V]']

soc_f = soc
ocv_f = ocv

min_soc = 0.3
max_soc = 0.7

mask = (soc >= min_soc) & (soc <= max_soc)
soc = soc[mask]
ocv = ocv[mask]

X = np.column_stack([
    np.ones(len(soc)),  # First column: all 1's
    soc.values          # Second column: SOC values
])

# β̂ = (XᵀX)⁻¹XᵀY
X_transpose = X.T
X_transpose_X = X_transpose @ X  # Matrix multiplication
X_transpose_X_inv = np.linalg.inv(X_transpose_X)  # Inverse
X_transpose_Y = X_transpose @ ocv

# β̂ = [ a  b ]^T
beta_mle = X_transpose_X_inv @ X_transpose_Y

a, b = beta_mle  

# Calculate R-squared for the linear model
# Predicted values
y_pred = a + b * soc

# Calculate R-squared
ss_res = np.sum((ocv - y_pred) ** 2)  # Residual sum of squares
ss_tot = np.sum((ocv - np.mean(ocv)) ** 2)  # Total sum of squares
r_squared = 1 - (ss_res / ss_tot)

print(f"Linear model: y = {a:.4f} + {b:.4f} * x")
print(f"R-squared value: {r_squared:.6f}")

if plott == True:
    plt.xlim(-0.03, 1.03)          
    plt.ylim(2.5, 4.5) 
    
    plt.scatter(soc_f, ocv_f, alpha=0.6, color='blue', label='Actual OCV data', s=10)
    plt.plot(soc, a + b * soc, color='red', linewidth=2)
    plt.legend()
    plt.xlabel('SOC')
    plt.ylabel('OCV [V]')
    plt.title(' GLM for OCV Data')
    plt.savefig('5.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def fit_ocv_polynomial_coefficients(soc, ocv, degree=8):
    """
    Fit an nth-order polynomial to OCV-SOC data using numpy's polyfit.
    
    Parameters:
    -----------
    soc : array-like
        State of Charge values (0-1 or 0-100)
    ocv : array-like
        Open Circuit Voltage values
    degree : int, optional
        Degree of polynomial (default: 8)
    
    Returns:
    --------
    coefficients : numpy.ndarray
        Polynomial coefficients from highest to lowest degree
    """
    # Convert to numpy arrays if not already
    soc_array = np.array(soc)
    ocv_array = np.array(ocv)
    
    # Fit polynomial using numpy's polyfit
    coefficients = np.polyfit(soc_array, ocv_array, degree)
    
    return coefficients

# Usage with your data
ocv_coefficients = fit_ocv_polynomial_coefficients(soc, ocv, degree=8)
ocv_func = np.poly1d(ocv_coefficients)

# Optional: Calculate R-squared for the polynomial model as well
y_pred_poly = ocv_func(soc)
ss_res_poly = np.sum((ocv - y_pred_poly) ** 2)
r_squared_poly = 1 - (ss_res_poly / ss_tot)
print(f"\nPolynomial model (degree 8) R-squared value: {r_squared_poly:.6f}")
# Calculate residuals from the linear model
residuals = ocv - y_pred  # Actual - Predicted

# Create QQ plot for residuals
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals (Linear Model)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True, alpha=0.3)
plt.savefig('QQ.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

# Also print some statistics about residuals
print(f"\nResidual Statistics:")
print(f"Mean of residuals: {np.mean(residuals):.6f}")
print(f"Standard deviation of residuals: {np.std(residuals):.6f}")
print(f"Skewness: {stats.skew(residuals):.6f}")
print(f"Kurtosis: {stats.kurtosis(residuals):.6f}")
# At the end of OCV_GLM.py, add:
__all__ = ['a', 'b', 'ocv_func']
