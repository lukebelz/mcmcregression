import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import itertools
import pickle
from server.data import *

def run_pymc_model(X, y, ols_model):
    # Check if 'const' (intercept) exists in the OLS model parameters
    if 'const' in ols_model.params:
        ols_intercept = ols_model.params['const']
    else:
        ols_intercept = 0  # Default intercept to 0 if no constant is present
    
    # Extract OLS coefficients, excluding intercept if it's missing
    ols_coeffs = ols_model.params.drop('const', errors='ignore')
    ols_residual_std = ols_model.resid.std()

    with pm.Model() as model:
        # Priors using OLS estimates or defaults
        intercept = pm.Normal('intercept', mu=ols_intercept, sigma=ols_residual_std)
        coeffs = [pm.Normal(f'beta_{var}', mu=ols_coeffs[var], sigma=ols_residual_std) for var in X.columns]
        sigma = pm.HalfNormal('sigma', sigma=ols_residual_std)

        # Linear model
        mu = intercept + sum([coeffs[i] * X.iloc[:, i] for i in range(len(coeffs))])

        # Likelihood
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

        # NUTS step
        step = pm.NUTS(target_accept=0.9, max_treedepth=12)

        # MCMC sampling
        trace = pm.sample(draws=1500, tune=1000, chains=4, cores=10, step=step, return_inferencedata=True)
    
        # Posterior predictive check (inside the context)
        post_pred = pm.sample_posterior_predictive(trace)

    # Dynamically check the available key in post_pred
    if 'Y_obs' in post_pred.posterior_predictive:
        y_pred_array = post_pred.posterior_predictive['Y_obs']  # Access posterior predictive data
    else:
        available_key = list(post_pred.posterior_predictive.keys())[0]
        y_pred_array = post_pred.posterior_predictive[available_key]

    # Compute the mean and std using xarray functions and extract the data with .values
    y_pred = y_pred_array.mean(dim=['chain', 'draw']).values  # Extract NumPy array
    y_pred_std = y_pred_array.std(dim=['chain', 'draw']).values  # Extract std as NumPy array

    # Ensure y_pred is a NumPy array for computation
    if isinstance(y_pred, np.ndarray):
        # Calculate SSE for the full model
        SSE_full = np.sum((y - y_pred) ** 2)  # Now y_pred is a NumPy array
    else:
        raise ValueError("y_pred is not a NumPy array")

    # Degrees of freedom for the full model
    df_full = len(y) - (len(coeffs) + 1)

    # Reduced model: Only with intercept
    y_mean = np.mean(y)
    SSE_reduced = np.sum((y - y_mean) ** 2)
    
    # Degrees of freedom for the reduced model
    df_reduced = len(y) - 1

    # Safeguard against division by zero or near-zero difference
    if np.isclose(SSE_reduced, SSE_full):
        F_stat = np.nan
        p_value = np.nan
    else:
        # Calculate the F-statistic
        F_stat = ((SSE_reduced - SSE_full) / (df_reduced - df_full)) / (SSE_full / df_full)

        # Calculate the p-value for the F-statistic
        p_value = 1 - stats.f.cdf(F_stat, df_reduced - df_full, df_full)

    # Calculate R-squared and Adjusted R-squared
    SS_total = np.sum((y - np.mean(y)) ** 2)
    R_squared = 1 - (SSE_full / SS_total)
    R_squared_adj = 1 - (1 - R_squared) * (len(y) - 1) / (len(y) - len(coeffs) - 1)

    return {
        'SSE': SSE_full,
        'R_squared': R_squared,
        'R_squared_adjusted': R_squared_adj,
        'p_value': p_value,
        'trace': trace
    }

def run_ols_model(X, y):
    X = sm.add_constant(X)  # Adds the intercept term
    model = sm.OLS(y, X).fit()
    
    # Calculate the overall p-value for the F-statistic (model significance)
    f_pvalue = model.f_pvalue  # Overall p-value of the F-statistic
    
    return {
        'SSE': np.sum(model.resid ** 2),
        'R_squared': model.rsquared,
        'R_squared_adjusted': model.rsquared_adj,
        'p_value': f_pvalue  # Return the overall p-value of the model
    }, model

###############
### Set Up ###
###############

# Define the independent variables
independent_vars = data_vars["independent_vars"]

# Define the dependent variable (target variable)
target_var = data_vars["target_var"]

# Load the dataset with specified columns
df = pd.read_csv(data_vars["file_name"], usecols=independent_vars + target_var)

# Clean the data: Handling NaN and inf values
# Replace infinite values with NaN and then drop rows with any NaN or inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Slice dataframe
df_limited = df.iloc[:100]

# Generate all possible combinations of independent variables
combinations = []
for i in range(1, len(independent_vars) + 1):
    combinations += list(itertools.combinations(independent_vars, i))

# Define the cleaned y (target variable)
y = df_limited[target_var[0]]

# Reset iteration counts
interation = 0
total_iterations = len(combinations)

###############
##### OLS #####
###############

results_ols = {}
ols_models = {}

# For each combination of independent variables, fit the OLS model
for combination in combinations:
    X = df_limited[list(combination)]  # Select the independent variables (already cleaned)
    sorted_combination = tuple(sorted(combination))  # Sort the tuple for consistent keys
    
    # Run the OLS model with the cleaned data
    results_ols[sorted_combination], ols_models[sorted_combination] = run_ols_model(X, y)
    
    interation += 1
    print(f'Finished iteration {interation} out of {total_iterations} of OLS')

# Reset iteration count for MCMC
interation = 0

###############
##### MCMC ####
###############

results_mcmc = {}

# For each combination of independent variables, fit the MCMC model
for combination in combinations:
    X = df_limited[list(combination)]  # Select the independent variables (already cleaned)
    sorted_combination = tuple(sorted(combination))  # Sort the tuple for consistent keys
    
    # Run the MCMC model with the OLS model as input
    results_mcmc[sorted_combination] = run_pymc_model(X, y, ols_models[sorted_combination])
    
    interation += 1
    print(f'Finished iteration {interation} out of {total_iterations} of MCMC')

###############
#### Export ###
###############

# Export the results to a pickle file
with open('model_results.pkl', 'wb') as f:
    pickle.dump({'results_mcmc': results_mcmc, 'results_ols': results_ols, 'combinations': combinations}, f)