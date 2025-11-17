# Project 2: Advanced MCMC Methods - Metropolis-Hastings and Gibbs Sampling

## Objective
Implement and compare advanced MCMC methods (Metropolis-Hastings and Gibbs sampling) for Bayesian inference on real financial data, extending beyond the rejection sampling used in Project 1.

## Requirements
1. **Metropolis-Hastings Algorithm**: Implement MH with symmetric proposal for Normal-Inverse Gamma model
2. **Gibbs Sampling**: Leverage conjugate priors for efficient sampling
3. **Real Data Analysis**: Use XRP/USDT returns or similar financial time series
4. **Data Assessment**: Check normality assumption using skewness, kurtosis, and visual plots
5. **Hypothesis Testing**: Compute Bayes factors for point null and interval hypotheses
6. **Diagnostics**: Include convergence analysis, trace plots, autocorrelation
7. **Comparison**: Compare efficiency and accuracy across MCMC methods

## Model Specification
- Likelihood: r_t ~ N(μ, σ²)
- Priors: μ ~ N(μ₀, σ₀²), σ² ~ InvGamma(α₀, β₀)
- Posterior inference for mean return and volatility parameters

## Deliverables
- Python implementations (mh_mcmc.py, gibbs_mcmc.py, bayes_factor.py, plots_mcmc.py)
- Comprehensive PDF report with methodology, results, code, and output
- Convergence diagnostics and efficiency analysis
- Interpretation of results in financial context
