# ADVANCED MCMC METHODS FOR FINANCIAL TIME SERIES ANALYSIS


- Project 2 - MA4740: Bayesian Statistics
- Under the guidance of Prof. Arunabha Majumdar
- Indian Institute of Technology, Hyderabad

## PROJECT OVERVIEW

This project implements and compares two fundamental Markov Chain Monte Carlo 
(MCMC) methods—Metropolis-Hastings and Gibbs Sampling—for Bayesian inference 
on cryptocurrency financial time series data. We analyze XRP/USDT log returns 
using a Normal-Inverse Gamma conjugate prior model, with comprehensive 
convergence diagnostics and hypothesis testing via Bayes factors.

KEY HIGHLIGHTS:
- From-scratch implementation of two MCMC algorithms
- 10,000 posterior samples with 1,000 burn-in iterations
- Comprehensive convergence diagnostics (trace plots, autocorrelation, running averages)
- Bayes factor hypothesis testing using Jeffreys' scale
- Financial interpretation with risk metrics and trading implications
- ~2,100 data points of real cryptocurrency returns

## OBJECTIVES


1. Implement Metropolis-Hastings with symmetric random walk proposals
2. Implement Gibbs Sampling leveraging conjugate prior structure
3. Assess normality using skewness and kurtosis
4. Compute Bayes factors for point null and interval hypotheses
5. Compare methods in terms of convergence, mixing, and computational efficiency
6. Interpret results in financial context (volatility, VaR, trading strategies)

## MODEL SPECIFICATION


HIERARCHICAL MODEL:

Likelihood:  r_t | μ, σ² ~ N(μ, σ²),  t = 1, ..., n
Priors:      μ ~ N(μ₀ = 0, σ₀² = 1)
             σ² ~ InvGamma(α₀ = 2, β₀ = 0.001)

DATA:
- Asset: XRP/USDT cryptocurrency pair
- Observations: 2,098 daily log returns
- Transformation: r_t = log(P_t / P_{t-1})
- Sample Mean: -0.001283
- Sample Volatility: 5.44% daily (103.96% annualized)



## INSTALLATION & SETUP


PREREQUISITES:
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Pandas

INSTALL DEPENDENCIES:

### Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install numpy scipy matplotlib seaborn pandas


## QUICK START


RUN COMPLETE ANALYSIS:

./run.sh

This script will:
1. Check dependencies
2. Run MCMC samplers (Metropolis-Hastings and Gibbs)
3. Generate diagnostic plots
4. Compute Bayes factors
5. Save results and summaries

MANUAL EXECUTION:

python main_analysis.py

### View results
cat results/plot_results.txt


## PROJECT STRUCTURE

```
.
├── LICENSE
├── README.md
├── bayes_factor.py
├── data.csv
├── figures
│   ├── autocorr_gibbs_mu.png
│   ├── autocorr_metropolis_hastings_mu.png
│   ├── convergence_gibbs.png
│   ├── convergence_metropolis_hastings.png
│   ├── posterior_comparison_mu.png
│   ├── posterior_comparison_sigma.png
│   ├── qq_plot_normality.png
│   ├── trace_gibbs.png
│   └── trace_metropolis_hastings.png
├── gibbs_mcmc.py
├── logs
│   └── analysis_20251117_221649.log
├── main_analysis.py
├── mh_mcmc.py
├── plots_mcmc.py
├── problem.md
├── report.pdf
├── results
│   ├── analysis_summary.txt
│   ├── bayes_factors.csv
│   ├── mcmc_comparison_summary.csv
│   └── plot_results.txt
└── run.sh
```
## KEY RESULTS


POSTERIOR ESTIMATES (GIBBS SAMPLING):

Parameter          | Mean      | Std. Dev.  | 95% Credible Interval
-------------------|-----------|------------|----------------------
μ (mean return)    | -0.00125  | 0.00119    | [-0.00357, 0.00110]
σ² (variance)      | 0.00296   | 0.000093   | [0.00278, 0.00315]
σ (daily volatility)| 5.44%     | -          | [5.28%, 5.61%]

BAYES FACTOR RESULTS:

Hypothesis                    | BF₀₁           | Interpretation
------------------------------|----------------|--------------------------------
H₀: μ = 0                     | 6.49 × 10¹²⁴   | Decisive evidence for H₀
H₀: μ ∈ [-0.001, 0.001]       | 480.77         | Decisive evidence for H₀
H₀: μ > 0                     | 0.2916         | Substantial evidence against H₀

CONCLUSION: No systematic directional bias; mean return is statistically zero.

METHOD COMPARISON:

Metric                  | Metropolis-Hastings | Gibbs Sampling
------------------------|---------------------|----------------
Runtime                 | 0.25s               | 0.68s
Acceptance Rate         | 91.01%              | 100%
Autocorrelation (lag 1) | 0.974               | 0.007
Autocorrelation (lag 10)| 0.773               | 0.014
Mixing Quality          | Poor                | Excellent

WINNER: Gibbs sampling (55× better mixing despite 2.7× slower per iteration)


## FINANCIAL INTERPRETATION


RISK METRICS:
- Daily Volatility: 5.44% (very high)
- Annualized Volatility: 103.96%
- Value-at-Risk (95%): 10.67% daily loss
- Probability of Positive Return: 14.58%

TRADING IMPLICATIONS:

NOT RECOMMENDED: Directional long/short strategies
- No evidence of positive expected return (BF decisively supports μ = 0)
- Only 14.58% probability of positive daily return

RECOMMENDED: Volatility-based strategies
- Options strategies (straddles, strangles)
- Market-neutral approaches
- Mean-reversion on extreme moves (>2σ)

POSITION SIZING:
- High Risk Asset: Requires conservative allocation
- Recommended: 5-10% of portfolio maximum
- Use strict stop-losses: ±8-10% levels
- Heavy tails warning: Normal model likely underestimates tail risk (kurtosis = 19.67)


## DIAGNOSTICS & VALIDATION


NORMALITY ASSESSMENT:
- Skewness: -0.52 (moderate left skew)
- Excess Kurtosis: 19.67 (extremely heavy tails)
- Conclusion: Significant departure from normality

WARNING: VaR and extreme event probabilities likely underestimated

CONVERGENCE CHECKS:
✓ Trace Plots: Chains explore parameter space without getting stuck
✓ Running Averages: Stabilize after burn-in
✓ Autocorrelation: Gibbs shows near-zero, MH shows high values
✓ Posterior Agreement: Both methods converge to same distribution


## METHODOLOGY


METROPOLIS-HASTINGS ALGORITHM:

Algorithm: Random walk with symmetric proposal

1. Initialize θ⁽⁰⁾ = (μ⁽⁰⁾, σ²⁽⁰⁾)
2. For i = 1, ..., N:
   - Propose θ* ~ q(θ* | θ⁽ⁱ⁻¹⁾) using Normal random walk
   - Compute acceptance probability: α = min(1, p(θ*|data) / p(θ⁽ⁱ⁻¹⁾|data))
   - Accept θ⁽ⁱ⁾ = θ* with probability α, else θ⁽ⁱ⁾ = θ⁽ⁱ⁻¹⁾

Proposal Distributions:
- μ* = μ⁽ⁱ⁻¹⁾ + ε_μ, where ε_μ ~ N(0, 0.005²)
- σ²* = σ²⁽ⁱ⁻¹⁾ + ε_σ, where ε_σ ~ N(0, 0.0001²)

Tuning: Proposal variance adjusted to achieve 20-50% acceptance rate

GIBBS SAMPLING ALGORITHM:

Algorithm: Alternate sampling from full conditionals

1. Initialize σ²⁽⁰⁾
2. For i = 1, ..., N:
   - Sample μ⁽ⁱ⁾ ~ p(μ | σ²⁽ⁱ⁻¹⁾, data)
   - Sample σ²⁽ⁱ⁾ ~ p(σ² | μ⁽ⁱ⁾, data)

Full Conditional Distributions:

μ | σ², data ~ N(μₙ, τₙ⁻¹) where:
- τₙ = 1/σ₀² + n/σ²
- μₙ = τₙ⁻¹(μ₀/σ₀² + n·r̄/σ²)

σ² | μ, data ~ InvGamma(αₙ, βₙ) where:
- αₙ = α₀ + n/2
- βₙ = β₀ + (1/2)Σ(rₜ - μ)²

Advantage: No tuning required, 100% acceptance rate


## OUTPUT FILES


NUMERICAL RESULTS:

results/plot_results.txt - Complete analysis report including:
- Data summary statistics
- Normality assessment (skewness, kurtosis)
- Posterior estimates with credible intervals
- Autocorrelation at multiple lags
- Bayes factor hypothesis tests
- Financial interpretation
- Method comparison
- Recommendations

results/mcmc_comparison_summary.csv - Posterior summary table

results/bayes_factors.csv - Hypothesis testing results

DIAGNOSTIC PLOTS:

Trace Plots: trace_metropolis_hastings.png, trace_gibbs.png
- Show MCMC chain exploration over iterations
- Good mixing = rapid exploration without getting stuck

Autocorrelation Plots: autocorr_*_mu.png
- Measure correlation between successive samples
- Lower autocorrelation = more independent samples

Convergence Diagnostics: convergence_*.png
- Running averages should stabilize
- Posterior densities should be smooth

Posterior Comparison: posterior_comparison_*.png
- Overlay distributions from both methods
- Should overlap if both converged correctly


## THEORY & BACKGROUND


WHY MCMC?

Direct sampling from posterior p(θ|data) is often intractable. MCMC methods 
construct a Markov chain whose stationary distribution is the posterior, 
allowing us to:
1. Generate samples from complex distributions
2. Compute posterior expectations via Monte Carlo averaging
3. Obtain credible intervals via empirical quantiles

WHEN TO USE WHICH METHOD?

Use Gibbs Sampling when:
- Full conditional distributions are available
- Conjugate priors exist
- You want guaranteed acceptance
- No tuning desired

Use Metropolis-Hastings when:
- Conjugate structure unavailable
- Working with non-standard distributions
- Full conditionals cannot be sampled directly
- Need more general-purpose algorithm

CONVERGENCE DIAGNOSTICS EXPLAINED:

Trace Plots: Visual check for stationarity
- Good: Fuzzy caterpillar, no trends
- Bad: Drifting, stuck states, slow exploration

Autocorrelation: Measures sample dependence
- ACF should decay to ~0 quickly
- High autocorrelation = many correlated samples needed

Running Average: Cumulative mean over iterations
- Should stabilize after burn-in
- Indicates convergence to posterior mean


## EXTENSIONS & FUTURE WORK


MODEL IMPROVEMENTS:

1. Heavy-Tailed Distributions:
   - Replace Normal with Student-t likelihood
   - Better captures extreme events (kurtosis = 19.67)
   - More realistic for cryptocurrency returns

2. GARCH Models:
   - Time-varying volatility
   - Captures volatility clustering
   - Models σₜ² as function of past returns

3. Hierarchical Models:
   - Multiple assets simultaneously
   - Correlation structure
   - Asset-specific and common factors

METHODOLOGICAL EXTENSIONS:

1. Advanced MCMC:
   - Hamiltonian Monte Carlo (HMC)
   - No-U-Turn Sampler (NUTS)
   - Adaptive MCMC for automatic tuning

2. Parallel Chains:
   - Multiple chains from different starting points
   - Gelman-Rubin diagnostic
   - Increased confidence in convergence

3. Model Selection:
   - Compare Normal vs. Student-t via Bayes factors
   - DIC (Deviance Information Criterion)
   - WAIC (Widely Applicable Information Criterion)


## TROUBLESHOOTING

COMMON ISSUES:

Low Acceptance Rate (MH < 20%):
- Decrease proposal variance
- Proposals too aggressive
- Chain rejecting too often

High Acceptance Rate (MH > 50%):
- Increase proposal variance
- Proposals too conservative
- Not exploring efficiently

High Autocorrelation:
- Increase thinning (keep every k-th sample)
- Run longer chains
- Try Gibbs if applicable

Non-Convergence:
- Increase burn-in period
- Check for multimodal posterior
- Verify prior specification
- Check data preprocessing

RUNNING ON DIFFERENT DATA:

To analyze your own data:

1. Replace data.csv with your price data
2. Ensure column named "Price" exists
3. Run: python main_analysis.py

To modify priors:

# In main_analysis.py
prior_params = {
    'mu0': 0.0,        # Prior mean for μ
    'sigma0_sq': 1.0,  # Prior variance for μ
    'alpha0': 2.0,     # InvGamma shape
    'beta0': 0.001     # InvGamma scale
}


## IMPLEMENTATION DETAILS


FILE: mh_mcmc.py
- MetropolisHastings class
- Log-likelihood computation
- Log-prior computation (Normal-Inverse Gamma)
- Symmetric proposal mechanism
- Accept/reject step
- Returns: samples, acceptance rate

FILE: gibbs_mcmc.py
- GibbsSampler class
- Full conditional for μ | σ², data (Normal)
- Full conditional for σ² | μ, data (Inverse Gamma)
- Alternating sampling
- No rejection step (always accepts)
- Returns: samples only

FILE: bayes_factor.py
- BayesFactorAnalysis class
- Marginal likelihood computation
- Point null hypothesis testing
- Interval hypothesis testing
- Skewness and kurtosis calculation
- Jeffreys' scale interpretation

FILE: plots_mcmc.py
- MCMCPlotter class
- Trace plot generation
- Autocorrelation function plots
- Convergence diagnostics
- Posterior comparison plots
- Summary statistics table

FILE: main_analysis.py
- Data loading from CSV
- Normality assessment
- MCMC execution (both methods)
- Plot generation
- Bayes factor computation
- Results export to CSV and TXT


## COMPUTATIONAL PARAMETERS


MCMC SETTINGS:
- Number of samples: 10,000
- Burn-in period: 1,000 iterations
- Total iterations: 11,000 per method

METROPOLIS-HASTINGS:
- Proposal std for μ: 0.005
- Proposal std for σ²: 0.0001
- Target acceptance: 20-50%
- Achieved: 91.01% (too high, could use larger steps)

GIBBS SAMPLING:
- No tuning parameters required
- Acceptance rate: 100% (always)
- Uses analytical full conditionals

PRIORS:
- μ₀ = 0 (neutral prior)
- σ₀² = 1 (weakly informative)
- α₀ = 2 (shape parameter)
- β₀ = 0.001 (scale parameter)


## AUTHOR


Student Name: Karan Gupta
Roll Number: CS23BTECH11023
Course: MA4740 - Bayesian Statistics
Instructor: Prof. Arunabha Majumdar
Institution: Indian Institute of Technology, Hyderabad
Date: November 2025


## LICENSE

This project is for academic purposes as part of MA4740 coursework.


## ACKNOWLEDGMENTS


- Prof. Arunabha Majumdar for course instruction and guidance
- IIT Hyderabad Department of Mathematics
- Cryptocurrency price data sources


## CONTACT


For questions about this project:
- Email: [cs23btech11023@iith.ac.in]

Last updated: November 17, 2025

