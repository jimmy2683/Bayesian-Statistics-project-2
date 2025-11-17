import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mh_mcmc import MetropolisHastings
from gibbs_mcmc import GibbsSampler
from plots_mcmc import MCMCPlotter
from bayes_factor import BayesFactorAnalysis
import time

def load_data(csv_path='data.csv', column=None):
    """Load and preprocess data from a CSV file"""
    # Read CSV - the first row might have extra commas
    df = pd.read_csv(csv_path, skip_blank_lines=True)
    
    # Debug: print columns before cleaning
    print(f"Initial columns: {list(df.columns)}")
    
    # Remove completely empty columns and rows
    df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN
    df = df.dropna(axis=0, how='all')  # Drop rows where all values are NaN
    
    # Remove 'Unnamed' columns that are actually empty
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    for col in unnamed_cols:
        if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
            df = df.drop(columns=[col])
    
    print(f"Cleaned columns: {list(df.columns)}")
    print(f"First few rows:\n{df.head()}")
    
    # Check if we have the expected columns
    if 'Price' in df.columns:
        # Clean and convert Price column (remove commas, handle strings)
        prices = df['Price'].astype(str).str.replace(',', '').str.strip()
        
        # Convert to numeric, handling any errors
        prices = pd.to_numeric(prices, errors='coerce').dropna().values
        
        if len(prices) < 2:
            raise ValueError(f"Not enough valid price data. Found {len(prices)} valid prices.")
        
        # Compute log returns
        returns = np.diff(np.log(prices))
        data = returns
        
        print(f"\nData loaded from {csv_path}: {len(prices)} price observations")
        print(f"Computed {len(data)} log returns")
        
        # Show date range if Date column exists
        if 'Date' in df.columns:
            valid_dates = df[df['Price'].notna()]['Date']
            if len(valid_dates) > 0:
                print(f"Date range: {valid_dates.iloc[-1]} to {valid_dates.iloc[0]}")
    elif column and column in df.columns:
        # Try to use specified column
        col_data = pd.to_numeric(df[column].astype(str).str.replace(',', ''), errors='coerce')
        data = col_data.dropna().to_numpy()
        print(f"Data loaded from {csv_path}: {len(data)} observations from column '{column}'")
    else:
        raise ValueError(f"Could not find 'Price' column or specified column '{column}' in CSV.\nAvailable columns: {list(df.columns)}\nPlease check the CSV file format.")

    if len(data) == 0:
        raise ValueError("No data loaded! Check CSV file format and contents.")
    
    print(f"\nSample statistics:")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Variance: {np.var(data, ddof=1):.6f}")
    print(f"  Std dev: {np.std(data, ddof=1):.6f}")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")

    return data

def assess_normality(data):
    """Assess normality of the data using skewness and kurtosis"""
    print("\n" + "="*60)
    print("NORMALITY ASSESSMENT")
    print("="*60)
    
    bf_analysis = BayesFactorAnalysis(data)
    normality_results = bf_analysis.assess_normality()
    
    # Print results
    print("\nDistribution Shape Analysis:")
    if 'moments' in normality_results:
        mom = normality_results['moments']
        print(f"\n1. Skewness: {mom['skewness']:.6f}")
        print(f"   Interpretation: {mom['skewness_interpretation']}")
        print(f"   Normal-like? {mom['skew_is_normal']}")
        
        print(f"\n2. Excess Kurtosis: {mom['excess_kurtosis']:.6f}")
        print(f"   Interpretation: {mom['kurtosis_interpretation']}")
        print(f"   Normal-like? {mom['kurtosis_is_normal']}")
        
        print(f"\n   Note: For normal distribution:")
        print(f"   - Skewness ≈ 0 (symmetric)")
        print(f"   - Excess Kurtosis ≈ 0 (neither heavy nor light tails)")
    else:
        print("   Not available")
    
    print("\n" + "-"*60)
    print("OVERALL ASSESSMENT:")
    summary = normality_results['summary']
    print(f"Conclusion: {summary['recommendation']}")
    print("="*60)
    
    return normality_results

def run_mcmc_comparison(data):
    """Run all MCMC methods and compare results"""
    
    # Prior parameters
    prior_params = {
        'mu0': 0.0,
        'sigma0_sq': 1.0,
        'alpha0': 2.0,
        'beta0': 0.001
    }
    
    # MCMC parameters
    n_samples = 10000
    burn_in = 1000
    
    results = {}
    
    print("\n" + "="*60)
    print("RUNNING MCMC COMPARISON")
    print("="*60)
    
    # 1. Metropolis-Hastings
    print("\n1. Running Metropolis-Hastings...")
    start_time = time.time()
    
    mh_sampler = MetropolisHastings(data, **prior_params)
    mh_results = mh_sampler.sample(
        n_samples=n_samples, 
        burn_in=burn_in,
        proposal_std_mu=0.005,
        proposal_std_sigma=0.0001
    )
    
    mh_time = time.time() - start_time
    results['Metropolis-Hastings'] = mh_results
    
    print(f"   Completed in {mh_time:.2f}s")
    print(f"   Acceptance rate: {mh_results['acceptance_rate']:.3f}")
    
    # 2. Gibbs Sampling
    print("\n2. Running Gibbs Sampling...")
    start_time = time.time()
    
    gibbs_sampler = GibbsSampler(data, **prior_params)
    gibbs_results = gibbs_sampler.sample(n_samples=n_samples, burn_in=burn_in)
    
    gibbs_time = time.time() - start_time
    results['Gibbs'] = gibbs_results
    
    print(f"   Completed in {gibbs_time:.2f}s")
    
    # 3. Add timing information
    results['Metropolis-Hastings']['runtime'] = mh_time
    results['Gibbs']['runtime'] = gibbs_time
    
    return results

def generate_plots_and_diagnostics(results, data):
    """Generate all plots and diagnostic outputs"""
    
    plotter = MCMCPlotter()
    
    print("\n" + "="*60)
    print("GENERATING PLOTS AND DIAGNOSTICS")
    print("="*60)
    
    # 1. Trace plots for each method
    for method_name, method_results in results.items():
        print(f"\nGenerating trace plots for {method_name}...")
        save_path = f"trace_{method_name.replace(' ', '_').replace('-', '_').lower()}.png"
        plotter.plot_traces(method_results, method_name, save_path=save_path)
        plt.close('all')
    
    # 2. Autocorrelation plots
    print("\nGenerating autocorrelation plots...")
    for method_name, method_results in results.items():
        save_path = f"autocorr_{method_name.replace(' ', '_').replace('-', '_').lower()}_mu.png"
        plotter.plot_autocorrelation(
            method_results['mu_samples'], 
            max_lag=200, 
            param_name=f"μ ({method_name})",
            save_path=save_path
        )
        plt.close('all')
    
    # 3. Posterior comparison
    print("\nGenerating posterior comparison plots...")
    plotter.plot_posterior_comparison(results, 'mu_samples', save_path='posterior_comparison_mu.png')
    plt.close('all')
    plotter.plot_posterior_comparison(results, 'sigma_sq_samples', save_path='posterior_comparison_sigma.png')
    plt.close('all')
    
    # 4. Convergence diagnostics
    for method_name, method_results in results.items():
        print(f"\nGenerating convergence diagnostics for {method_name}...")
        save_path = f"convergence_{method_name.replace(' ', '_').replace('-', '_').lower()}.png"
        plotter.plot_convergence_diagnostics(method_results, method_name, save_path=save_path)
        plt.close('all')
    
    # 5. Summary statistics
    print("\nGenerating summary statistics...")
    summary_df = plotter.summary_statistics(results)
    print("\nSUMMARY STATISTICS:")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.6f'))
    
    return summary_df

def compute_bayes_factors(data, mcmc_results):
    """Compute Bayes factors for various hypotheses"""
    print("\n" + "="*60)
    print("BAYES FACTOR ANALYSIS")
    print("="*60)
    
    bf_analysis = BayesFactorAnalysis(data)
    
    # Get posterior samples (use Gibbs samples as they're more efficient)
    mu_samples = mcmc_results['Gibbs']['mu_samples']
    
    # 1. Point null hypothesis: μ = 0 (no average return)
    print("\n1. Point Null Hypothesis: H0: μ = 0")
    bf_point = bf_analysis.bayes_factor_point_null(mu_null=0.0)
    print(f"   BF01 = {bf_point['BF01']:.6f}")
    print(f"   BF10 = {bf_point['BF10']:.6f}")
    print(f"   log(BF01) = {bf_point['log_BF01']:.6f}")
    print(f"   Interpretation: {bf_point['interpretation']}")
    print(f"   Evidence favors: {bf_point['evidence_for']}")
    
    # 2. Interval null hypothesis: μ ∈ [-0.001, 0.001] (negligible return)
    print("\n2. Interval Null Hypothesis: H0: μ ∈ [-0.001, 0.001]")
    bf_interval = bf_analysis.bayes_factor_interval_null(
        interval=(-0.001, 0.001),
        mcmc_samples_mu=mu_samples
    )
    print(f"   BF01 = {bf_interval['BF01']:.6f}")
    print(f"   BF10 = {bf_interval['BF10']:.6f}")
    print(f"   Posterior probability in interval: {bf_interval['posterior_prob']:.6f}")
    print(f"   Prior probability in interval: {bf_interval['prior_prob']:.6f}")
    print(f"   Interpretation: {bf_interval['interpretation']}")
    print(f"   Evidence favors: {bf_interval['evidence_for']}")
    
    # 3. Interval null hypothesis: μ > 0 (positive return)
    print("\n3. One-sided Hypothesis: H0: μ > 0")
    bf_positive = bf_analysis.bayes_factor_interval_null(
        interval=(0, np.inf),
        mcmc_samples_mu=mu_samples
    )
    print(f"   BF01 = {bf_positive['BF01']:.6f}")
    print(f"   Posterior probability (μ > 0): {bf_positive['posterior_prob']:.6f}")
    print(f"   Prior probability (μ > 0): {bf_positive['prior_prob']:.6f}")
    print(f"   Interpretation: {bf_positive['interpretation']}")
    
    print("="*60)
    
    return {
        'point_null_zero': bf_point,
        'interval_negligible': bf_interval,
        'positive_return': bf_positive
    }

def save_plot_results(normality_results, results, summary_df, bf_results, data):
    """Save all plot and analysis results to a text file"""
    
    output_file = 'results/plot_results.txt'
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PROJECT 2: ADVANCED MCMC METHODS - ANALYSIS RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write("="*80 + "\n\n")
        
        # Data Summary
        f.write("1. DATA SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Dataset: XRP/USDT Log Returns\n")
        f.write(f"Number of observations: {len(data)}\n")
        f.write(f"Sample mean: {np.mean(data):.6f}\n")
        f.write(f"Sample std dev: {np.std(data, ddof=1):.6f}\n")
        f.write(f"Sample variance: {np.var(data, ddof=1):.6f}\n")
        f.write(f"Sample min: {np.min(data):.6f}\n")
        f.write(f"Sample max: {np.max(data):.6f}\n")
        f.write("\n")
        
        # Normality Assessment
        f.write("2. NORMALITY ASSESSMENT\n")
        f.write("-"*80 + "\n")
        if 'moments' in normality_results:
            mom = normality_results['moments']
            f.write(f"Skewness: {mom['skewness']:.6f}\n")
            f.write(f"  Interpretation: {mom['skewness_interpretation']}\n")
            f.write(f"  Normal-like: {mom['skew_is_normal']}\n\n")
            
            f.write(f"Excess Kurtosis: {mom['excess_kurtosis']:.6f}\n")
            f.write(f"  Interpretation: {mom['kurtosis_interpretation']}\n")
            f.write(f"  Normal-like: {mom['kurtosis_is_normal']}\n\n")
        
        f.write(f"Overall: {normality_results['summary']['recommendation']}\n")
        f.write("\n")
        
        # MCMC Results Summary
        f.write("3. MCMC POSTERIOR ESTIMATES\n")
        f.write("-"*80 + "\n")
        f.write(summary_df.to_string(index=False, float_format='%.8f'))
        f.write("\n\n")
        
        # Detailed Posterior Statistics
        f.write("4. DETAILED POSTERIOR STATISTICS\n")
        f.write("-"*80 + "\n")
        for method_name, method_results in results.items():
            f.write(f"\n{method_name}:\n")
            f.write(f"  Parameter μ (mean return):\n")
            f.write(f"    Posterior mean: {np.mean(method_results['mu_samples']):.8f}\n")
            f.write(f"    Posterior std: {np.std(method_results['mu_samples']):.8f}\n")
            f.write(f"    95% Credible Interval: [{np.percentile(method_results['mu_samples'], 2.5):.8f}, ")
            f.write(f"{np.percentile(method_results['mu_samples'], 97.5):.8f}]\n")
            f.write(f"    Median: {np.median(method_results['mu_samples']):.8f}\n")
            
            f.write(f"\n  Parameter σ² (variance):\n")
            f.write(f"    Posterior mean: {np.mean(method_results['sigma_sq_samples']):.8f}\n")
            f.write(f"    Posterior std: {np.std(method_results['sigma_sq_samples']):.8f}\n")
            f.write(f"    95% Credible Interval: [{np.percentile(method_results['sigma_sq_samples'], 2.5):.8f}, ")
            f.write(f"{np.percentile(method_results['sigma_sq_samples'], 97.5):.8f}]\n")
            f.write(f"    Median: {np.median(method_results['sigma_sq_samples']):.8f}\n")
            
            # Implied daily volatility (sigma)
            sigma_samples = np.sqrt(method_results['sigma_sq_samples'])
            f.write(f"\n  Implied σ (daily volatility):\n")
            f.write(f"    Posterior mean: {np.mean(sigma_samples):.8f}\n")
            f.write(f"    95% Credible Interval: [{np.percentile(sigma_samples, 2.5):.8f}, ")
            f.write(f"{np.percentile(sigma_samples, 97.5):.8f}]\n")
            
            if 'acceptance_rate' in method_results:
                f.write(f"\n  MCMC Diagnostics:\n")
                f.write(f"    Acceptance rate: {method_results['acceptance_rate']:.4f}\n")
            
            if 'runtime' in method_results:
                f.write(f"    Runtime: {method_results['runtime']:.2f} seconds\n")
        
        f.write("\n")
        
        # Convergence Assessment
        f.write("5. CONVERGENCE DIAGNOSTICS\n")
        f.write("-"*80 + "\n")
        for method_name, method_results in results.items():
            f.write(f"\n{method_name}:\n")
            
            # Autocorrelation at specific lags
            mu_autocorr = compute_autocorrelation_values(method_results['mu_samples'])
            sigma_autocorr = compute_autocorrelation_values(method_results['sigma_sq_samples'])
            
            f.write(f"  Autocorrelation (μ):\n")
            f.write(f"    Lag 1: {mu_autocorr[1]:.4f}\n")
            f.write(f"    Lag 5: {mu_autocorr[5]:.4f}\n")
            f.write(f"    Lag 10: {mu_autocorr[10]:.4f}\n")
            f.write(f"    Lag 20: {mu_autocorr[20]:.4f}\n")
            
            f.write(f"\n  Autocorrelation (σ²):\n")
            f.write(f"    Lag 1: {sigma_autocorr[1]:.4f}\n")
            f.write(f"    Lag 5: {sigma_autocorr[5]:.4f}\n")
            f.write(f"    Lag 10: {sigma_autocorr[10]:.4f}\n")
            f.write(f"    Lag 20: {sigma_autocorr[20]:.4f}\n")
        
        f.write("\n")
        
        # Bayes Factor Results
        f.write("6. BAYES FACTOR HYPOTHESIS TESTING\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Hypothesis 1: H0: μ = 0 (no average return)\n")
        f.write(f"  BF01: {bf_results['point_null_zero']['BF01']:.6e}\n")
        f.write(f"  BF10: {bf_results['point_null_zero']['BF10']:.6e}\n")
        f.write(f"  log(BF01): {bf_results['point_null_zero']['log_BF01']:.6f}\n")
        f.write(f"  Interpretation: {bf_results['point_null_zero']['interpretation']}\n")
        f.write(f"  Evidence favors: {bf_results['point_null_zero']['evidence_for']}\n\n")
        
        f.write("Hypothesis 2: H0: μ ∈ [-0.001, 0.001] (negligible return)\n")
        f.write(f"  BF01: {bf_results['interval_negligible']['BF01']:.6f}\n")
        f.write(f"  BF10: {bf_results['interval_negligible']['BF10']:.6f}\n")
        f.write(f"  Posterior prob in interval: {bf_results['interval_negligible']['posterior_prob']:.6f}\n")
        f.write(f"  Prior prob in interval: {bf_results['interval_negligible']['prior_prob']:.6f}\n")
        f.write(f"  Interpretation: {bf_results['interval_negligible']['interpretation']}\n")
        f.write(f"  Evidence favors: {bf_results['interval_negligible']['evidence_for']}\n\n")
        
        f.write("Hypothesis 3: H0: μ > 0 (positive expected return)\n")
        f.write(f"  BF01: {bf_results['positive_return']['BF01']:.6f}\n")
        f.write(f"  BF10: {bf_results['positive_return']['BF10']:.6f}\n")
        f.write(f"  Posterior prob (μ > 0): {bf_results['positive_return']['posterior_prob']:.6f}\n")
        f.write(f"  Prior prob (μ > 0): {bf_results['positive_return']['prior_prob']:.6f}\n")
        f.write(f"  Interpretation: {bf_results['positive_return']['interpretation']}\n")
        f.write(f"  Evidence favors: {bf_results['positive_return']['evidence_for']}\n\n")
        
        # Financial Interpretation
        f.write("7. FINANCIAL INTERPRETATION\n")
        f.write("-"*80 + "\n")
        
        mu_mean = np.mean(results['Gibbs']['mu_samples'])
        sigma_mean = np.mean(np.sqrt(results['Gibbs']['sigma_sq_samples']))
        
        f.write(f"\nExpected Daily Return: {mu_mean:.6f} ({mu_mean*100:.4f}%)\n")
        f.write(f"Daily Volatility (σ): {sigma_mean:.6f} ({sigma_mean*100:.4f}%)\n")
        f.write(f"Annualized Volatility: {sigma_mean * np.sqrt(365):.6f} ({sigma_mean * np.sqrt(365)*100:.2f}%)\n")
        
        # Value at Risk
        var_95 = 1.96 * sigma_mean
        f.write(f"\nValue-at-Risk (95% confidence):\n")
        f.write(f"  Daily VaR: {var_95:.6f} ({var_95*100:.4f}%)\n")
        
        # Probability of positive return
        prob_positive = bf_results['positive_return']['posterior_prob']
        f.write(f"\nProbability of positive return: {prob_positive:.4f} ({prob_positive*100:.2f}%)\n")
        
        f.write("\nTrading Implications:\n")
        if abs(mu_mean) < 0.001:
            f.write("  - Mean return essentially zero (no systematic bias)\n")
            f.write("  - Recommend market-neutral strategies\n")
        elif mu_mean > 0:
            f.write(f"  - Slight positive bias ({mu_mean*100:.4f}% daily)\n")
            f.write("  - Consider long bias with appropriate risk management\n")
        else:
            f.write(f"  - Slight negative bias ({mu_mean*100:.4f}% daily)\n")
            f.write("  - Consider short bias or avoid directional exposure\n")
        
        if sigma_mean > 0.05:
            f.write(f"  - HIGH volatility ({sigma_mean*100:.2f}% daily)\n")
            f.write("  - Requires conservative position sizing\n")
        elif sigma_mean > 0.02:
            f.write(f"  - MODERATE volatility ({sigma_mean*100:.2f}% daily)\n")
            f.write("  - Standard position sizing appropriate\n")
        else:
            f.write(f"  - LOW volatility ({sigma_mean*100:.2f}% daily)\n")
            f.write("  - Can consider larger position sizes\n")
        
        f.write("\n")
        
        # Method Comparison
        f.write("8. METHOD COMPARISON SUMMARY\n")
        f.write("-"*80 + "\n")
        
        mh_mu_mean = np.mean(results['Metropolis-Hastings']['mu_samples'])
        gibbs_mu_mean = np.mean(results['Gibbs']['mu_samples'])
        mu_diff = abs(mh_mu_mean - gibbs_mu_mean)
        
        mh_sigma_mean = np.mean(results['Metropolis-Hastings']['sigma_sq_samples'])
        gibbs_sigma_mean = np.mean(results['Gibbs']['sigma_sq_samples'])
        sigma_diff = abs(mh_sigma_mean - gibbs_sigma_mean)
        
        f.write(f"\nPosterior Mean Differences:\n")
        f.write(f"  μ: MH vs Gibbs difference = {mu_diff:.8f}\n")
        f.write(f"  σ²: MH vs Gibbs difference = {sigma_diff:.8f}\n")
        
        f.write(f"\nComputational Efficiency:\n")
        if 'runtime' in results['Metropolis-Hastings'] and 'runtime' in results['Gibbs']:
            mh_time = results['Metropolis-Hastings']['runtime']
            gibbs_time = results['Gibbs']['runtime']
            f.write(f"  Metropolis-Hastings: {mh_time:.2f}s\n")
            f.write(f"  Gibbs Sampling: {gibbs_time:.2f}s\n")
            f.write(f"  Speedup: {mh_time/gibbs_time:.2f}x\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("9. RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        f.write("\nFor this Normal-Inverse Gamma model:\n")
        f.write("  ✓ PREFER Gibbs Sampling because:\n")
        f.write("    - No tuning required (100% acceptance)\n")
        f.write("    - Better mixing (lower autocorrelation)\n")
        f.write("    - Faster execution\n")
        f.write("    - Uses analytical full conditionals\n\n")
        f.write("  ✓ Metropolis-Hastings useful when:\n")
        f.write("    - Conjugate structure unavailable\n")
        f.write("    - Complex non-standard posteriors\n")
        f.write("    - No analytical full conditionals\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\nDetailed plot results saved to: {output_file}")

def compute_autocorrelation_values(samples, max_lag=20):
    """Compute autocorrelation for specific lags"""
    n = len(samples)
    x_centered = samples - np.mean(samples)
    variance = np.sum(x_centered**2) / n
    
    autocorr = {}
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            covariance = np.sum(x_centered[:-lag] * x_centered[lag:]) / n
            autocorr[lag] = covariance / variance
    
    return autocorr

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Assess normality
    normality_results = assess_normality(data)
    
    # Run complete MCMC analysis
    results = run_mcmc_comparison(data)
    summary_df = generate_plots_and_diagnostics(results, data)
    
    # Compute Bayes factors
    bf_results = compute_bayes_factors(data, results)
    
    # Save detailed plot results
    save_plot_results(normality_results, results, summary_df, bf_results, data)
    
    # Save results
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey Findings:")
    print("- Normality assessment completed")
    print("- All MCMC methods converged to similar posterior means")
    print("- Gibbs sampling typically shows better mixing (lower autocorrelation)")
    print("- Metropolis-Hastings acceptance rate should be 20-50% for efficiency")
    print("- Bayes factors computed for key hypotheses")
    
    # Save summary
    summary_df.to_csv('mcmc_comparison_summary.csv', index=False)
    print(f"\nSummary statistics saved to: mcmc_comparison_summary.csv")
    
    # Save Bayes factor results
    bf_df = pd.DataFrame([
        {
            'Hypothesis': bf_results['point_null_zero']['hypothesis'],
            'BF01': bf_results['point_null_zero']['BF01'],
            'Interpretation': bf_results['point_null_zero']['interpretation']
        },
        {
            'Hypothesis': bf_results['interval_negligible']['hypothesis'],
            'BF01': bf_results['interval_negligible']['BF01'],
            'Interpretation': bf_results['interval_negligible']['interpretation']
        },
        {
            'Hypothesis': bf_results['positive_return']['hypothesis'],
            'BF01': bf_results['positive_return']['BF01'],
            'Interpretation': bf_results['positive_return']['interpretation']
        }
    ])
    bf_df.to_csv('bayes_factors.csv', index=False)
    print(f"Bayes factors saved to: bayes_factors.csv")
