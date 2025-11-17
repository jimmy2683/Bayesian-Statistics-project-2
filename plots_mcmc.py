import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional
import pandas as pd

class MCMCPlotter:
    """Comprehensive plotting utilities for MCMC diagnostics and results"""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        
    def plot_traces(self, results: Dict, method_name: str = "MCMC", 
                   thin: int = 1, save_path: Optional[str] = None):
        """Plot trace plots for mu and sigma_sq"""
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        mu_samples = results['mu_samples'][::thin]
        sigma_sq_samples = results['sigma_sq_samples'][::thin]
        
        # Mu trace
        axes[0].plot(mu_samples, alpha=0.7, linewidth=0.5)
        axes[0].set_title(f'{method_name}: μ Trace Plot')
        axes[0].set_ylabel('μ')
        axes[0].grid(True, alpha=0.3)
        
        # Sigma squared trace  
        axes[1].plot(sigma_sq_samples, alpha=0.7, linewidth=0.5)
        axes[1].set_title(f'{method_name}: σ² Trace Plot')
        axes[1].set_ylabel('σ²')
        axes[1].set_xlabel('Iteration')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
    
    def plot_autocorrelation(self, samples: np.ndarray, max_lag: int = 100, 
                           param_name: str = "Parameter", save_path: Optional[str] = None):
        """Plot autocorrelation function - shows correlation between successive samples"""
        autocorr = self._autocorrelation(samples, max_lag)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(max_lag + 1), autocorr, 'b-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10% threshold')
        plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
        
        plt.title(f'Autocorrelation: {param_name}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim([-1.1, 1.1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
    
    def plot_posterior_comparison(self, results_dict: Dict[str, Dict], 
                                param: str = 'mu_samples', 
                                save_path: Optional[str] = None):
        """Compare posterior distributions across different methods"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        param_label = 'μ' if param == 'mu_samples' else 'σ²'
        
        # Density plots (histograms)
        for method_name, results in results_dict.items():
            samples = results[param]
            axes[0].hist(samples, bins=50, alpha=0.6, density=True, 
                        label=f'{method_name}')
        
        axes[0].set_title(f'Posterior Density Comparison: {param_label}')
        axes[0].set_xlabel(param_label)
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plots showing spread
        data_for_box = [results[param] for results in results_dict.values()]
        labels_for_box = list(results_dict.keys())
        
        axes[1].boxplot(data_for_box, labels=labels_for_box)
        axes[1].set_title(f'Posterior Distribution Comparison: {param_label}')
        axes[1].set_ylabel(param_label)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
    
    def plot_convergence_diagnostics(self, results: Dict, method_name: str = "MCMC",
                                   save_path: Optional[str] = None):
        """Convergence diagnostics: running averages and posterior densities"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        mu_samples = results['mu_samples']
        sigma_sq_samples = results['sigma_sq_samples']
        
        # Running averages - should stabilize if converged
        mu_running_mean = np.cumsum(mu_samples) / np.arange(1, len(mu_samples) + 1)
        sigma_running_mean = np.cumsum(sigma_sq_samples) / np.arange(1, len(sigma_sq_samples) + 1)
        
        axes[0, 0].plot(mu_running_mean)
        axes[0, 0].set_title(f'{method_name}: μ Running Average')
        axes[0, 0].set_ylabel('Running Mean of μ')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(sigma_running_mean)
        axes[0, 1].set_title(f'{method_name}: σ² Running Average')
        axes[0, 1].set_ylabel('Running Mean of σ²')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Posterior density plots (histograms)
        axes[1, 0].hist(mu_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title(f'{method_name}: μ Posterior Density')
        axes[1, 0].set_xlabel('μ')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(sigma_sq_samples, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title(f'{method_name}: σ² Posterior Density') 
        axes[1, 1].set_xlabel('σ²')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
    
    def _autocorrelation(self, x: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Compute sample autocorrelation function
        Measures correlation between samples at different lags
        """
        n = len(x)
        x_centered = x - np.mean(x)
        
        # Compute autocorrelation for each lag
        autocorr = np.zeros(max_lag + 1)
        variance = np.sum(x_centered**2) / n
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[lag] = 1.0
            else:
                covariance = np.sum(x_centered[:-lag] * x_centered[lag:]) / n
                autocorr[lag] = covariance / variance
        
        return autocorr
    
    def summary_statistics(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Generate summary statistics table"""
        summary_data = []
        
        for method_name, results in results_dict.items():
            mu_samples = results['mu_samples']
            sigma_sq_samples = results['sigma_sq_samples']
            
            row = {
                'Method': method_name,
                'μ_mean': np.mean(mu_samples),
                'μ_std': np.std(mu_samples),
                'μ_2.5%': np.percentile(mu_samples, 2.5),
                'μ_97.5%': np.percentile(mu_samples, 97.5),
                'σ²_mean': np.mean(sigma_sq_samples),
                'σ²_std': np.std(sigma_sq_samples), 
                'σ²_2.5%': np.percentile(sigma_sq_samples, 2.5),
                'σ²_97.5%': np.percentile(sigma_sq_samples, 97.5)
            }
            
            # Add method-specific metrics
            if 'acceptance_rate' in results:
                row['accept_rate'] = results['acceptance_rate']
            
            if 'runtime' in results:
                row['time_sec'] = results['runtime']
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
