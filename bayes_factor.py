import numpy as np
from scipy import stats
from scipy.special import loggamma
from typing import Tuple, Dict
import warnings

class BayesFactorAnalysis:
    """Bayes Factor calculations for hypothesis testing"""
    
    def __init__(self, data: np.ndarray, mu0: float = 0.0, sigma0_sq: float = 1.0,
                 alpha0: float = 2.0, beta0: float = 1.0):
        self.data = data
        self.n = len(data)
        self.data_mean = np.mean(data)
        self.data_var = np.var(data, ddof=1) if self.n > 1 else 1.0
        
        # Prior parameters
        self.mu0 = mu0
        self.sigma0_sq = sigma0_sq
        self.alpha0 = alpha0
        self.beta0 = beta0
    
    def marginal_likelihood_conjugate(self) -> float:
        """
        Compute marginal likelihood p(data) for Normal-Inverse Gamma conjugate model
        Uses analytical formula for conjugate priors
        """
        n = self.n
        
        # Posterior parameters
        sigma0_sq_n = 1 / (1/self.sigma0_sq + n)
        mu_n = sigma0_sq_n * (self.mu0/self.sigma0_sq + n*self.data_mean)
        alpha_n = self.alpha0 + n/2
        
        sum_sq = np.sum((self.data - self.data_mean)**2)
        beta_n = self.beta0 + 0.5 * sum_sq + \
                 0.5 * n * (self.data_mean - self.mu0)**2 / (1 + n*self.sigma0_sq)
        
        # Log marginal likelihood
        log_ml = -0.5 * n * np.log(2 * np.pi)
        log_ml += 0.5 * np.log(sigma0_sq_n / self.sigma0_sq)
        log_ml += loggamma(alpha_n) - loggamma(self.alpha0)
        log_ml += self.alpha0 * np.log(self.beta0) - alpha_n * np.log(beta_n)
        
        return log_ml
    
    def bayes_factor_point_null(self, mu_null: float = 0.0, 
                               sigma_sq_null: float = None) -> Dict:
        """
        Bayes Factor for point null hypothesis: H0: μ = μ_null
        BF01 = p(data | H0) / p(data | H1)
        """
        if sigma_sq_null is None:
            sigma_sq_null = self.data_var
        
        # Under H0: μ = μ_null, σ² known
        log_lik_h0 = -0.5 * self.n * np.log(2 * np.pi * sigma_sq_null)
        log_lik_h0 -= 0.5 * np.sum((self.data - mu_null)**2) / sigma_sq_null
        
        # Under H1: use marginal likelihood with conjugate priors
        log_ml_h1 = self.marginal_likelihood_conjugate()
        
        # Bayes Factor BF01
        log_bf01 = log_lik_h0 - log_ml_h1
        bf01 = np.exp(log_bf01)
        
        # Interpretation using Jeffreys' scale
        interpretation = self._interpret_bayes_factor(bf01, favor_null=True)
        
        return {
            'log_BF01': log_bf01,
            'BF01': bf01,
            'BF10': 1/bf01,
            'log_BF10': -log_bf01,
            'hypothesis': f'H0: μ = {mu_null}',
            'interpretation': interpretation,
            'evidence_for': 'H0 (null)' if bf01 > 1 else 'H1 (alternative)'
        }
    
    def bayes_factor_interval_null(self, interval: Tuple[float, float],
                                   mcmc_samples_mu: np.ndarray = None) -> Dict:
        """
        Bayes Factor for interval null hypothesis: H0: μ ∈ [a, b]
        Uses posterior samples from MCMC
        
        BF01 ≈ (posterior probability in interval) / (prior probability in interval)
        """
        a, b = interval
        
        if mcmc_samples_mu is None:
            raise ValueError("MCMC samples required for interval hypothesis testing")
        
        # Posterior probability that μ in [a, b]
        post_prob = np.mean((mcmc_samples_mu >= a) & (mcmc_samples_mu <= b))
        
        # Prior probability that μ in [a, b] (Normal prior)
        prior_prob = stats.norm.cdf(b, self.mu0, np.sqrt(self.sigma0_sq)) - \
                    stats.norm.cdf(a, self.mu0, np.sqrt(self.sigma0_sq))
        
        # Bayes Factor
        if prior_prob == 0:
            warnings.warn("Prior probability is zero - cannot compute Bayes factor")
            bf01 = np.inf if post_prob > 0 else np.nan
        else:
            bf01 = post_prob / prior_prob
        
        interpretation = self._interpret_bayes_factor(bf01, favor_null=True)
        
        return {
            'BF01': bf01,
            'BF10': 1/bf01 if bf01 > 0 else 0,
            'log_BF01': np.log(bf01) if bf01 > 0 else -np.inf,
            'posterior_prob': post_prob,
            'prior_prob': prior_prob,
            'hypothesis': f'H0: μ ∈ [{a:.4f}, {b:.4f}]',
            'interpretation': interpretation,
            'evidence_for': 'H0 (interval)' if bf01 > 1 else 'H1 (outside interval)'
        }
    
    def _interpret_bayes_factor(self, bf: float, favor_null: bool = True) -> str:
        """Interpret Bayes Factor using Jeffreys' scale (taught in class)"""
        if favor_null:
            if bf > 100:
                return "Decisive evidence for H0"
            elif bf > 30:
                return "Very strong evidence for H0"
            elif bf > 10:
                return "Strong evidence for H0"
            elif bf > 3:
                return "Substantial evidence for H0"
            elif bf > 1:
                return "Weak evidence for H0"
            elif bf > 1/3:
                return "Weak evidence for H1"
            elif bf > 1/10:
                return "Substantial evidence for H1"
            elif bf > 1/30:
                return "Strong evidence for H1"
            elif bf > 1/100:
                return "Very strong evidence for H1"
            else:
                return "Decisive evidence for H1"
        else:
            return self._interpret_bayes_factor(1/bf, favor_null=True)
    
    def assess_normality(self) -> Dict:
        """
        Assess normality using skewness and kurtosis (basic moments)
        These are simple concepts taught in basic statistics
        """
        results = {}
        
        # Check minimum sample size
        if self.n < 3:
            warnings.warn(f"Sample size ({self.n}) too small for normality assessment")
            results['summary'] = {
                'likely_normal': None,
                'recommendation': 'Sample size too small for assessment'
            }
            return results
        
        # Compute skewness and kurtosis
        try:
            skewness = stats.skew(self.data)
            excess_kurtosis = stats.kurtosis(self.data)  # Returns excess kurtosis
            
            # For normal distribution:
            # Skewness = 0 (symmetric)
            # Excess kurtosis = 0 (neither heavy nor light tails)
            
            skew_is_normal = abs(skewness) < 0.5  # Rule of thumb
            kurt_is_normal = abs(excess_kurtosis) < 1.0  # Rule of thumb
            
            results['moments'] = {
                'skewness': skewness,
                'excess_kurtosis': excess_kurtosis,
                'skew_is_normal': skew_is_normal,
                'kurtosis_is_normal': kurt_is_normal,
                'skewness_interpretation': self._interpret_skewness(skewness),
                'kurtosis_interpretation': self._interpret_kurtosis(excess_kurtosis)
            }
        except Exception as e:
            warnings.warn(f"Moment calculation failed: {e}")
        
        # Overall assessment
        if 'moments' in results:
            both_normal = results['moments']['skew_is_normal'] and results['moments']['kurtosis_is_normal']
            results['summary'] = {
                'likely_normal': both_normal,
                'recommendation': 'Data appears approximately normally distributed' if both_normal
                                else 'Data shows departure from normality'
            }
        else:
            results['summary'] = {
                'likely_normal': None,
                'recommendation': 'Could not assess normality'
            }
        
        return results
    
    def _interpret_skewness(self, skew: float) -> str:
        """Interpret skewness value"""
        if abs(skew) < 0.5:
            return "Approximately symmetric"
        elif skew > 0.5:
            return f"Right-skewed (tail on right side)"
        else:
            return f"Left-skewed (tail on left side)"
    
    def _interpret_kurtosis(self, kurt: float) -> str:
        """Interpret excess kurtosis value"""
        if abs(kurt) < 0.5:
            return "Normal-like tails"
        elif kurt > 0.5:
            return f"Heavy tails (more extreme values)"
        else:
            return f"Light tails (fewer extreme values)"