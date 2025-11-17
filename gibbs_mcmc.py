import numpy as np
from scipy import stats
from typing import Tuple

class GibbsSampler:
    """Gibbs sampler for Normal-Inverse Gamma conjugate model"""
    
    def __init__(self, data: np.ndarray, mu0: float = 0.0, sigma0_sq: float = 1.0, 
                 alpha0: float = 2.0, beta0: float = 1.0):
        self.data = data
        self.n = len(data)
        self.data_mean = np.mean(data)
        self.data_sum = np.sum(data)
        self.sum_sq_data = np.sum(data**2)
        
        # Prior parameters
        self.mu0 = mu0
        self.sigma0_sq = sigma0_sq
        self.alpha0 = alpha0
        self.beta0 = beta0
        
        # Precompute posterior parameters that don't depend on current values
        self._compute_posterior_params()
    
    def _compute_posterior_params(self):
        """Precompute posterior hyperparameters"""
        # Updated alpha (doesn't change)
        self.alpha_n = self.alpha0 + self.n / 2
        
        # Terms for mu posterior
        self.precision_mu = 1/self.sigma0_sq + self.n  # Will be divided by current sigma_sq
    
    def sample_mu_given_sigma_sq(self, sigma_sq: float) -> float:
        """Sample mu from its full conditional: mu | sigma_sq, data"""
        # Posterior precision and mean for mu
        precision_n = 1/self.sigma0_sq + self.n/sigma_sq
        mu_n = (self.mu0/self.sigma0_sq + self.data_sum/sigma_sq) / precision_n
        variance_n = 1/precision_n
        
        return np.random.normal(mu_n, np.sqrt(variance_n))
    
    def sample_sigma_sq_given_mu(self, mu: float) -> float:
        """Sample sigma_sq from its full conditional: sigma_sq | mu, data"""
        # Updated beta parameter
        beta_n = self.beta0 + 0.5 * (self.sum_sq_data - 2*mu*self.data_sum + self.n*mu**2)
        
        # Sample from Inverse Gamma
        return stats.invgamma.rvs(a=self.alpha_n, scale=beta_n)
    
    def sample(self, n_samples: int = 10000, burn_in: int = 1000,
               initial_mu: float = None, initial_sigma_sq: float = None) -> dict:
        """Run Gibbs sampler"""
        
        samples_mu = np.zeros(n_samples)
        samples_sigma_sq = np.zeros(n_samples)
        
        # Initialize
        if initial_mu is None:
            current_mu = self.data_mean
        else:
            current_mu = initial_mu
            
        if initial_sigma_sq is None:
            current_sigma_sq = np.var(self.data, ddof=1)
        else:
            current_sigma_sq = initial_sigma_sq
        
        # Gibbs sampling
        for i in range(n_samples + burn_in):
            # Sample mu given sigma_sq
            current_mu = self.sample_mu_given_sigma_sq(current_sigma_sq)
            
            # Sample sigma_sq given mu
            current_sigma_sq = self.sample_sigma_sq_given_mu(current_mu)
            
            # Store samples (after burn-in)
            if i >= burn_in:
                idx = i - burn_in
                samples_mu[idx] = current_mu
                samples_sigma_sq[idx] = current_sigma_sq
        
        return {
            'mu_samples': samples_mu,
            'sigma_sq_samples': samples_sigma_sq,
            'n_samples': n_samples,
            'burn_in': burn_in
        }
    
    def posterior_predictive_sample(self, mu_samples: np.ndarray, 
                                  sigma_sq_samples: np.ndarray, 
                                  n_pred: int = 1) -> np.ndarray:
        """Generate posterior predictive samples"""
        n_posterior_samples = len(mu_samples)
        pred_samples = np.zeros((n_posterior_samples, n_pred))
        
        for i in range(n_posterior_samples):
            pred_samples[i, :] = np.random.normal(
                mu_samples[i], 
                np.sqrt(sigma_sq_samples[i]), 
                size=n_pred
            )
        
        return pred_samples.flatten() if n_pred == 1 else pred_samples
