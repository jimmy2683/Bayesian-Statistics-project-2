import numpy as np
from scipy import stats
from scipy.special import loggamma
from typing import Tuple, List
import warnings

class MetropolisHastings:
    """Metropolis-Hastings sampler for Normal-Inverse Gamma model"""
    
    def __init__(self, data: np.ndarray, mu0: float = 0.0, sigma0_sq: float = 1.0, 
                 alpha0: float = 2.0, beta0: float = 1.0):
        self.data = data
        self.n = len(data)
        self.sum_data = np.sum(data)
        self.sum_sq_data = np.sum(data**2)
        
        # Prior parameters
        self.mu0 = mu0
        self.sigma0_sq = sigma0_sq
        self.alpha0 = alpha0
        self.beta0 = beta0
    
    def log_likelihood(self, mu: float, sigma_sq: float) -> float:
        """Log likelihood for Normal model"""
        if sigma_sq <= 0:
            return -np.inf
        return -0.5 * self.n * np.log(2 * np.pi * sigma_sq) - \
               0.5 * (self.sum_sq_data - 2 * mu * self.sum_data + self.n * mu**2) / sigma_sq
    
    def log_prior(self, mu: float, sigma_sq: float) -> float:
        """Log prior for mu and sigma_sq"""
        if sigma_sq <= 0:
            return -np.inf
        
        # Log prior for mu
        log_prior_mu = -0.5 * np.log(2 * np.pi * self.sigma0_sq) - \
                       0.5 * (mu - self.mu0)**2 / self.sigma0_sq
        
        # Log prior for sigma_sq (Inverse Gamma)
        log_prior_sigma = self.alpha0 * np.log(self.beta0) - \
                         loggamma(self.alpha0) - \
                         (self.alpha0 + 1) * np.log(sigma_sq) - \
                         self.beta0 / sigma_sq
        
        return log_prior_mu + log_prior_sigma
    
    def log_posterior(self, mu: float, sigma_sq: float) -> float:
        """Log posterior (unnormalized)"""
        return self.log_likelihood(mu, sigma_sq) + self.log_prior(mu, sigma_sq)
    
    def propose(self, current_mu: float, current_sigma_sq: float, 
                proposal_std_mu: float = 0.1, proposal_std_sigma: float = 0.1) -> Tuple[float, float]:
        """Symmetric proposal for both parameters"""
        new_mu = current_mu + np.random.normal(0, proposal_std_mu)
        new_sigma_sq = current_sigma_sq + np.random.normal(0, proposal_std_sigma)
        return new_mu, max(new_sigma_sq, 1e-6)  # Ensure positive variance
    
    def sample(self, n_samples: int = 10000, burn_in: int = 1000, 
               initial_mu: float = 0.0, initial_sigma_sq: float = 1.0,
               proposal_std_mu: float = 0.1, proposal_std_sigma: float = 0.1) -> dict:
        """Run Metropolis-Hastings sampler"""
        
        samples_mu = np.zeros(n_samples)
        samples_sigma_sq = np.zeros(n_samples)
        accepted = 0
        
        current_mu = initial_mu
        current_sigma_sq = initial_sigma_sq
        current_log_post = self.log_posterior(current_mu, current_sigma_sq)
        
        for i in range(n_samples + burn_in):
            # Propose new state
            new_mu, new_sigma_sq = self.propose(current_mu, current_sigma_sq, 
                                              proposal_std_mu, proposal_std_sigma)
            
            # Calculate acceptance probability
            new_log_post = self.log_posterior(new_mu, new_sigma_sq)
            log_alpha = min(0, new_log_post - current_log_post)
            
            # Accept or reject
            if np.log(np.random.uniform()) < log_alpha:
                current_mu = new_mu
                current_sigma_sq = new_sigma_sq
                current_log_post = new_log_post
                if i >= burn_in:
                    accepted += 1
            
            # Store samples (after burn-in)
            if i >= burn_in:
                idx = i - burn_in
                samples_mu[idx] = current_mu
                samples_sigma_sq[idx] = current_sigma_sq
        
        acceptance_rate = accepted / n_samples
        
        return {
            'mu_samples': samples_mu,
            'sigma_sq_samples': samples_sigma_sq,
            'acceptance_rate': acceptance_rate,
            'n_samples': n_samples,
            'burn_in': burn_in
        }
