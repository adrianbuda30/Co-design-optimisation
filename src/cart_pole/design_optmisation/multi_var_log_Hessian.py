import numpy as np
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

class DesignDistribution:
    def __init__(self, initial_mean, initial_cov):
        self.mean = np.array(initial_mean, dtype=np.float64)
        self.cov = initial_cov
        self.L = np.linalg.cholesky(self.cov)
        self.tril_indices = np.tril_indices(self.mean.shape[0])
        
    def residuals(self, params, samples, rewards):
        d = self.mean.shape[0]
        mean = params[:d]
        L = np.zeros((d, d))
        L[self.tril_indices] = params[d:]
        cov = L @ L.T

        log_likelihood = multivariate_normal.logpdf(samples, mean=mean, cov=cov)
        residuals = -rewards * log_likelihood
        return residuals

    def update_distribution(self, samples, rewards):
        d = self.mean.shape[0]
        initial_params = np.concatenate([self.mean, self.L[self.tril_indices]])
        rewards = np.array(rewards)  # Convert rewards to a NumPy array
        result = least_squares(self.residuals, initial_params, args=(samples, rewards), method='lm', verbose=0)

        updated_params = result.x
        self.mean = updated_params[:d]
        self.L = np.zeros((d, d))
        self.L[self.tril_indices] = updated_params[d:]
        self.cov = self.L @ self.L.T


    def sample_design(self):
        return np.random.multivariate_normal(self.mean, self.cov)
# Example usage
initial_mean = [10, 10, 10, 10]
initial_cov = np.eye(4) * 100 + 0.01
design_dist = DesignDistribution(initial_mean, initial_cov)

# Simulate some robot training episodes
num_episodes = 1000000
batch_size = 20
all_samples = []
all_rewards = []

tolerance = 1e-3
old_mean = np.array(initial_mean)

for i in range(0, num_episodes, batch_size):
    batch_samples = []
    batch_rewards = []
    for _ in range(batch_size):
        sampled_design = design_dist.sample_design()
        reward = -np.linalg.norm(sampled_design - np.array([10, 20, 30, 40]))**2
        batch_samples.append(sampled_design)
        batch_rewards.append(reward)
    design_dist.update_distribution(batch_samples, batch_rewards)
    print(f"Updated Mean: {design_dist.mean}")
    
    # Check for convergence
    mean_change = np.linalg.norm(design_dist.mean - old_mean)
    if mean_change < tolerance:
        break
    old_mean = np.copy(design_dist.mean)
