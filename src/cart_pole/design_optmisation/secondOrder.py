import numpy as np
import time
from scipy.optimize import minimize
# Define the DesignDistribution class as provided


import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm

class DesignDistribution_log:
    def __init__(self, initial_mean, initial_cov, alpha_mean=0.001, alpha_cov=0.001):
        self.mean = np.array(initial_mean)
        self.cov = initial_cov + np.eye(initial_mean.shape[0]) * 1e-6  # Regularize once here
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov
        self.m_mean = np.zeros_like(self.mean)
        self.v_mean = np.zeros_like(self.mean)
        self.m_cov = np.zeros_like(self.cov)
        self.v_cov = np.zeros_like(self.cov)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def update_distribution(self, samples, rewards):
        self.t += 1

        samples = np.array(samples)
        rewards = np.array(rewards)
        n_samples = len(samples)

        # Avoid repeated computation
        inv_cov = np.linalg.inv(self.cov)

        grad_mean = np.sum(rewards[:, None] * inv_cov.dot((samples - self.mean).T).T, axis=0) / n_samples
        grad_cov = np.sum(
            rewards[:, None, None] * (-inv_cov + inv_cov.dot(np.einsum('ij,ik->ijk', samples - self.mean, samples - self.mean)).dot(inv_cov)),
            axis=0
        ) / n_samples

        # BFGS to update mean
        res = minimize(lambda x: -np.dot(grad_mean, x - self.mean), self.mean, method='BFGS')
        if res.success:
            self.mean = res.x

        # First-order Adam optimizer for covariance
        self.m_cov = self.beta1 * self.m_cov + (1 - self.beta1) * grad_cov
        self.v_cov = self.beta2 * self.v_cov + (1 - self.beta2) * grad_cov ** 2
        m_cov_corr = self.m_cov / (1 - self.beta1 ** self.t)
        v_cov_corr = self.v_cov / (1 - self.beta2 ** self.t)
        self.cov += self.alpha_cov * m_cov_corr / (np.sqrt(v_cov_corr) + self.epsilon)
        
        # Ensure positive-definiteness
        self.cov = (self.cov + self.cov.T) / 2 + np.eye(self.cov.shape[0]) * 1e-6

    def sample_design(self):
        return np.random.multivariate_normal(self.mean, self.cov)

# your other code remains the same

# Define the reward function as provided

def reward_function(sampled_design):
    # Ensure that the input is a NumPy array
    sampled_design = np.array(sampled_design)
    # Generate a reward value by summing several sine waves
    reward = -np.linalg.norm((sampled_design - np.array([0.10, 0.20, 0.30, 0.40]))*(sampled_design - np.array([0.80, 0.70, 0.30, 0.90]))) # This is a mock reward function

    # reward = (
    #     np.sin(2 * np.pi * sampled_design[0])
    #     + 0.5 * np.sin(4 * np.pi * sampled_design[1])
    #     + 0.25 * np.sin(8 * np.pi * sampled_design[2])
    #     - 0.5 * np.sin(3 * np.pi * sampled_design[3])
    # )
    return reward

num_distributions = 2**2
distributions = []
for i in range(num_distributions):
    # Initialize the mean with random values between 0 and 1 for each dimension
    initial_mean = np.random.rand(4)
    
    # Initialize the covariance matrix with random values for variance in each dimension
    # You might also wish to add random correlation terms (off-diagonal elements), but
    # remember that the resulting matrix must be positive-definite.
    initial_cov = np.diag(np.random.rand(4) * 0.5) + 0.001
    
    design_dist = DesignDistribution_log(initial_mean, initial_cov)
    distributions.append(design_dist)




# Simulate some robot training episodes
num_episodes = 10
batch_size = 1
num_iterations = 1000
evaluate_interval = 10

for iteration in range(num_iterations):
    for i in range(0, num_episodes, batch_size):
        for design_dist in distributions:
            batch_samples = []
            batch_rewards = []
            for _ in range(batch_size):
                sampled_design = design_dist.sample_design()
                reward = reward_function(sampled_design)
                batch_samples.append(sampled_design)
                batch_rewards.append(reward)
            design_dist.update_distribution(batch_samples, batch_rewards)
    # Evaluate and select the best distributions every evaluate_interval iterations
    if ((iteration + 1) % evaluate_interval == 0) and not(len(distributions) == 1):
        scores = []
        for design_dist in distributions:

            score = np.mean([reward_function(design_dist.sample_design()) for _ in range(100)])
            scores.append(score)
        sorted_indices = np.argsort(scores)[::-1]
        # Keep only half of the best performing distributions
        distributions = [distributions[idx] for idx in sorted_indices[:len(distributions) // 2]]
        # Print the iteration and the best performing design
    print(f"Iteration: {iteration}, Best Mean: {distributions[0].mean}, Score: {reward_function(distributions[0].mean)}")
    
    # If only the best distribution is left, break

