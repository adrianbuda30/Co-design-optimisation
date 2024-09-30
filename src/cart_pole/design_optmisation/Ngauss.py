import numpy as np
import time
# Define the DesignDistribution class as provided

class DesignDistribution:
    def __init__(self, initial_mean, initial_cov, alpha_mean=0.01, alpha_cov=0.01, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = np.array(initial_mean, dtype=np.float64)  # Change to float64
        self.cov = initial_cov
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
        self.min_values = np.array(min_values)
        self.max_values = np.array(max_values)


    def _is_within_bounds(self, sample):
        return np.all(sample >= self.min_values) and np.all(sample <= self.max_values)
    
    def sample_design(self):
        sample = np.random.multivariate_normal(self.mean, self.cov)
        iter = 0
        while not self._is_within_bounds(sample):
            sample = np.random.multivariate_normal(self.mean, self.cov)
            iter += 1
            if iter > 10:
                    if(np.all(sample >= self.max_values)):
                        sample = self.max_values
                        return sample
                    elif(np.all(sample <= self.min_values)):
                        sample = self.min_values
                        return sample 
        return sample

    def update_distribution(self, samples, rewards):
        self.t += 1

        samples = np.array(samples)

        # print(np.mean(samples, axis=0))

        rewards = np.array(rewards)
        
        grad_mean = np.mean(rewards[:, np.newaxis] * (samples - self.mean) / np.diagonal(self.cov), axis=0)
        grad_cov = np.mean(rewards[:, np.newaxis, np.newaxis] * 
                          (np.einsum('ij,ik->ijk', samples - self.mean, samples - self.mean) / self.cov - np.eye(self.mean.shape[0])), 
                          axis=0)
        
        # Adam optimizer for mean
        self.m_mean = self.beta1 * self.m_mean + (1 - self.beta1) * grad_mean
        self.v_mean = self.beta2 * self.v_mean + (1 - self.beta2) * grad_mean**2
        m_mean_corr = self.m_mean / (1 - self.beta1**self.t)
        v_mean_corr = self.v_mean / (1 - self.beta2**self.t)
        self.mean += self.alpha_mean * m_mean_corr / (np.sqrt(v_mean_corr) + self.epsilon)

        # Adam optimizer for covariance
        self.m_cov = self.beta1 * self.m_cov + (1 - self.beta1) * grad_cov
        self.v_cov = self.beta2 * self.v_cov + (1 - self.beta2) * grad_cov**2
        m_cov_corr = self.m_cov / (1 - self.beta1**self.t)
        v_cov_corr = self.v_cov / (1 - self.beta2**self.t)
        self.cov += self.alpha_cov * m_cov_corr / (np.sqrt(v_cov_corr) + self.epsilon)
        self.cov += np.eye(self.mean.shape[0]) * 1e-6  # Add small positive value to the diagonal

class DesignDistribution_log:
    def __init__(self, initial_mean, initial_cov, alpha_mean=0.001, alpha_cov=0.001, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = np.array(initial_mean, dtype=np.float64)  # Change to float64
        self.cov = initial_cov
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
        self.min_values = np.array(min_values)
        self.max_values = np.array(max_values)

    def update_distribution(self, samples, rewards):
        self.t += 1

        samples = np.array(samples)
        rewards = np.array(rewards)
        n_samples = len(samples)
        d = self.mean.shape[0]  # Dimension of the data

        # Compute the gradients of the log-likelihood with respect to the mean and covariance
        grad_mean = np.zeros_like(self.mean)
        grad_cov = np.zeros_like(self.cov)

        for i in range(n_samples):
            x_minus_mean = samples[i] - self.mean
            inv_cov = np.linalg.inv(self.cov)
            grad_mean += rewards[i] * inv_cov.dot(x_minus_mean)
            grad_cov += rewards[i] * (-inv_cov + inv_cov.dot(np.outer(x_minus_mean, x_minus_mean)).dot(inv_cov))

        # Normalize the gradients by the number of samples
        grad_mean /= n_samples
        grad_cov /= n_samples

        # grad_mean = -grad_mean
        # grad_cov = -grad_cov
        # Adam optimizer for mean
        self.m_mean = self.beta1 * self.m_mean + (1 - self.beta1) * grad_mean
        self.v_mean = self.beta2 * self.v_mean + (1 - self.beta2) * grad_mean**2
        m_mean_corr = self.m_mean / (1 - self.beta1**self.t)
        v_mean_corr = self.v_mean / (1 - self.beta2**self.t)
        self.mean += self.alpha_mean * m_mean_corr / (np.sqrt(v_mean_corr) + self.epsilon)

        # Adam optimizer for covariance
        self.m_cov = self.beta1 * self.m_cov + (1 - self.beta1) * grad_cov
        self.v_cov = self.beta2 * self.v_cov + (1 - self.beta2) * grad_cov**2
        m_cov_corr = self.m_cov / (1 - self.beta1**self.t)
        v_cov_corr = self.v_cov / (1 - self.beta2**self.t)
        self.cov += self.alpha_cov * m_cov_corr / (np.sqrt(v_cov_corr) + self.epsilon)
        self.cov += np.eye(self.mean.shape[0]) * 1e-6  # Add small positive value to the diagonal

    def _is_within_bounds(self, sample):
        return np.all(sample >= self.min_values) and np.all(sample <= self.max_values)
    
    def sample_design(self):
        sample = np.random.multivariate_normal(self.mean, self.cov)
        while not self._is_within_bounds(sample):
            sample = np.random.multivariate_normal(self.mean, self.cov)
        return sample
# Define the reward function as provided


class DesignDistribution_diagGauss:
    def __init__(self, initial_mean, initial_var, alpha_mean=0.01, alpha_var=0.01, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = np.array(initial_mean, dtype=np.float64)  
        self.var = np.array(initial_var, dtype=np.float64)  # Variance, not Covariance
        self.alpha_mean = alpha_mean
        self.alpha_var = alpha_var
        self.m_mean = np.zeros_like(self.mean)
        self.v_mean = np.zeros_like(self.mean)
        self.m_var = np.zeros_like(self.var)
        self.v_var = np.zeros_like(self.var)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.min_values = np.array(min_values)
        self.max_values = np.array(max_values)

    def _is_within_bounds(self, sample):
        return np.all(sample >= self.min_values) and np.all(sample <= self.max_values)

    def sample_design(self):
        sample = self.mean + np.sqrt(self.var) * np.random.randn(len(self.mean))
        iter = 0
        while not self._is_within_bounds(sample):
            sample = self.mean + np.sqrt(self.var) * np.random.randn(len(self.mean))
            iter += 1
            if iter > 10:
                if(np.all(sample >= self.max_values)):
                    sample = self.max_values
                    return sample
                elif(np.all(sample <= self.min_values)):
                    sample = self.min_values
                    return sample
        return sample

    def update_distribution(self, samples, rewards):
        self.t += 1

        samples = np.array(samples)
        rewards = np.array(rewards)
        
        grad_mean = np.mean(rewards[:, np.newaxis] * (samples - self.mean) / self.var, axis=0)
        grad_var = np.mean(rewards[:, np.newaxis] * ((samples - self.mean)**2 / self.var - 1), axis=0)
        
        # Adam optimizer for mean
        self.m_mean = self.beta1 * self.m_mean + (1 - self.beta1) * grad_mean
        self.v_mean = self.beta2 * self.v_mean + (1 - self.beta2) * grad_mean**2
        m_mean_corr = self.m_mean / (1 - self.beta1**self.t)
        v_mean_corr = self.v_mean / (1 - self.beta2**self.t)
        self.mean += self.alpha_mean * m_mean_corr / (np.sqrt(v_mean_corr) + self.epsilon)

        # Adam optimizer for variance
        self.m_var = self.beta1 * self.m_var + (1 - self.beta1) * grad_var
        self.v_var = self.beta2 * self.v_var + (1 - self.beta2) * grad_var**2
        m_var_corr = self.m_var / (1 - self.beta1**self.t)
        v_var_corr = self.v_var / (1 - self.beta2**self.t)
        self.var += self.alpha_var * m_var_corr / (np.sqrt(v_var_corr) + self.epsilon)
        self.var += 1e-6  # Add small positive value to keep variance non-negative



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

# Initialize the 10 multi-variate distributions
num_distributions = 2**5
distributions = []
for i in range(num_distributions):
    initial_mean = np.array([0.5, 0.5, 0.5, 0.5])
    initial_cov = np.array([0.5, 0.5, 0.5, 0.5])
    design_dist = DesignDistribution_diagGauss(initial_mean, initial_cov)
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

