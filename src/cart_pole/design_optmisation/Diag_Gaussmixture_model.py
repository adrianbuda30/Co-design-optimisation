import numpy as np
import torch

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

class DesignDistribution_torch:
    def __init__(self, initial_mean, initial_var, alpha_mean=0.01, alpha_var=0.01, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = torch.tensor(initial_mean, dtype=torch.float64, requires_grad=True)
        self.var = torch.tensor(initial_var, dtype=torch.float64, requires_grad=True)
        self.optimizer_mean = torch.optim.Adam([self.mean], lr=alpha_mean)
        self.optimizer_var = torch.optim.Adam([self.var], lr=alpha_var)

        self.optimizer = torch.optim.Adam([self.mean, self.var], lr=0.0001)
        self.min_values = torch.tensor(min_values, dtype=torch.float64)
        self.max_values = torch.tensor(max_values, dtype=torch.float64)
        
    def _is_within_bounds(self, sample):
        return torch.all(sample >= self.min_values) and torch.all(sample <= self.max_values)

    def sample_design(self):
        with torch.no_grad():  # Sampling should not affect gradients
            sample = self.mean + torch.sqrt(self.var) * torch.randn_like(self.mean)
        return sample.numpy()

    def update_distribution(self, samples, rewards):
        samples = torch.tensor(samples, dtype=torch.float64)
        rewards = torch.tensor(rewards, dtype=torch.float64)

        # Compute gradients with respect to mean
        loss_mean = torch.mean(rewards * torch.sum((torch.log(samples) - torch.log(self.mean)) / torch.log(self.var), dim=1))

        # Update mean
        self.optimizer_mean.zero_grad()  # Zero the gradients for the mean optimizer
        loss_mean.backward(retain_graph=True)  # Compute the gradients for mean
        self.optimizer_mean.step()  # Update the mean parameters

        # Compute gradients with respect to variance
        loss_var = torch.mean(rewards * torch.sum(((samples - self.mean)**2) / self.var - 1, dim=1))

        # Update variance
        self.optimizer_var.zero_grad()  # Zero the gradients for the variance optimizer
        loss_var.backward()  # Compute the gradients for variance
        self.optimizer_var.step()  # Update the variance parameters

        # Ensure that the variance remains non-negative
        with torch.no_grad():
            self.var.clamp_(min=1e-6)



def reward_function(sampled_design):
    # Ensure that the input is a NumPy array
    # Generate a reward value by summing several sine waves
    reward = -np.linalg.norm((sampled_design - np.array([0.10, 0.20, 0.30, 0.40]))*(sampled_design - np.array([0.80, 0.70, 0.30, 0.90]))) # This is a mock reward function
    return reward

# Initialize the 10 multi-variate distributions
num_distributions = 1
distributions = []
for i in range(num_distributions):
    initial_mean = np.array([0.1, 0.3, 0.5, 0.7])
    initial_cov = np.array([0.5, 0.5, 0.5, 0.5])
    design_dist = DesignDistribution_torch(initial_mean, initial_cov)
    distributions.append(design_dist)



# Simulate some robot training episodes
num_episodes = 1
batch_size = 1
num_iterations = 100000
evaluate_interval = 1000000

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
                print(design_dist.mean.detach().numpy(),design_dist.var.detach().numpy())
            design_dist.update_distribution(batch_samples, batch_rewards)