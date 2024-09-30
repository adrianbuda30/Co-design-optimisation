import numpy as np

class DesignDistribution:
    def __init__(self, initial_mean, initial_std, alpha_mean=0.1, alpha_std=0.1):
        self.mean = initial_mean
        self.std = initial_std
        self.alpha_mean = alpha_mean
        self.alpha_std = alpha_std
        self.m_mean = 0
        self.v_mean = 0
        self.m_std = 0
        self.v_std = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def sample_design(self):
        return self.mean + self.std * np.random.normal()

    def update_distribution(self, samples, rewards):
        self.t += 1
        samples = np.array(samples)
        rewards = np.array(rewards)
        
        # Calculate gradients
        grad_mean = np.mean(rewards * (samples - self.mean) / self.std**2)
        grad_std = np.mean(rewards * (-1/self.std + (samples - self.mean)**2/self.std**3))
        
        # Adam optimizer
        self.m_mean = self.beta1 * self.m_mean + (1 - self.beta1) * grad_mean
        self.v_mean = self.beta2 * self.v_mean + (1 - self.beta2) * grad_mean**2
        m_mean_corr = self.m_mean / (1 - self.beta1**self.t)
        v_mean_corr = self.v_mean / (1 - self.beta2**self.t)
        self.mean += self.alpha_mean * m_mean_corr / (np.sqrt(v_mean_corr) + self.epsilon)

        self.m_std = self.beta1 * self.m_std + (1 - self.beta1) * grad_std
        self.v_std = self.beta2 * self.v_std + (1 - self.beta2) * grad_std**2
        m_std_corr = self.m_std / (1 - self.beta1**self.t)
        v_std_corr = self.v_std / (1 - self.beta2**self.t)
        self.std += self.alpha_std * m_std_corr / (np.sqrt(v_std_corr) + self.epsilon)

# Example usage
design_dist = DesignDistribution(10, 5)

# Simulate some robot training episodes
num_episodes = 100
batch_size = 1
all_samples = []
all_rewards = []

for i in range(0, num_episodes, batch_size):
    batch_samples = []
    batch_rewards = []
    for _ in range(batch_size):
        sampled_design = design_dist.sample_design()
        reward = -abs((sampled_design - 35)**2) # This is a mock reward function
        batch_samples.append(sampled_design)
        batch_rewards.append(reward)
    design_dist.update_distribution(batch_samples, batch_rewards)
    print(f"Updated Mean: {design_dist.mean}, Updated Std: {design_dist.std}")
