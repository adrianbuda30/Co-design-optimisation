import numpy as np

class DesignDistribution:
    def __init__(self, initial_mean, initial_std):
        self.mean = initial_mean
        self.std = initial_std

    def sample_design(self):
        return self.mean + self.std * np.random.normal()

    def update_distribution(self, samples, rewards):
        alpha_mean = 0.1
        alpha_std = 0.1
        
        # Calculate gradients
        grad_mean = rewards * (samples - self.mean) / self.std**2
        grad_std =  rewards * (-1/self.std + (samples - self.mean)**2/self.std**3)

        print(f"Mean Gradient: {grad_mean}, Std Gradient: {grad_std}")

        # Update mean and std
        self.mean += alpha_mean * grad_mean
        self.std += alpha_std * grad_std

# Example usage
design_dist = DesignDistribution(10, 2)

# Simulate some robot training episodes
num_episodes = 1000
all_samples = []
all_rewards = []

for _ in range(num_episodes):
    sampled_design = design_dist.sample_design()
    reward = -abs((sampled_design - 19)**3) # This is a mock reward function

    # Update the design distribution based on feedback
    design_dist.update_distribution(sampled_design, reward)
    print(f"Updated Mean: {design_dist.mean}, Updated Std: {design_dist.std}")
