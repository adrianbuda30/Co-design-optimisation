import numpy as np
import torch
import torch.optim as optim

def sophisticated_reward_function(sample):
    target = np.array([10, 20, 30, 40])
    reward = np.linalg.norm(sample - target) ** 2 
    print(reward)
    return reward

class DesignDistribution:
    def __init__(self, initial_mean, initial_cov):
        self.mean = torch.tensor(initial_mean, dtype=torch.float32, requires_grad=True)
        self.cov = torch.tensor(initial_cov, dtype=torch.float32, requires_grad=True)
        self.optimizer = optim.Adam([self.mean, self.cov], lr=0.01)

    def sample_design(self):
        return np.random.multivariate_normal(self.mean.detach().numpy(), self.cov.detach().numpy())

    def update_distribution(self, samples, rewards):
        samples = torch.tensor(samples, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        self.optimizer.zero_grad()
        
        # Compute the loss. Note that we're assuming a diagonal covariance matrix here.
        loss = -torch.mean(torch.sum(((samples - self.mean) ** 2) / torch.diag(self.cov), dim=1) * rewards)
        
        loss.backward()
        self.optimizer.step()

# Initialize design distribution
initial_mean = np.array([0.0, 0.0, 0.0, 0.0])
initial_cov = np.diag([1.0, 1.0, 1.0, 1.0])

design_dist = DesignDistribution(initial_mean, initial_cov)

# Training settings
num_episodes = 1000
batch_size = 10

# Training loop
for episode in range(num_episodes):
    batch_samples = []
    batch_rewards = []
    
    for _ in range(batch_size):
        sample = design_dist.sample_design()
        reward = sophisticated_reward_function(sample)
        
        batch_samples.append(sample)
        batch_rewards.append(reward)
    
    design_dist.update_distribution(np.array(batch_samples), np.array(batch_rewards))

    # print(f"After episode {np.mean(batch_rewards)}, updated mean: {design_dist.mean.detach().numpy()}, updated covariance diagonal: {torch.diag(design_dist.cov).detach().numpy()}")
