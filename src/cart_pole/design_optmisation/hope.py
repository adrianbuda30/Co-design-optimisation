import numpy as np
import torch

class DesignDistribution:
    def __init__(self, initial_mean, initial_cov):
        self.mean = torch.tensor(initial_mean, dtype=torch.float32, requires_grad=True)
        self.log_cov = torch.tensor(np.log(initial_cov), dtype=torch.float32, requires_grad=True)

        self.optimizer = torch.optim.Adam([self.mean, self.log_cov], lr=0.01)

    def sample(self):
        cov = torch.exp(self.log_cov)
        design = torch.distributions.MultivariateNormal(self.mean, torch.diag(cov)).sample()
        return design

# Simulated reward function
def reward_function(design):
    return -torch.norm(design - torch.tensor([0.1, 0.2, 0.3, 0.4]))

# Initialize Design Distribution
initial_mean = [0.5, 0.5, 0.5, 0.5]
initial_cov = [0.1, 0.1, 0.1, 0.1]
design_distribution = DesignDistribution(initial_mean, initial_cov)

# Optimization loop
for iteration in range(100):
    design_distribution.optimizer.zero_grad()

    design = design_distribution.sample()
    reward = reward_function(design)

    # Negative reward because we're doing gradient ascent
    (-reward).backward()
    design_distribution.optimizer.step()

    # To see the progress
    print(f"Iteration: {iteration+1}, Design: {design.detach().numpy()}, Reward: {reward.item()}, Updated Mean: {design_distribution.mean.detach().numpy()}")

