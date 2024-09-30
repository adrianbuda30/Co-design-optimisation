import torch
import torch.distributions as D
import numpy as np
import torch.optim as optim

class DesignDistribution_log:
    def __init__(self, initial_mean, initial_cov, lr_mean=0.01, lr_cov=0.01):
        self.mean = torch.tensor(initial_mean, dtype=torch.float32, requires_grad=True)
        self.cov = torch.tensor(initial_cov, dtype=torch.float32, requires_grad=True)
        
        self.optimizer_mean = optim.Adam([self.mean], lr=lr_mean)
        self.optimizer_cov = optim.Adam([self.cov], lr=lr_cov)

    def update_distribution(self, batch_rewards, batch_samples):

        mean_rewards = torch.mean(torch.tensor(batch_rewards, dtype=torch.float32))

        grad_mean = torch.zeros_like(self.mean)
        grad_cov = torch.zeros_like(self.cov)

        for i in range(len(batch_rewards)):
            self.optimizer_mean.zero_grad()
            self.optimizer_cov.zero_grad()

            sample = torch.tensor(batch_samples[i], dtype=torch.float32)
            neg_log_likelihood = D.MultivariateNormal(self.mean, self.cov).log_prob(sample)
            neg_log_likelihood.backward(retain_graph=True)

            grad_mean += self.mean.grad
            grad_cov += self.cov.grad

        grad_mean /= len(batch_rewards)
        grad_cov /= len(batch_rewards)

        self.mean.grad = grad_mean * mean_rewards
        self.cov.grad = grad_cov * mean_rewards

        self.optimizer_mean.step()
        self.optimizer_cov.step()


    def sample_design(self):
        return D.MultivariateNormal(self.mean, self.cov).sample()

def reward_function(sampled_design):
    reward = torch.norm((sampled_design - torch.tensor([1.0, 2.0, 3, 4.0], dtype=torch.float32)))
    return reward.item()

initial_mean = 2.5 * np.random.rand(4).astype(np.float32)
initial_cov = np.eye(4, dtype=np.float32) * 5 + 5

design_dist = DesignDistribution_log(initial_mean, initial_cov)

num_episodes = 100000
batch_size = 1

for i in range(0, num_episodes, batch_size):
    batch_rewards = []
    batch_samples = []

    for _ in range(batch_size):
        sampled_design = design_dist.sample_design().detach()
        reward = reward_function(sampled_design)
        
        batch_samples.append(sampled_design.numpy())
        batch_rewards.append(reward)

    design_dist.update_distribution(batch_rewards, batch_samples)
    
    print(f"mean: {design_dist.mean} reward: {np.mean(batch_rewards)}")
