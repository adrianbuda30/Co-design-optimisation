import torch
import torch.distributions as D
import numpy as np
import torch.optim as optim

class DesignDistribution_log:
    def __init__(self, initial_mean, initial_d, initial_u, initial_v, lr_mean=1, lr_d=1, lr_uv=1):
        self.mean = torch.tensor(initial_mean, requires_grad=True)
        self.D = torch.tensor(initial_d, requires_grad=True)
        self.U = torch.tensor(initial_u, requires_grad=True)
        self.V = torch.tensor(initial_v, requires_grad=True)

        self.optimizer_mean = optim.Adam([self.mean], lr=lr_mean)
        self.optimizer_d = optim.Adam([self.D], lr=lr_d)
        self.optimizer_uv = optim.Adam([self.U, self.V], lr=lr_uv)

    def update_distribution(self, batch_rewards, batch_samples):
        mean_rewards = torch.mean(torch.tensor(batch_rewards))
        self.optimizer_mean.zero_grad()
        self.optimizer_d.zero_grad()
        self.optimizer_uv.zero_grad()

        # Compute averaged gradients
        grads = [torch.zeros_like(param) for param in [self.mean, self.D, self.U, self.V]]
        for reward, sample in zip(batch_rewards, batch_samples):
            jitter = 1e-3  # Define a small jitter value
            cov_diag = torch.nn.functional.softplus(self.D) + jitter
            cov_factor = torch.mm(self.U, self.V.t())
            loss = -D.LowRankMultivariateNormal(self.mean, cov_diag=cov_diag, cov_factor=cov_factor).log_prob(torch.tensor(sample)) * reward
            loss.backward(retain_graph=True)
            for i, param in enumerate([self.mean, self.D, self.U, self.V]):
                grads[i] += param.grad / len(batch_rewards)

        # Apply gradients
        for i, param in enumerate([self.mean, self.D, self.U, self.V]):
            param.grad = grads[i]

        self.optimizer_mean.step()
        self.optimizer_d.step()
        self.optimizer_uv.step()

        # Ensure U and V retain their two-dimensional shape
        if len(self.U.shape) != 2:
            self.U = self.U.unsqueeze(-1)
        if len(self.V.shape) != 2:
            self.V = self.V.unsqueeze(-1)

    def sample_design(self):
        jitter = 1e-3  # Define a small jitter value
        cov_diag = self.D + jitter  # Add jitter
        cov_factor = torch.mm(self.U, self.V.t())
        return D.LowRankMultivariateNormal(self.mean, cov_diag=cov_diag, cov_factor=cov_factor).sample()

def reward_function(sampled_design):
    return -torch.norm(torch.tensor(sampled_design) - torch.tensor([10.0, 20.0, 30, 40])).item()

# Initialize parameters
initial_mean = 50 * np.random.rand(4).astype(np.float32)
initial_d = np.random.rand(4).astype(np.float32)
initial_u = np.random.rand(4, 2).astype(np.float32)
initial_v = np.random.rand(4, 2).astype(np.float32)

design_dist = DesignDistribution_log(initial_mean, initial_d, initial_u, initial_v)

num_episodes = 100000
batch_size = 1

for i in range(num_episodes):
    batch_rewards = []
    batch_samples = []

    for _ in range(batch_size):
        sampled_design = design_dist.sample_design().detach().numpy()
        reward = reward_function(sampled_design)
        batch_samples.append(sampled_design)
        batch_rewards.append(reward)

    design_dist.update_distribution(batch_rewards, batch_samples)
    print(f"Episode {i+1}/{num_episodes} - mean: {design_dist.mean.detach().numpy()} reward: {np.mean(batch_rewards)}")
