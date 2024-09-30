import torch
import torch.distributions as D
import torch.nn.functional as F  # Import the functional module for softplus
import numpy as np
import torch.optim as optim

class DesignDistribution_log:
    def __init__(self, initial_mean, initial_std, lr_mean=0.01, lr_std=0.01, min_parameters = np.ones(4, dtype=np.float32), max_parameters = np.ones(4, dtype=np.float32)):
        self.mean = torch.tensor(initial_mean, dtype=torch.float32, requires_grad=True)
        self.std = torch.tensor(initial_std, dtype=torch.float32, requires_grad=True)
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters
        self.optimizer_mean = optim.Adam([self.mean], lr=lr_mean)
        self.optimizer_std = optim.Adam([self.std], lr=lr_std)

    def update_distribution(self, batch_rewards, batch_samples):

        mean_rewards = torch.mean(torch.tensor(batch_rewards, dtype=torch.float32))

        grad_mean = torch.zeros_like(self.mean)
        grad_std = torch.zeros_like(self.std)

        for i in range(len(batch_rewards)):
            self.optimizer_mean.zero_grad()
            self.optimizer_std.zero_grad()

            sample = torch.tensor(batch_samples[i], dtype=torch.float32)
            neg_log_likelihood = D.Normal(self.mean, F.softplus(self.std)).log_prob(sample).sum()
            neg_log_likelihood.backward(retain_graph=True)

            grad_mean += self.mean.grad
            grad_std += self.std.grad

        grad_mean /= len(batch_rewards)
        grad_std /= len(batch_rewards)

        self.mean.grad = grad_mean * mean_rewards
        self.std.grad = grad_std * mean_rewards

        self.optimizer_mean.step()
        self.optimizer_std.step()

    def sample_design(self):
        counter = 0
        while counter < 1000:
            sample = D.Normal(self.mean, F.softplus(self.std)).sample()
            
            # Check if all sampled values are within bounds
            if torch.all(sample >= torch.tensor(self.min_parameters, dtype=torch.float32)) and \
            torch.all(sample <= torch.tensor(self.max_parameters, dtype=torch.float32)):
                return sample

            counter += 1
        
        # After 10 attempts, return the out-of-bounds parameter as its respective min or max
        sample = sample.detach().numpy()
        min_parameters = np.array(self.min_parameters)
        max_parameters = np.array(self.max_parameters)
        for i in range(len(sample)):
            if sample[i] < min_parameters[i]:
                sample[i] = min_parameters[i]
            elif sample[i] > max_parameters[i]:
                sample[i] = max_parameters[i]
                
        return torch.tensor(sample, dtype=torch.float32)
    
    def get_mean(self):
        return self.mean.detach().numpy()
    def get_std(self):
        return F.softplus(self.std).detach().numpy()

def reward_function(sampled_design):
    reward = torch.norm((sampled_design - torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)))
    return reward.item()
