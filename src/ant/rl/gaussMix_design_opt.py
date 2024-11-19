import torch
import torch.optim as optim
import torch.distributions as D
import numpy as np




from sklearn.mixture import GaussianMixture


class DesignDistribution_log:
    def __init__(self, initial_means, initial_stds, lr_mean=0.001, lr_std=0.001, lr_weight = 0.001,
                 min_parameters=np.ones(25, dtype=np.float32), max_parameters=np.ones(25, dtype=np.float32),
                 max_std=np.ones(25, dtype=np.float32), min_std=np.ones(25, dtype=np.float32), weights=None):

        self.components = len(initial_means)
        self.means = [torch.tensor(mean, dtype=torch.float32, requires_grad=True) for mean in initial_means]
        self.stds = [torch.tensor(std, dtype=torch.float32, requires_grad=True) for std in initial_stds]
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters
        self.min_std = min_std
        self.max_std = max_std

        self.optimizer_means = [optim.Adam([mean], lr=lr_mean) for mean in self.means]
        self.optimizer_stds = [optim.Adam([std], lr=lr_std) for std in self.stds]

        self.weights = torch.nn.Parameter(torch.tensor(weights if weights is not None else np.ones(self.components) / self.components,
                                    dtype=torch.float32))


    def update_distribution(self, batch_rewards, batch_samples, n_envs_train):
        neg_log_likelihood = 0
        for i in range(n_envs_train):
            batch_samples_tensor = torch.tensor(batch_samples)
            batch_rewards_tensor = torch.tensor(batch_rewards)
            sample = batch_samples_tensor[:, i]
            reward = batch_rewards_tensor[:, i]

            for j in range(self.components):

                self.optimizer_means[j].zero_grad()
                self.optimizer_stds[j].zero_grad()

                neg_log_likelihood -= D.Normal(self.means[j], self.stds[j]).log_prob(
                    sample[:,j]) * reward / n_envs_train



        neg_log_likelihood.backward(retain_graph=True)
        for j in range(self.components):
            self.optimizer_means[j].step()
            self.optimizer_stds[j].step()

            self.means[j].data = torch.clamp(self.means[j], min=torch.tensor(self.min_parameters[j]),
                                             max=torch.tensor(self.max_parameters[j]))
            self.stds[j].data = torch.clamp(self.stds[j], min=torch.tensor(self.min_std[j]),
                                            max=torch.tensor(self.max_std[j]))


    def sample_design(self):
        samples = []
        for j in range(self.components):
            counter = 0
            while counter < 1000:
                sample = D.Normal(self.means[j], self.stds[j]).sample()

                if sample >= self.min_parameters[j] and sample <= self.max_parameters[j]:
                    samples.append(sample)
                    break

                counter += 1

        #sample = sample.detach().numpy()

        min_parameters = np.array(self.min_parameters)
        max_parameters = np.array(self.max_parameters)
        for j in range(len(samples)):
            if samples[j] < min_parameters[j]:
                samples[j] = min_parameters[j]
            elif samples[j] > max_parameters[j]:
                samples[j] = max_parameters[j]

        return torch.tensor(samples, dtype=torch.float32)

    def get_mean(self):
        return [mean.detach().numpy() for mean in self.means]

    def get_std(self):
        return [std.detach().numpy() for std in self.stds]

    def get_weight(self):
        return [weight.detach().numpy() for weight in self.weights]

class DesignDistribution_GMM:
    def __init__(self, initial_means, initial_stds, lr_mean=0.001, lr_std=0.001, lr_weight = 0.001,
                 min_parameters=np.ones(25, dtype=np.float32), max_parameters=np.ones(25, dtype=np.float32),
                 max_std=np.ones(25, dtype=np.float32), min_std=np.ones(25, dtype=np.float32), weights=None):

        self.components = len(initial_means)
        self.means = [torch.tensor(mean, dtype=torch.float32, requires_grad=True) for mean in initial_means]
        self.stds = [torch.tensor(std, dtype=torch.float32, requires_grad=True) for std in initial_stds]
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters
        self.min_std = min_std
        self.max_std = max_std

        self.optimizer_means = [optim.Adam([mean], lr=lr_mean) for mean in self.means]
        self.optimizer_stds = [optim.Adam([std], lr=lr_std) for std in self.stds]


    def update_distribution(self, batch_rewards, batch_samples, n_envs_train):
        neg_log_likelihood = 0

        batch_samples_tensor = torch.tensor(batch_samples)
        batch_rewards_tensor = torch.tensor(batch_rewards)
        sample = batch_samples_tensor
        reward = batch_rewards_tensor

        for j in range(self.components):

            self.optimizer_means[j].zero_grad()
            self.optimizer_stds[j].zero_grad()

            neg_log_likelihood -= D.Normal(self.means[j], self.stds[j]).log_prob(
                sample[:,j]) * reward



        neg_log_likelihood.backward(retain_graph=True)

        for j in range(self.components):
            self.optimizer_means[j].step()
            self.optimizer_stds[j].step()

            self.means[j].data = torch.clamp(self.means[j], min=torch.tensor(self.min_parameters[j]),
                                             max=torch.tensor(self.max_parameters[j]))
            self.stds[j].data = torch.clamp(self.stds[j], min=torch.tensor(self.min_std[j]),
                                            max=torch.tensor(self.max_std[j]))


    def sample_design(self):
        samples = []
        for j in range(self.components):
            counter = 0
            while counter < 1000:
                sample = D.Normal(self.means[j], self.stds[j]).sample()

                if sample >= self.min_parameters[j] and sample <= self.max_parameters[j]:
                    samples.append(sample)
                    break

                counter += 1

        #sample = sample.detach().numpy()

        min_parameters = np.array(self.min_parameters)
        max_parameters = np.array(self.max_parameters)
        for j in range(len(samples)):
            if samples[j] < min_parameters[j]:
                samples[j] = min_parameters[j]
            elif samples[j] > max_parameters[j]:
                samples[j] = max_parameters[j]

        return torch.tensor(samples, dtype=torch.float32)

    def get_mean(self):
        return [mean.detach().numpy() for mean in self.means]

    def get_std(self):
        return [std.detach().numpy() for std in self.stds]

