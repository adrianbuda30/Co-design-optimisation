import torch
import torch.optim as optim
import torch.distributions as D
import numpy as np



from sklearn.mixture import GaussianMixture


class DesignDistribution_log:
    def __init__(self, initial_means, initial_stds, lr_mean=0.001, lr_std=0.001, lr_weight = 0.001,
                 min_parameters=np.ones(2, dtype=np.float32), max_parameters=np.ones(2, dtype=np.float32),
                 max_std=np.ones(2, dtype=np.float32), min_std=np.ones(2, dtype=np.float32), weights=None):

        self.components = len(initial_means)  # Number of components in GMM
        self.means = [torch.tensor(mean, dtype=torch.float32, requires_grad=True) for mean in initial_means]
        self.stds = [torch.tensor(std, dtype=torch.float32, requires_grad=True) for std in initial_stds]
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters
        self.min_std = min_std
        self.max_std = max_std

        # Optimizers for each component
        self.optimizer_means = [optim.Adam([mean], lr=lr_mean) for mean in self.means]
        self.optimizer_stds = [optim.Adam([std], lr=lr_std) for std in self.stds]

        # Initialize and set weights as a trainable parameter
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
                # Zero gradients for each component
                self.optimizer_means[j].zero_grad()
                self.optimizer_stds[j].zero_grad()

                # Calculate the log probability for the current component
                neg_log_likelihood -= self.weights[j] * D.Normal(self.means[j], self.stds[j]).log_prob(
                    sample[:,j]) * reward / n_envs_train



        # Backpropagation step for each component
        neg_log_likelihood.backward(retain_graph=True)
        print(neg_log_likelihood)

        # Update means and stds for each component
        for j in range(self.components):
            self.optimizer_means[j].step()
            self.optimizer_stds[j].step()

            # Clamp mean and std values to remain within valid parameter ranges
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

                # Check if all sampled values are within bounds
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

    def remove_low_reward_components(self, rewards):
        # Calculate the average rewards for each component
        avg_rewards = torch.tensor([torch.mean(r) for r in rewards])
        # Sort components by their rewards
        indices = torch.argsort(avg_rewards)
        # Keep the top 50% of components with the highest rewards
        top_indices = indices[len(indices) // 2:]
        self.means = [self.means[i] for i in top_indices]
        self.stds = [self.stds[i] for i in top_indices]
        self.weights = torch.tensor([self.weights[i] for i in top_indices])
        self.weights /= self.weights.sum()  # Normalize the weights
        self.optimizer_means = [self.optimizer_means[i] for i in top_indices]
        self.optimizer_stds = [self.optimizer_stds[i] for i in top_indices]
        self.components = len(self.means)





class DesignDistribution_GMM:
    def __init__(self, initial_means, initial_stds, lr_mean=0.001, lr_std=0.001, lr_weight=0.001,
                 min_parameters=np.ones(2, dtype=np.float32), max_parameters=np.ones(2, dtype=np.float32),
                 max_std=np.ones(2, dtype=np.float32), min_std=np.ones(2, dtype=np.float32), weights=None):

        self.components = len(initial_means)  # Number of components in GMM
        self.means = [torch.tensor(mean, dtype=torch.float32, requires_grad=True) for mean in initial_means]
        self.stds = [torch.tensor(std, dtype=torch.float32, requires_grad=True) for std in initial_stds]
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters
        self.min_std = min_std
        self.max_std = max_std
        self.n_components = 128  # Initial number of components
        self.iteration_count = 0  # Counter to keep track of iterations

        # Optimizers for each component
        self.optimizer_means = [optim.Adam([mean], lr=lr_mean) for mean in self.means]
        self.optimizer_stds = [optim.Adam([std], lr=lr_std) for std in self.stds]

        # Fit GMM with the initial number of components
        self.optimal_gmm = GaussianMixture(n_components=self.n_components, random_state=42,
                                           covariance_type='full', init_params='kmeans', means_init=self.means,
                                           max_iter=100, n_init=20)

    def update_distribution(self, batch_rewards, batch_samples, n_envs_train):
        self.iteration_count += 1  # Increment iteration counter

        neg_log_likelihood = 0

        # Fit Gaussian mixture models
        self.optimal_gmm.fit(batch_samples, batch_rewards)

        for i in range(n_envs_train):
            batch_samples_tensor = torch.tensor(batch_samples)
            batch_rewards_tensor = torch.tensor(batch_rewards)

            sample = batch_samples_tensor[:, i]
            reward = batch_rewards_tensor[:, i]

            # Zero gradients for each component
            self.optimizer_means.zero_grad()
            self.optimizer_stds.zero_grad()

            # Calculate the log probability for the current component
            neg_log_likelihood -= self.optimal_gmm.log_prob(sample) * reward / n_envs_train

        # Backpropagation step for each component
        neg_log_likelihood.backward(retain_graph=True)

        # Update means and stds for each component
        for j in range(self.components):
            self.optimizer_means[j].step()
            self.optimizer_stds[j].step()

            # Clamp mean and std values to remain within valid parameter ranges
            self.means[j].data = torch.clamp(self.means[j], min=torch.tensor(self.min_parameters[j]),
                                             max=torch.tensor(self.max_parameters[j]))
            self.stds[j].data = torch.clamp(self.stds[j], min=torch.tensor(self.min_std[j]),
                                            max=torch.tensor(self.max_std[j]))

        # Set your own means and covariances
        self.optimal_gmm.means_ = self.means
        self.optimal_gmm.covariances_ = self.stds

        # Check if it's time to halve the components
        if self.iteration_count % 10 == 0:
            self.halve_components(batch_rewards)

    def halve_components(self, rewards):
        """
        Function to halve the number of components based on the highest mean rewards.
        """
        # Get the mean rewards for each component
        mean_rewards = rewards.mean(axis=0)

        # Get the indices of the components with the highest mean rewards
        top_indices = np.argsort(mean_rewards)[-self.n_components // 2:]

        # Keep only the top components
        self.means = [self.means[i] for i in top_indices]
        self.stds = [self.stds[i] for i in top_indices]

        # Update the number of components
        self.n_components //= 2

        # Update the GMM with the reduced number of components
        self.optimal_gmm = GaussianMixture(n_components=self.n_components, random_state=42,
                                           covariance_type='full', init_params='kmeans', means_init=self.means,
                                           max_iter=100, n_init=20)

    def sample_design(self):
        samples = []
        counter = 0
        while counter < 1000:
            sample = self.optimal_gmm.sample(1)

            # Check if all sampled values are within bounds
            if (sample >= self.min_parameters).all() and (sample <= self.max_parameters).all():
                samples.append(sample)
                break

            counter += 1

        # Ensure samples remain within bounds
        samples = np.clip(samples, self.min_parameters, self.max_parameters)
        return torch.tensor(samples, dtype=torch.float32)

    def get_mean(self):
        return [mean.detach().numpy() for mean in self.means]

    def get_std(self):
        return [std.detach().numpy() for std in self.stds]

    def get_weight(self):
        return [weight.detach().numpy() for weight in self.weights]

