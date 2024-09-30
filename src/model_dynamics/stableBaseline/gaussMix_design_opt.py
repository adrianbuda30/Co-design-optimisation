import numpy as np
import torch

class DesignDistribution:
    def __init__(self, initial_mean, initial_cov, alpha_mean=0.01, alpha_cov=0.01, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = np.array(initial_mean, dtype=np.float64)  # Change to float64
        self.cov = initial_cov
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov
        self.m_mean = np.zeros_like(self.mean)
        self.v_mean = np.zeros_like(self.mean)
        self.m_cov = np.zeros_like(self.cov)
        self.v_cov = np.zeros_like(self.cov)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.min_values = np.array(min_values)
        self.max_values = np.array(max_values)


    def _is_within_bounds(self, sample):
        return np.all(sample >= self.min_values) and np.all(sample <= self.max_values)
    
    def sample_design(self):
        sample = np.random.multivariate_normal(self.mean, self.cov)
        iter = 0
        while not self._is_within_bounds(sample):
            sample = np.random.multivariate_normal(self.mean, self.cov)
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

        # print(np.mean(samples, axis=0))

        rewards = np.array(rewards)
        
        grad_mean = np.mean(rewards[:, np.newaxis] * (samples - self.mean) / np.diagonal(self.cov), axis=0)
        grad_cov = np.mean(rewards[:, np.newaxis, np.newaxis] * 
                          (np.einsum('ij,ik->ijk', samples - self.mean, samples - self.mean) / self.cov - np.eye(self.mean.shape[0])), 
                          axis=0)
        
        # Adam optimizer for mean
        self.m_mean = self.beta1 * self.m_mean + (1 - self.beta1) * grad_mean
        self.v_mean = self.beta2 * self.v_mean + (1 - self.beta2) * grad_mean**2
        m_mean_corr = self.m_mean / (1 - self.beta1**self.t)
        v_mean_corr = self.v_mean / (1 - self.beta2**self.t)
        self.mean += self.alpha_mean * m_mean_corr / (np.sqrt(v_mean_corr) + self.epsilon)

        # Adam optimizer for covariance
        self.m_cov = self.beta1 * self.m_cov + (1 - self.beta1) * grad_cov
        self.v_cov = self.beta2 * self.v_cov + (1 - self.beta2) * grad_cov**2
        m_cov_corr = self.m_cov / (1 - self.beta1**self.t)
        v_cov_corr = self.v_cov / (1 - self.beta2**self.t)
        self.cov += self.alpha_cov * m_cov_corr / (np.sqrt(v_cov_corr) + self.epsilon)
        self.cov += np.eye(self.mean.shape[0]) * 1e-6  # Add small positive value to the diagonal


class DesignDistribution_log:
    def __init__(self, initial_mean, initial_cov, alpha_mean=0.01, alpha_cov=0.01, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = np.array(initial_mean, dtype=np.float64)  # Change to float64
        self.cov = initial_cov
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov
        self.m_mean = np.zeros_like(self.mean)
        self.v_mean = np.zeros_like(self.mean)
        self.m_cov = np.zeros_like(self.cov)
        self.v_cov = np.zeros_like(self.cov)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.min_values = np.array(min_values)
        self.max_values = np.array(max_values)

    def update_distribution(self, samples, rewards):
        self.t += 1
        samples = np.array(samples)
        rewards = np.array(rewards)
        n_samples = len(samples)

        d = self.mean.shape[0]  # Dimension of the data
        # Compute the gradients of the log-likelihood with respect to the mean and covariance
        grad_mean = np.zeros_like(self.mean)
        grad_cov = np.zeros_like(self.cov)

        for i in range(n_samples):
            x_minus_mean = samples[i] - self.mean
            inv_cov = np.linalg.inv(self.cov)
            grad_mean += rewards[i] * inv_cov.dot(x_minus_mean)
            grad_cov += rewards[i] * (-inv_cov + inv_cov.dot(np.outer(x_minus_mean, x_minus_mean)).dot(inv_cov))

        # Normalize the gradients by the number of samples
        grad_mean /= n_samples
        grad_cov /= n_samples

        # grad_mean = -grad_mean
        # grad_cov = -grad_cov
        # Adam optimizer for mean
        self.m_mean = self.beta1 * self.m_mean + (1 - self.beta1) * grad_mean
        self.v_mean = self.beta2 * self.v_mean + (1 - self.beta2) * grad_mean**2
        m_mean_corr = self.m_mean / (1 - self.beta1**self.t)
        v_mean_corr = self.v_mean / (1 - self.beta2**self.t)
        self.mean += self.alpha_mean * m_mean_corr / (np.sqrt(v_mean_corr) + self.epsilon)

        # Adam optimizer for covariance
        self.m_cov = self.beta1 * self.m_cov + (1 - self.beta1) * grad_cov
        self.v_cov = self.beta2 * self.v_cov + (1 - self.beta2) * grad_cov**2
        m_cov_corr = self.m_cov / (1 - self.beta1**self.t)
        v_cov_corr = self.v_cov / (1 - self.beta2**self.t)
        self.cov += self.alpha_cov * m_cov_corr / (np.sqrt(v_cov_corr) + self.epsilon)
        self.cov += np.eye(self.mean.shape[0]) * 1e-6  # Add small positive value to the diagonal

    def _is_within_bounds(self, sample):
        return np.all(sample >= self.min_values) and np.all(sample <= self.max_values)
    
    def sample_design(self):
        sample = np.random.multivariate_normal(self.mean, self.cov)
        while not self._is_within_bounds(sample):
            sample = np.random.multivariate_normal(self.mean, self.cov)
        return sample


class DesignDistribution_pytorch:
    def __init__(self, initial_mean, initial_cov, alpha_mean=0.01, alpha_cov=0.01, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = torch.tensor(initial_mean, dtype=torch.float64, requires_grad=True)
        self.cov = torch.tensor(initial_cov, dtype=torch.float64, requires_grad=True)
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov
        self.min_values = torch.tensor(min_values, dtype=torch.float64)
        self.max_values = torch.tensor(max_values, dtype=torch.float64)
        self.optimizer = torch.optim.Adam([self.mean, self.cov], lr=self.alpha_mean)

    def _is_within_bounds(self, sample):
        return torch.all(sample >= self.min_values) and torch.all(sample <= self.max_values)
    
    def sample_design(self):
        sample = torch.distributions.MultivariateNormal(self.mean, self.cov).sample()
        iter = 0
        while not self._is_within_bounds(sample):
            sample = torch.distributions.MultivariateNormal(self.mean, self.cov).sample()
            iter += 1
            if iter > 10:
                sample = torch.where(sample > self.max_values, self.max_values, sample)
                sample = torch.where(sample < self.min_values, self.min_values, sample)
                return sample.cpu().numpy() if sample.is_cuda else sample.numpy()
        return sample.cpu().numpy() if sample.is_cuda else sample.numpy()


    def update_distribution(self, samples, rewards):
        samples = torch.tensor(samples, dtype=torch.float64)
        rewards = torch.tensor(rewards, dtype=torch.float64)
        
        grad_mean = torch.mean(rewards[:, None] * (samples - self.mean) / torch.diag(self.cov), dim=0)
        grad_cov = torch.mean(rewards[:, None, None] * 
                              (torch.einsum('ij,ik->ijk', samples - self.mean, samples - self.mean) / self.cov - torch.eye(self.mean.shape[0])), 
                              dim=0)

        self.optimizer.zero_grad()
        loss = -torch.sum(grad_mean + grad_cov)  # We're maximizing, hence the negative sign
        loss.backward()
        self.optimizer.step()
        
        # Add small positive value to the diagonal to ensure positive-definite covariance matrix
        with torch.no_grad():
            self.cov += torch.eye(self.mean.shape[0]) * 1e-6


class DesignDistribution_Cholesky_decomposition:
    def __init__(self, initial_mean, initial_L, alpha_mean=0.01, alpha_cov=0.01, min_values=[0.1, 0.1, 0.1, 0.1], max_values=[1, 1, 1, 1]):
        self.mean = torch.tensor(initial_mean, dtype=torch.float32, requires_grad=True)
        self.L = torch.tensor(initial_L, dtype=torch.float32, requires_grad=True)
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov
        self.min_values = torch.tensor(min_values, dtype=torch.float32)
        self.max_values = torch.tensor(max_values, dtype=torch.float32)

    def _is_within_bounds(self, sample):
        return torch.all(sample >= self.min_values) and torch.all(sample <= self.max_values)

    def sample_design(self):
        epsilon = torch.randn_like(self.mean)
        sample = self.mean + torch.matmul(self.L, epsilon)
        return sample.detach().numpy()

    def update_distribution(self, samples, rewards):
        samples = torch.tensor(samples, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Calculate the negative log-likelihood
        epsilon = torch.inverse(self.L) @ (samples - self.mean).T
        log_likelihood = -torch.sum(epsilon ** 2, dim=0) - 2 * torch.sum(torch.log(torch.diag(self.L)))

        # Calculate the loss function
        loss = -torch.sum(rewards * log_likelihood)
        
        # Zero gradients, perform a backward pass, and update the weights.
        self.mean.grad = None
        self.L.grad = None
        loss.backward()

        with torch.no_grad():
            self.mean -= self.alpha_mean * self.mean.grad
            self.L -= self.alpha_cov * self.L.grad

        # Ensure that L remains a lower triangular matrix with positive diagonal elements
        with torch.no_grad():
            self.L.copy_(torch.tril(self.L))
            self.L.diagonal().copy_(torch.clamp(self.L.diagonal(), min=1e-6))
