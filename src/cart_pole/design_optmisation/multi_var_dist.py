import numpy as np

class DesignDistribution:
    def __init__(self, initial_mean, initial_cov, alpha_mean=0.01, alpha_cov=0.01):
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

    def sample_design(self):
        return np.random.multivariate_normal(self.mean, self.cov)

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

def sophisticated_reward_function(sampled_design, target_design, reward_mult):
    diff = sampled_design - reward_mult * target_design
    term1 = np.sum(np.sin(diff))
    term2 = np.exp(-np.sum(diff**2))
    term3 = np.linalg.norm(diff, ord=4)
    reward = term1 + term2 - term3
    return reward


reward_mult = 1
np.set_printoptions(precision=2)

initial_mean = reward_mult*50*np.random.rand(4)
initial_cov = np.eye(4) * 100 * reward_mult + 0.01  # Add small positive value to the diagonal to avoid division by zero

design_dist = DesignDistribution(initial_mean, initial_cov)

# Simulate some robot training episodes
num_episodes = 10000
batch_size = 1
all_samples = []
all_rewards = []

for i in range(0, num_episodes, batch_size):
    batch_samples = []
    batch_rewards = []
    for _ in range(batch_size):
        sampled_design = design_dist.sample_design()
        reward = -np.linalg.norm((sampled_design - reward_mult * np.array([10, 20, 30, 40]))) # This is a mock reward function
        # reward = sophisticated_reward_function(sampled_design, np.array([10, 20, 30, 40]), reward_mult)
        # reward = reward_function(sampled_design)
        batch_samples.append(sampled_design)
        batch_rewards.append(reward)
    # if(np.mean(batch_rewards) > -150):
    #     break

    design_dist.update_distribution(batch_samples, batch_rewards)
    print(design_dist.mean)