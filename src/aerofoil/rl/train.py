import time
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import pandas as pd

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from env_LQR import LQREnv
from env_MPC_HEBO import MPCEnv
from env_aerofoil import AerofoilEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gaussMix_design_opt import DesignDistribution_log as DesignDistribution
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from scipy.io import loadmat, savemat
from stable_baselines3.common.policies import ActorCriticPolicy

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch.nn as nn
import torch
import math

DESIGN_PARAMS = [
    {"name": "x1", "def_value": 1.5},
    {"name": "x2", "def_value": 0.60},
    {"name": "x3", "def_value": 0.35},
    {"name": "x4", "def_value": 1800},
    {"name": "x5", "def_value": 370},
    {"name": "x6", "def_value": 390}
]

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 net_arch, activation_fn=nn.ReLU,
                 *args, **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space,
                                           lr_schedule,
                                           net_arch=net_arch,
                                           activation_fn=activation_fn,
                                           *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.observation_space,
                                           self.net_arch,
                                           self.activation_fn)


class CustomNetwork(nn.Module):

    def __init__(self, observation_space, net_arch, activation_fn, design_obs=5, trajectory_obs=20):
        super(CustomNetwork, self).__init__()

        input_dim = observation_space.shape[0]
        self.encoder_layer_size_design = 4  # Size of the encoded latent space
        self.encoder_layer_size_traj = 4  # Size of the encoded latent space

        self.design_obs = design_obs
        self.trajectory_obs = trajectory_obs

        # Encoder for the last n observations
        self.encoder_design = nn.Sequential(
            nn.Linear(self.design_obs, 8),
            nn.Tanh(),
            nn.Linear(8, self.encoder_layer_size_design),
            nn.Tanh(),
        )

        # Encoder for the second last n observations
        self.encoder_traj = nn.Sequential(
            nn.Linear(self.trajectory_obs, 4),
            nn.Tanh(),
            nn.Linear(4, self.encoder_layer_size_traj),
            nn.Tanh(),
        )

        self.policy_net = self.build_mlp(input_dim - (
                    self.design_obs + self.trajectory_obs) + self.encoder_layer_size_traj + self.encoder_layer_size_design,
                                         net_arch['pi'], activation_fn)
        self.value_net = self.build_mlp(input_dim - (
                    self.design_obs + self.trajectory_obs) + self.encoder_layer_size_traj + self.encoder_layer_size_design,
                                        net_arch['vf'], activation_fn)

        self.latent_dim_pi = net_arch['pi'][-1]
        self.latent_dim_vf = net_arch['vf'][-1]

    def build_mlp(self, input_dim, net_arch, activation_fn):
        layers = []
        last_layer_dim = input_dim
        for layer_size in net_arch:
            layers.append(nn.Linear(last_layer_dim, layer_size))
            layers.append(activation_fn())
            last_layer_dim = layer_size
        return nn.Sequential(*layers)

    def forward(self, obs):
        obs_design = obs[:, -self.design_obs:]
        obs_traj = obs[:, -self.design_obs - self.trajectory_obs:-self.design_obs]
        obs_rest = obs[:, :-self.design_obs - self.trajectory_obs]

        encoder_design = self.encoder_design(obs_design)
        encoder_traj = self.encoder_traj(obs_traj)

        combined_obs = torch.cat([encoder_design, encoder_traj, obs_rest], dim=1)

        policy_output = self.policy_net(combined_obs)
        value_output = self.value_net(combined_obs)

        return policy_output, value_output

    def forward_actor(self, obs):
        obs_design = obs[:, -self.design_obs:].float()
        obs_traj = obs[:, -self.design_obs - self.trajectory_obs:-self.design_obs].float()
        obs_rest = obs[:, :-self.design_obs - self.trajectory_obs].float()

        encoder_design = self.encoder_design(obs_design)
        encoder_traj = self.encoder_traj(obs_traj)

        combined_obs = torch.cat([encoder_design, encoder_traj, obs_rest], dim=1)
        latent_pi = self.policy_net(combined_obs)
        return latent_pi

    def forward_critic(self, obs):
        obs_design = obs[:, -self.design_obs:].float()
        obs_traj = obs[:, -self.design_obs - self.trajectory_obs:-self.design_obs].float()
        obs_rest = obs[:, :-self.design_obs - self.trajectory_obs].float()

        encoder_design = self.encoder_design(obs_design)
        encoder_traj = self.encoder_traj(obs_traj)

        combined_obs = torch.cat([encoder_design, encoder_traj, obs_rest], dim=1)
        latent_vf = self.value_net(combined_obs)
        return latent_vf

class constant_design(BaseCallback):
    def __init__(self, model_name=f"matfile", n_steps_train=512 * 100, n_envs_train=16, verbose=0):

        super(constant_design, self).__init__(verbose)
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.episodic_rewards = []
        self.episode_mean_reward = []
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_design_params = []
        self.mat_reward = []
        self.mat_iteration = []
        self.mat_time = []
        self.model_name = model_name
        self.save_recorded_data = n_steps_train * n_envs_train * 1

    def _on_rollout_start(self) -> bool:
        # reset the environments
        for i in range(self.n_envs_train):
            self.training_env.env_method('reset', indices=[i])

        return True

    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            self.episode_mean_reward = 0
            for i, done in enumerate(dones):
                if done or self.episode_length[i] >= self.n_steps_train:
                    current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
                    dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                    print(
                        f"Env ID: {dist_env_id}, episode reward: {self.episode_rewards[i]},  Mean episode length: {self.episode_length[i]}, Design parameters: {current_design_params}")
                    # Initialize an empty list to store episodic rewards
                    self.episode_mean_reward = self.episode_mean_reward + self.episode_rewards[i]
                    self.logger.record("mean reward", self.episode_rewards[i])
                    self.logger.record("mean episode length", self.episode_length[i])

                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0
                    # self.episode_mean_reward = self.episode_mean_reward/self.n_envs_train
        return True

    def _on_rollout_end(self) -> bool:
        self.model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/constant_design/{self.model_name}")
        print(self.episodic_rewards)
        return True


class random_design(BaseCallback):
    def __init__(self, model_name=f"matfile", n_steps_train=512 * 10, n_envs_train=64, verbose=0):

        super(random_design, self).__init__(verbose)
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.episode_times = {}
        self.convergence_steps = {}
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_design_params = []
        self.mat_reward = []
        self.mat_iteration = []
        self.mat_time = []
        self.model_name = model_name
        self.average_episode_length = []
        self.average_reward = []
        self.average_convergence_time = []

    def _on_rollout_start(self) -> bool:
        # reset the environments
        self.counter = 0
        for i in range(self.n_envs_train):
            self.training_env.env_method('reset', indices=[i])

        return True

    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'infos' in self.locals:
            info = self.locals['infos']
            for i, times in enumerate(info):
                self.episode_times[i] = self.episode_times.get(i, 0) + times['convergence_time']
                self.convergence_steps[i] = self.convergence_steps.get(i, 0) + times['convergence_steps']

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done or self.episode_length[i] >= self.n_steps_train:
                    current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
                    dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                    print(
                        f"Env ID: {dist_env_id}, episode reward: {self.episode_rewards[i]}, episode convergence time:{self.episode_times[i]}, Mean episode length: {self.episode_length[i]}, Design parameters: {current_design_params}")

                    self.average_episode_length.append(self.episode_length[i])
                    self.average_reward.append(self.episode_rewards[i])
                    self.average_convergence_time.append(self.episode_times[i])
                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.episode_times[i] = 0
                    self.episode_length[i] = 0
                    self.convergence_steps[i] = 0
                    self.counter += 1
        return True

    def _on_rollout_end(self) -> bool:
        self.logger.record("mean episode length", np.sum(self.average_episode_length) / self.counter)
        self.logger.record("mean reward", np.sum(self.average_reward) / self.counter)
        self.logger.record("mean convergence time", np.sum(self.average_convergence_time) / self.counter)
        self.average_episode_length = []
        self.average_reward = []
        self.average_convergence_time = []

        self.model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/random_design/{self.model_name}")
        return True


class Schaff_callback(BaseCallback):
    def __init__(self, model_name=f"matfile", n_update_dist=1000, n_steps_train=512 * 100, n_envs_train=8,
                 lr_mean_schaff=0.01, lr_std_schaff=0.01, n_number_dist=8, verbose=0):

        super(Schaff_callback, self).__init__(verbose)
        self.episode_rewards = {}
        self.episode_length = {}
        self.model_name = model_name
        self.mat_file_name = model_name
        # Initialize the distributions
        self.num_distributions = n_number_dist
        self.num_envs = n_envs_train
        self.distributions = []
        self.min_design_params = [1.0, 1.0, 1.0, 1.0]
        self.max_design_params = [200.0, 200.0, 200.0, 200.0]
        self.n_steps_train = n_steps_train
        self.steps_update_distribution = n_steps_train * n_envs_train * n_update_dist
        self.steps_chop_distribution = n_steps_train * n_envs_train * 50
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.model_save_interval = n_steps_train * n_envs_train * 1
        np.set_printoptions(precision=5)
        # initialize matlab data
        self.mat_dist_mean = [[] for _ in range(self.num_distributions)]
        self.mat_dist_std = [[] for _ in range(self.num_distributions)]
        self.mat_design_params = [[] for _ in range(self.num_distributions)]
        self.mat_reward = [[] for _ in range(self.num_distributions)]
        self.mat_mean_reward = [[] for _ in range(self.num_distributions)]
        self.mat_iteration = [[] for _ in range(self.num_distributions)]
        self.mat_iter = 0
        self.Schaffs_batch_size = 10
        self.Schaffs_batch_size_iter = [0 for _ in range(self.num_distributions)]
        self.Schaffs_batch_design = [[] for _ in range(self.num_distributions)]
        self.Schaffs_batch_reward = [[] for _ in range(self.num_distributions)]

        z = 2  # Number of standard deviations to cover

        for _ in range(self.num_distributions):
            self.initial_mean = np.array([min_val + (max_val - min_val) * np.random.rand()
                                          for min_val, max_val in zip(self.min_design_params, self.max_design_params)])
            # self.initial_std = np.ones(4, dtype=np.float32) * 0.01  # Initialize std deviation as you prefer
            self.initial_std = np.array([(max_val - min_val) / (2 * z)
                                         for min_val, max_val, max_val in
                                         zip(self.min_design_params, self.max_design_params, self.max_design_params)])

            self.design_dist = DesignDistribution(self.initial_mean, self.initial_std,
                                                  min_parameters=self.min_design_params,
                                                  max_parameters=self.max_design_params, lr_mean=lr_mean_schaff,
                                                  lr_std=lr_std_schaff)
            print(self.initial_mean, self.design_dist.get_mean())
            self.distributions.append(self.design_dist)

    def _on_step(self) -> bool:

        if self.num_timesteps > self.steps_update_distribution:
            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                for i, reward in enumerate(rewards):
                    self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                    self.episode_length[i] = self.episode_length.get(i, 0) + 1

            if 'dones' in self.locals:
                dones = self.locals['dones']
                for i, done in enumerate(dones):
                    if done:

                        # total_episode_mean_reward = self.episode_rewards[i] / self.episode_length[i]
                        total_episode_mean_reward = self.episode_rewards[i]

                        # Rescale reward to be between 0 and 1

                        rescaled_reward = self.episode_rewards[i] / (0.75 * self.n_steps_train)
                        rescaled_reward = np.clip(rescaled_reward, 0, 1)
                        rescaled_reward = self.convert_range(rescaled_reward, 0, 1, 0, 10)
                        rescaled_reward = np.clip(rescaled_reward, 0, 10)
                        rescaled_reward = math.exp(rescaled_reward)
                        torch_reward = torch.tensor(-rescaled_reward, dtype=torch.float32)

                        # Calculate the mean reward for the episode
                        dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]

                        current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
                        print(
                            f"Env ID: {dist_env_id}, Mean episode length: {self.episode_length[i]} , Current arm length: {current_design_params}, Mean reward: {total_episode_mean_reward}, distribution mean: {self.distributions[(dist_env_id)].get_mean()}, distribution std: {self.distributions[(dist_env_id)].get_std()}")
                        self.logger.record("mean reward", self.episode_rewards[i])
                        self.logger.record("mean episode length", self.episode_length[i])
                        self.Schaffs_batch_reward[dist_env_id].append(torch_reward)
                        self.Schaffs_batch_design[dist_env_id].append(current_design_params * 100)
                        self.Schaffs_batch_size_iter[dist_env_id] += 1

                        # Make batches before updating
                        # Update the distributions based on the episode reward
                        if self.Schaffs_batch_size_iter[dist_env_id] >= self.Schaffs_batch_size:
                            self.distributions[(dist_env_id)].update_distribution(
                                self.Schaffs_batch_design[dist_env_id], self.Schaffs_batch_design[dist_env_id])
                            self.Schaffs_batch_design[dist_env_id] = []
                            self.Schaffs_batch_reward[dist_env_id] = []
                            self.Schaffs_batch_size_iter[dist_env_id] = 0

                        # Modify the environment parameter based on the episode reward
                        new_design_params = self.distributions[(dist_env_id)].sample_design().detach().numpy()
                        new_design_params = np.clip(new_design_params, self.min_design_params, self.max_design_params)

                        # Calling the set_design_params method of the environment
                        self.training_env.env_method('set_design_params', new_design_params / 100, indices=[i])

                        # Logging
                        self.mat_dist_mean[dist_env_id].append(self.distributions[(dist_env_id)].get_mean())
                        self.mat_dist_std[dist_env_id].append(self.distributions[(dist_env_id)].get_std())
                        self.mat_design_params[dist_env_id].append(current_design_params)
                        self.mat_reward[dist_env_id].append(self.episode_rewards[i])
                        self.mat_mean_reward[dist_env_id].append(total_episode_mean_reward)
                        self.mat_iteration[dist_env_id].append(self.episode_length[i])

                        # Reset episode reward accumulator
                        self.episode_rewards[i] = 0
                        self.episode_length[i] = 0
                        total_episode_mean_reward = 0

            # save matlab data
            if self.num_timesteps % self.save_recorded_data == 0:
                output_data = {
                    "dist_mean": np.array(self.mat_dist_mean),
                    "dist_std": np.array(self.mat_dist_std),
                    "design_params": np.array(self.mat_design_params),
                    "reward": np.array(self.mat_reward),
                    "mean_reward": np.array(self.mat_mean_reward),
                    "iteration": np.array(self.mat_iteration)
                }
                print("saving to output.mat...")
                self.mat_iter += 1
                file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/{self.mat_file_name}_{self.mat_iter}.mat"
                savemat(file_path, output_data)
                self.model.save(
                    f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/{self.model_name}_{self.mat_iter}")
            # chop low performing distributions
            if self.num_timesteps % self.steps_chop_distribution == 0:

                if len(self.distributions) > 1:
                    print("Updating design distribution...")
                    # Step 1: Calculate mean rewards for each distribution
                    mean_rewards = [np.mean(self.mat_mean_reward[i]) for i in range(len(self.mat_mean_reward))]

                    # Step 2: Sort the distributions by mean rewards and take the top half
                    sorted_indices = np.argsort(mean_rewards)[::-1]  # Sort in descending order
                    top_indices = sorted_indices[:len(sorted_indices) // 2]  # Take the top half

                    # Step 3: Keep only the top-performing distributions
                    self.distributions = [self.distributions[i] for i in top_indices]
                    self.mat_dist_mean = [self.mat_dist_mean[i] for i in top_indices]
                    self.mat_dist_std = [self.mat_dist_std[i] for i in top_indices]
                    self.mat_design_params = [self.mat_design_params[i] for i in top_indices]
                    self.mat_reward = [self.mat_reward[i] for i in top_indices]
                    self.mat_mean_reward = [self.mat_mean_reward[i] for i in top_indices]
                    self.mat_iteration = [self.mat_iteration[i] for i in top_indices]

                    # Step 4: Set the new distributions for each environment id
                    # Number of distributions left
                    num_distributions_left = len(self.distributions)

                    # Number of environments per distribution
                    envs_per_distribution = self.num_envs // num_distributions_left  # Assuming this division is exact

                    # Assign env IDs
                    for i, distribution_id in enumerate(range(num_distributions_left)):
                        for k in range(envs_per_distribution):
                            # Here, i is the index of the distribution,
                            # and i+k is the index of the environment.
                            self.training_env.env_method('set_env_id', distribution_id,
                                                         indices=[i * envs_per_distribution + k])

                    print(f"Kept {len(top_indices)} top-performing distributions.")
                    print(
                        f"New distribution means: {[self.distributions[i].get_mean() for i in range(len(self.distributions))]}")
                    print(
                        f"New distribution stds: {[self.distributions[i].get_std() for i in range(len(self.distributions))]}")
                    print(
                        f"New ditribution env IDs: {[self.training_env.env_method('get_env_id', indices=[i])[0] for i in range(self.num_envs)]}")
                    print(
                        f"New distribution mean rewards: {[np.mean(self.mat_mean_reward[i]) for i in range(len(self.mat_mean_reward))]}")

        else:
            if self.num_timesteps % self.model_save_interval == 0:
                self.model.save(
                    f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/{self.model_name}_{self.mat_iter}")
                print(f"Inter Model saved: {self.mat_iter}")

            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                for i, reward in enumerate(rewards):
                    self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                    self.episode_length[i] = self.episode_length.get(i, 0) + 1

            if 'dones' in self.locals:
                dones = self.locals['dones']
                for i, done in enumerate(dones):
                    if done:
                        self.logger.record("mean reward", self.episode_rewards[i])
                        self.logger.record("mean episode length", self.episode_length[i])
                        print(
                            f"Env ID: {i}, Mean reward: {self.episode_rewards[i]}, Mean episode length: {self.episode_length[i]}, arm length: {self.training_env.env_method('get_design_params', indices=[i])[0]}")
                        # Reset episode reward accumulator
                        self.episode_rewards[i] = 0
                        self.episode_length[i] = 0

    def convert_range(self, x, min_x, max_x, min_y, max_y):
        return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y


class Hebo_callback(BaseCallback):

    def __init__(self, model_name=f"matfile", n_steps_train=512 * 10, n_envs_train=100, batch_size_opt=10,
                 design_params_limits=np.array([0.01, 1]), verbose=0):

        super(Hebo_callback, self).__init__(verbose)

        self.batch_iterations = n_steps_train * n_envs_train
        self.steps_update_distribution = self.batch_iterations * 1  # Set to batch_iterations * 1 for clarity
        self.n_envs_train = n_envs_train
        self.model_name = model_name
        self.min_design_params = design_params_limits[0]
        self.max_design_params = design_params_limits[1]

        self.distributions = []
        self.mat_design_params = []
        self.mat_reward = []
        self.mat_iteration = []
        self.mat_time = []
        self.state = 'propose_design'
        self.design_process = False  # Initialize to False
        self.mat_file_name = model_name
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.reduce_batch_size = n_steps_train * n_envs_train * 1
        self.batch_size_opt = batch_size_opt
        self.n_steps_train = 10000
        #self.model.evaluate_current_policy = True


        # Initialize
        self.episode_rewards = {}
        self.episode_length = {}
        self.episode_times = {}
        self.convergence_steps = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
        self.best_design = []
        self.best_design_reward = []
        self.average_length = []
        self.average_reward = []
        self.average_time = []
        self.checker = [False for _ in range(self.n_envs_train)]

        np.set_printoptions(precision=5)

        space = DesignSpace().parse([
            #{'name': 'x1', 'type': 'num', 'lb': 0.01, 'ub': 100},
            {'name': 'x2', 'type': 'num', 'lb': 0.01, 'ub': 0.8},
            {'name': 'x3', 'type': 'num', 'lb': 0.01, 'ub': 0.8},
            {'name': 'x4', 'type': 'num', 'lb': 50, 'ub': 5000},
            {'name': 'x5', 'type': 'num', 'lb': 50, 'ub': 5000},
            #{'name': 'x6', 'type': 'num', 'lb': 0.01, 'ub': 5000}
        ])

        self.opt = HEBO(space)
        print("Initialisation callback: ")

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        print("Training started")
        print(self.n_envs_train)
        # set the environment id for each environment
        for i in range(self.n_envs_train):
            self.training_env.env_method('set_env_id', i, indices=[i])

        return True

    def _on_rollout_start(self) -> bool:

        # reset the environments
        self.checker = [False for _ in range(self.n_envs_train)]
        self.average_reward = []
        self.average_length = []
        self.average_time = []

        # Reset episode reward accumulator
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.episode_rewards = {}
        self.episode_length = {}
        self.episode_times = {}
        self.convergence_steps = {}


        print("...Updating the new lengths...")
        start_time = time.time()

        self.rec = self.opt.suggest(n_suggestions=self.n_envs_train // self.batch_size_opt)

        for i in range(self.n_envs_train // self.batch_size_opt):
            for j in range(self.batch_size_opt):
                new_design_params = []
                curr_rec = self.rec.iloc[i]
                for design_param in DESIGN_PARAMS:
                    try:
                        new_design_params.append(curr_rec.at[design_param["name"]])
                    except:
                        new_design_params.append(design_param["def_value"])
                new_design_params = np.array(new_design_params)
                self.training_env.env_method('set_design_params', new_design_params,
                                             indices=[i * self.batch_size_opt + j])
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                print(
                    f"env: {i * self.batch_size_opt + j:<1.2f}, real id:{dist_env_id}, design parameters: {new_design_params}")

        print(f"Design proposal took {time.time() - start_time:.2f} seconds")
        self.training_env.env_method('reset', indices=range(self.n_envs_train))
        return True

    def _on_step(self) -> bool:
        st = time.time()

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                if self.checker[i] == False:
                    self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                    self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'infos' in self.locals:
            info = self.locals['infos']
            for i, times in enumerate(info):
                if self.checker[i] == False:
                    self.episode_times[i] = self.episode_times.get(i, 0) + times['convergence_time']
                    self.convergence_steps[i] = self.convergence_steps.get(i, 0) + times['convergence_steps']

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done:
                    self.checker[i] = True



        return True

    def _on_rollout_end(self) -> bool:

        if self.num_timesteps >= self.steps_update_distribution - self.n_envs_train:

            print("...Starting design distribution update...")
            start_time = time.time()

            scores = []

            self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
            self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
            self.convergence_time_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
            self.convergence_steps_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
            for i in range(self.n_envs_train // self.batch_size_opt):
                # average batch reward
                sum_reward = 0
                total_episode_length = 0
                convergence_time = 0
                total_convergence_steps = 1
                for j in range(self.batch_size_opt):
                    sum_reward += self.episode_rewards[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]
                    convergence_time += self.episode_times[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]
                    total_episode_length += self.episode_length[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]
                    total_convergence_steps = self.convergence_steps[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]
                self.design_rewards_avg[i] = sum_reward / self.batch_size_opt
                self.episode_length_avg[i] = total_episode_length / self.batch_size_opt
                self.convergence_steps_avg[i] = total_convergence_steps / self.batch_size_opt
                self.convergence_time_avg[i] = convergence_time / (self.batch_size_opt * self.convergence_steps_avg[i])

                if self.convergence_time_avg[i] == 0:
                    self.convergence_time_avg[i] = 100
                self.average_length.append(self.episode_length_avg[i])
                self.average_reward.append(self.design_rewards_avg[i])
                self.average_time.append(self.convergence_time_avg[i])
                score_array = np.array(self.design_rewards_avg[i]).reshape(-1, 1)  # Convert to NumPy array
                scores.append(-score_array)  # HEBO minimizes, so we need to negate the scores

                # Logging
                current_design_params = \
                self.training_env.env_method('get_design_params', indices=[i * self.batch_size_opt])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_opt])[0]
                print(
                    f"Env ID: {dist_env_id}, mean reward: {self.design_rewards_avg[i]}, Mean episode length: {self.episode_length_avg[i]}, mean convergence time: {self.convergence_time_avg[i]}, arm length: {current_design_params}")

                # Matlab logging
                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.design_rewards_avg[i])
                self.mat_iteration.append(self.episode_length_avg[i])
                self.mat_time.append(self.convergence_time_avg[i])

            self.logger.record("mean reward", np.sum(self.average_reward) / (self.n_envs_train // self.batch_size_opt))
            self.logger.record("mean episode length", np.sum(self.average_length) / (self.n_envs_train // self.batch_size_opt))


            # Update the design distribution
            scores = np.array(scores)  # Make sure the outer list is also a NumPy array

            self.opt.observe(self.rec, scores)

            self.state = 'propose_design'

            # After all iterations, print the best input and output
            best_idx = self.opt.y.argmin()
            best_design = self.opt.X.iloc[best_idx]
            best_design_reward = self.opt.y[best_idx]

            print(f"Best design: {best_design}, best reward: {best_design_reward}")
            self.best_design.append(best_design)
            self.best_design_reward.append(best_design_reward)

            print(f"Design distribution update took {time.time() - start_time:.2f} seconds")
        else:
            # Logging
            for i in range(self.n_envs_train):
                current_design_params = self.training_env.env_method('get_design_params', indices=[i])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                print(
                    f"Env ID: {dist_env_id}, episode reward: {self.episode_rewards[i]}, mean reward: {self.episode_rewards[i] / self.design_iteration[i]}, design iter: {self.design_iteration[i]}, episode length: {self.episode_length[i]}, convergence time: {self.episode_times[i] / self.convergence_steps[i]} arm length: {current_design_params}")
                self.logger.record("mean reward", self.episode_rewards[i] / self.design_iteration[i])
                self.logger.record("convergence time", (self.episode_times[i] / (self.design_iteration[i] * self.convergence_steps[i])))
                self.logger.record("mean episode length", self.episode_length[i])

                # Matlab logging
                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.episode_rewards[i] / self.design_iteration[i])
                self.mat_time.append((self.episode_times[i] / (self.design_iteration[i] * self.convergence_steps[i])))
                self.mat_iteration.append(self.episode_length[i] / self.design_iteration[i])


        if self.num_timesteps % self.save_recorded_data == 0:
            output_data = {
                "design_params": np.array(self.mat_design_params),
                "reward": np.array(self.mat_reward),
                "convergence_time": np.array(self.mat_time),
                "iteration": np.array(self.mat_iteration)
            }
            print("saving matlab data...")

            file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/{self.mat_file_name}.mat"
            savemat(file_path, output_data)
            print("saving current model...")
            self.model.save(
                f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/HEBO/{self.model_name}")
            print("Model saved")

        # increasing the batch size with each iteration
        # if self.num_timesteps % self.reduce_batch_size == 0:
        #     print("Increasing batch size...")
        #     self.batch_size_opt = 2 * self.batch_size_opt
        #     if self.batch_size_opt > self.n_envs_train:
        #         self.batch_size_opt = self.n_envs_train

        return True

class Hebo_Gauss_callback(BaseCallback):

    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 10, n_envs_train=20,
                 design_params_limits=np.array([0.01, 1]), verbose=0):

        super(Hebo_Gauss_callback, self).__init__(verbose)

        self.batch_iterations = n_steps_train * n_envs_train
        self.steps_update_distribution = self.batch_iterations * 0  # Set to batch_iterations * 1 for clarity
        self.n_envs_train = n_envs_train
        self.model_name = model_name
        self.min_bound = design_params_limits[0]
        self.max_bound = design_params_limits[1]
        self.model = model
        self.distributions = []
        self.mat_arm_length = []
        self.mat_best_design_gauss = []
        self.mat_design_upper_bound = []
        self.mat_design_lower_bound = []
        self.mat_reward = []
        self.mat_iteration = []
        self.mat_time = []
        self.mat_design_params = []
        self.design_process = False  # Initialize to False
        self.mat_file_name = model_name
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.reduce_batch_size = n_steps_train * n_envs_train * 1


        # Initialize

        self.state = 'hebo_init'  # random or gaussian or init_gaussian or hebo or init_hebo

        self.sampling_hebo_time_period = 4
        self.sampling_gauss_time_period = 4
        self.sampling_random_time_period = 0
        self.sampling_hebo_ctr = 0
        self.sampling_gauss_ctr = 0
        self.sampling_random_ctr = 0
        self.hebo_design_history = []
        self.hebo_reward_history = []

        self.percentile_top_designs = 20  # 0 to 100
        self.batch_size_hebo = 10
        self.batch_size_gauss = 1
        self.batch_size_random = 1
        self.optimal_components = 1
        self.optimal_gmm = None
        self.hebo_init_ctr = 0
        self.gauss_init_ctr = 0

        self.episode_rewards = {}
        self.episode_length = {}
        self.episode_times = {}
        self.convergence_steps = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
        self.convergence_time_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
        self.best_design = []
        self.best_design_reward = []
        self.target_pos_param = []
        self.init_joint_pos = []
        self.logger_reward = []
        self.logger_reward_prev = -1000
        self.logger_episode_length = []
        self.logger_convergence_time = []
        # define hebo design space based on the number of components
        self.design_space_lb = np.array([0.01, 0.01])
        self.design_space_ub = np.array([0.8, 0.8])
        self.design_space_history = []
        self.best_design_gauss = np.array([0.1, 0.1])
        self.best_design_reward_gauss = -1000
        self.mat_best_reward_policy = -1000
        # Range of possible components
        self.n_components_range = range(1, 4)
        self.sample_half_gauss = False

        self.column_names = ['x1', 'x2']
        self.rec_gauss_his = pd.DataFrame(columns=self.column_names)
        self.scores_gauss_his = []
        np.set_printoptions(precision=3)

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        print("Training started")

    def _on_rollout_start(self) -> bool:

        if self.state == "hebo_init":
            print("initialising HEBO")

            space_config = []
            self.opt = None

            for i in range(1, len(self.design_space_lb) + 1):
                space_config.append({
                    'name': f'x{i}',  # This will create 'x1', 'x2', etc.
                    'type': 'num',
                    'lb': self.design_space_lb[i - 1],  # Assuming self.min_arm_length is defined
                    'ub': self.design_space_ub[i - 1]  # Assuming self.max_arm_length is defined
                })
            space = DesignSpace().parse(space_config)
            self.opt = HEBO(space)
            self.hebo_init_ctr += 1
            self.state = "hebo"

        if self.state == "gauss_init":
            print("initialising gaussian mixture models")
            sorted_indices = []
            top_indices = []
            self.top_designs = []
            self.top_rewards = []
            top_designs_scaled = []

            # decrease the time period for sampling from hebo
            # self.sampling_hebo_time_period -= 5
            # if self.sampling_hebo_time_period <= 30:
            # self.sampling_hebo_time_period = 30

            self.sampling_gauss_time_period = 2

            # define the gaussian mixture models based on the history of hebo
            # order top performing designs based on reward in descending order
            self.hebo_design_history = np.array(self.hebo_design_history)
            self.hebo_reward_history = np.concatenate(self.hebo_reward_history, axis=0)
            sorted_indices = np.argsort(self.hebo_reward_history)[::-1]

            # take the top percentile of the designs
            top_indices = sorted_indices[:int((self.percentile_top_designs / 100) * len(sorted_indices))]

            # keep designs greater than the average reward
            avg_reward = np.mean(self.hebo_reward_history)
            # top_indices = [i for i in range(len(self.hebo_reward_history)) if self.hebo_reward_history[i] > avg_reward]
            print(f"avg reward: {avg_reward}, top indices: {len(top_indices)}")
            print(
                f"top desgins: {self.hebo_design_history[top_indices]}, top rewards: {self.hebo_reward_history[top_indices]}")
            self.top_designs = self.hebo_design_history[top_indices]
            self.top_rewards = self.hebo_reward_history[top_indices]

            # find the lower and upper bound of the top designs for each dimension

            self.design_space_lb = np.min(self.top_designs, axis=0)
            self.design_space_ub = np.max(self.top_designs, axis=0)

            self.design_space_history.append([self.design_space_lb, self.design_space_ub])
            print(f"Design space lower bound: {self.design_space_lb}")
            print(f"Design space upper bound: {self.design_space_ub}")
            print(f"Design space history: {self.design_space_history}")

            self.mat_design_upper_bound.append(self.design_space_ub)
            self.mat_design_lower_bound.append(self.design_space_lb)

            # scale self.top_designs to the design space bounded by lb and ub
            top_designs_scaled = (self.top_designs - self.design_space_lb) / (self.design_space_ub - self.design_space_lb)
            # normalise top rewards
            top_rewards_norm = (self.top_rewards - np.min(self.top_rewards)) / (np.max(self.top_rewards) - np.min(self.top_rewards))

            bic_scores = []
            aic_scores = []

            # Fit GMMs with different number of components
            for n_components in self.n_components_range:
                gmm = GaussianMixture(n_components=n_components, random_state=42,
                                      covariance_type='full', init_params='kmeans',
                                      max_iter=100, n_init=20)

                # fit until convergence
                gmm.fit(top_designs_scaled, top_rewards_norm)
                print(f"gmm {n_components} converged: {gmm.converged_}, with iterations: {gmm.n_iter_}")

                #gmm.fit(top_designs_scaled)
                bic_scores.append(gmm.bic(top_designs_scaled))
                aic_scores.append(gmm.aic(top_designs_scaled))

                # Choosing the optimal number of components
                self.optimal_components = np.argmin(bic_scores) + 1  # +1 because range starts from 1
                print(f"Optimal number of components according to BIC: {self.optimal_components}")

                # Fit GMM with the optimal number of components
                self.optimal_gmm = GaussianMixture(n_components=self.optimal_components, random_state=42,
                                                   covariance_type='full', init_params='kmeans',
                                                   max_iter=100, n_init=20)

            # fit gaussian mixture models
            self.optimal_gmm.fit(top_designs_scaled, top_rewards_norm)

            print(f"GMM means: {self.optimal_gmm.means_}")
            print(f"GMM covariances: {self.optimal_gmm.covariances_}")
            print(
                f"GMM means originally scaled: {self.optimal_gmm.means_ * (self.design_space_ub - self.design_space_lb) + self.design_space_lb}")
            self.hebo_design_history = []
            self.hebo_reward_history = []

            # design_plot = self.optimal_gmm.sample(400)
            # clusters = self.optimal_gmm.predict(design_plot[0])

            # design_plot_ds = design_plot[0] * (self.design_space_ub - self.design_space_lb) + self.design_space_lb
            # # Plotting the clusters in 3D
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(111, projection='3d')
            # scatter = ax.scatter(design_plot_ds[:, 0], design_plot_ds[:, 1], design_plot_ds[:, 2], c=clusters, cmap='viridis', marker='o')
            # legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
            # ax.add_artist(legend1)
            # ax.set_title("Gaussian Mixture Model Clustering on Complex 3D Data")
            # ax.set_xlabel("X axis")
            # ax.set_ylabel("Y axis")
            # ax.set_zlabel("Z axis")
            # ax.set_xlim(0.1, 2.0)
            # ax.set_ylim(0.1, 2.0)
            # ax.set_zlim(0.1, 2.0)
            # plt.show()

            self.gauss_init_ctr += 1

            self.state = "gauss"

        if self.state == "hebo":

            # turn off train when sampling from hebo
            self.model.evaluate_current_policy = True
            print("...Updating the new lengths...")
            start_time = time.time()

            if not self.sample_half_gauss:
                # sample from hebos
                self.rec = self.opt.suggest(n_suggestions=(self.n_envs_train // self.batch_size_hebo))
            else:
                # sample half from hebos
                self.rec = self.opt.suggest(n_suggestions=(self.n_envs_train // (self.batch_size_hebo * 2)))

                # #sample half from gaussian mixture models
                # gauss_output_scaled = self.optimal_gmm.sample(self.n_envs_train // (self.batch_size_hebo*2))
                # #scale the sampled designs to the design space bounded by lb and ub
                # gauss_output = gauss_output_scaled[0] * (self.design_space_ub - self.design_space_lb) + self.design_space_lb
                # gauss_output = np.clip(gauss_output, self.design_space_lb, self.design_space_ub)
                # # Convert gauss_output[0] to a DataFrame with the same column names
                # gauss_output_df = pd.DataFrame(gauss_output, columns=self.column_names)

                prev_top_designs = pd.DataFrame(self.top_designs[:int(self.n_envs_train // (self.batch_size_hebo * 2))],
                                                columns=self.column_names)

                # Now concatenate the DataFrames
                self.rec = pd.concat([self.rec, prev_top_designs], ignore_index=True)
                print("first hebo", self.rec)

                self.sample_half_gauss = False

            for i in range(self.n_envs_train // self.batch_size_hebo):
                for j in range(self.batch_size_hebo):
                    self.new_design_params = self.rec.values[i]
                    self.training_env.env_method('__init__', i, indices=[i * self.batch_size_hebo + j])
                    self.training_env.env_method('set_design_params', self.new_design_params,
                                                 indices=[i * self.batch_size_hebo + j])

                    self.training_env.env_method('reset', indices=[i * self.batch_size_hebo + j])

                    # reset the environment
                    self.training_env.env_method('reset', indices=[i * self.batch_size_hebo + j])
                    dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_hebo + j])[0]

                    current_design_params = self.training_env.env_method('get_design_params', indices=[i * self.batch_size_hebo + j])[0]

                    # print(f"env: {i*self.batch_size_hebo + j:<1.2f}, real id:{dist_env_id}, limb_length: {self.limb_length}")

            print(f"Design proposal took {time.time() - start_time:.2f} seconds")
            self.sampling_hebo_ctr += 1

        if self.state == "gauss":

            # turn on train when sampling from gaussian mixture models
            self.model.evaluate_current_policy = False

            # sample from gaussian mixture models
            self.gauss_output = self.optimal_gmm.sample(self.n_envs_train // self.batch_size_gauss)
            self.gauss_designs = self.gauss_output[0]
            # scale the sampled designs to the design space bounded by lb and ub
            self.gauss_designs = self.gauss_designs * (
                        self.design_space_ub - self.design_space_lb) + self.design_space_lb
            self.gauss_designs = np.clip(self.gauss_designs, self.design_space_lb, self.design_space_ub)

            # sample from gaussian mixture models and set the designs in the environments
            for i in range(self.n_envs_train // self.batch_size_gauss):
                for j in range(self.batch_size_gauss):
                    self.training_env.env_method('__init__', i, indices=[i * self.batch_size_gauss + j])
                    self.training_env.env_method('set_design_params', self.gauss_designs[i],
                                                 indices=[i * self.batch_size_gauss + j])
                    self.training_env.env_method('reset', indices=[i * self.batch_size_gauss + j])

            self.sampling_gauss_ctr += 1
            self.sample_half_gauss = False

        return True

    def _on_rollout_end(self) -> bool:

        if self.state == "hebo":
            print("...Starting design distribution update...")
            start_time = time.time()

            scores = []

            self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
            self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
            self.convergence_time_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
            self.convergence_steps_avg = [0 for _ in range(self.n_envs_train // self.batch_size_hebo)]
            for i in range(self.n_envs_train // self.batch_size_hebo):
                # average batch reward
                sum_reward = 0
                total_episode_length = 0
                convergence_time = 0
                total_convergence_steps = 0
                for j in range(self.batch_size_hebo):
                    sum_reward += self.episode_rewards[i * self.batch_size_hebo + j] / self.design_iteration[
                        i * self.batch_size_hebo + j]
                    convergence_time += self.episode_times[i * self.batch_size_hebo + j] / self.design_iteration[
                        i * self.batch_size_hebo + j]
                    total_episode_length += self.episode_length[i * self.batch_size_hebo + j] / self.design_iteration[
                        i * self.batch_size_hebo + j]
                    total_convergence_steps = self.convergence_steps[i * self.batch_size_gauss + j] / self.design_iteration[
                        i * self.batch_size_gauss + j]
                self.design_rewards_avg[i] = sum_reward / self.batch_size_hebo
                self.episode_length_avg[i] = total_episode_length / self.batch_size_hebo
                self.convergence_steps_avg[i] = total_convergence_steps
                self.convergence_time_avg[i] = convergence_time / (self.batch_size_gauss * self.convergence_steps_avg[i])
                score_array = np.array(self.design_rewards_avg[i]).reshape(-1, 1)  # Convert to NumPy array
                scores.append(-score_array)  # HEBO minimizes, so we need to negate the scores

                # Logging
                current_design_params = \
                self.training_env.env_method('get_design_params', indices=[i * self.batch_size_hebo])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_hebo])[0]
                #print(
                #    f"Env ID: {dist_env_id}, mean reward: {self.design_rewards_avg[i]}, Mean episode length: {self.episode_length_avg[i]}, arm length: {current_design_params}")

                self.logger.record("mean reward", self.design_rewards_avg[i])
                self.logger.record("mean episode length", self.episode_length_avg[i])
                self.logger.record("mean convergence time", self.convergence_time_avg[i])
                # Matlab logging
                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.design_rewards_avg[i])
                self.mat_iteration.append(self.episode_length_avg[i])
                self.mat_time.append(self.convergence_time_avg[i])

                self.logger_reward.append(self.design_rewards_avg[i])
                self.logger_episode_length.append(self.episode_length_avg[i])
                self.logger_convergence_time.append(self.convergence_time_avg[i])
                if self.design_rewards_avg[i] > self.mat_best_reward_policy:
                    self.mat_best_reward_policy = self.design_rewards_avg[i]
                    self.model.save(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/aerofoil/rl/trained_model/bestHeboDesign_{self.model_name}")

            self.logger.record("mean reward", np.mean(self.logger_reward))
            self.logger.record("mean episode length", np.mean(self.logger_episode_length))
            self.logger.record("mean convergence time", np.mean(self.logger_convergence_time))
            self.logger_reward = []
            self.logger_episode_length = []
            self.logger_convergence_time = []


            scores = np.array(scores)  # Make sure the outer list is also a NumPy array


            # save history of design and reward proposed by hebo
            self.hebo_design_history.extend(self.rec.values.tolist())
            self.hebo_reward_history.extend(np.concatenate(-scores).tolist())

            # Update the design distribution
            scores = scores.reshape(-1, 1)
            self.opt.observe(self.rec, scores)

            # #check if self.rec_gauss_his is empty
            # if len(self.scores_gauss_his) > 0:
            #     self.scores_gauss_his = np.array(self.scores_gauss_his).reshape(-1, 1)
            #     self.opt.observe(self.rec_gauss_his, self.scores_gauss_his)
            #     self.rec_gauss_his = pd.DataFrame(columns=self.column_names)
            #     self.scores_gauss_his = []
            # After all iterations, print the best input and output
            best_idx = self.opt.y.argmin()
            best_design = self.opt.X.iloc[best_idx]
            best_design_reward = self.opt.y[best_idx]


            print(f"Best design: {best_design}, best reward: {best_design_reward}")
            self.best_design.append(best_design)
            self.best_design_reward.append(best_design_reward)

            print(f"Design distribution update took {time.time() - start_time:.2f} seconds")


            # Reset episode reward accumulator
            self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
            self.design_iteration = [1 for _ in range(self.n_envs_train)]
            self.episode_rewards = {}
            self.episode_length = {}
            self.episode_times = {}
            self.convergence_steps = {}

        if self.state == "gauss":
            # calculate the reward for the sampled designs
            self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_gauss)]
            self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_gauss)]
            self.convergence_time_avg = [0 for _ in range(self.n_envs_train // self.batch_size_gauss)]
            self.convergence_steps_avg = [0 for _ in range(self.n_envs_train // self.batch_size_gauss)]

            for i in range(self.n_envs_train // self.batch_size_gauss):
                # average batch reward
                sum_reward = 0
                total_episode_length = 0
                convergence_time = 0
                total_convergence_steps = 0
                for j in range(self.batch_size_gauss):
                    sum_reward += self.episode_rewards[i * self.batch_size_gauss + j] / self.design_iteration[
                        i * self.batch_size_gauss + j]
                    convergence_time += self.episode_times[i * self.batch_size_gauss + j] / self.design_iteration[
                        i * self.batch_size_gauss + j]
                    total_episode_length += self.episode_length[i * self.batch_size_gauss + j] / self.design_iteration[
                        i * self.batch_size_gauss + j]
                    total_convergence_steps = self.convergence_steps[i * self.batch_size_gauss + j] / self.design_iteration[
                        i * self.batch_size_gauss + j]

                self.design_rewards_avg[i] = sum_reward / self.batch_size_gauss
                self.episode_length_avg[i] = total_episode_length / self.batch_size_gauss
                self.convergence_steps_avg[i] = total_convergence_steps
                self.convergence_time_avg[i] = convergence_time / (self.batch_size_gauss * self.convergence_steps_avg[i])
                self.logger_reward.append(self.design_rewards_avg[i])
                self.logger_episode_length.append(self.episode_length_avg[i])
                self.logger_convergence_time.append(self.convergence_time_avg[i])
                current_design_params = \
                self.training_env.env_method('get_design_params', indices=[i * self.batch_size_gauss])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_gauss])[0]

                # find the best design
                if self.design_rewards_avg[i] > self.best_design_reward_gauss:
                    self.best_design_reward_gauss = self.design_rewards_avg[i]
                    self.best_design_gauss = current_design_params
                    self.mat_best_design_gauss.append(current_design_params)
                    self.model.save(
                        f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/aerofoil/rl/trained_model/bestGaussDesign_{self.model_name}")

            # Logging
            self.logger.record("mean reward", np.mean(self.logger_reward))
            self.logger.record("mean episode length", np.mean(self.logger_episode_length))
            self.logger.record("mean convergence time", np.mean(self.logger_convergence_time))
            self.logger_reward = []
            self.logger_episode_length = []
            self.logger_convergence_time = []

            # save the best model
            #if np.mean(self.design_rewards_avg) >= self.logger_reward_prev:
            self.model.save(
                    f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/aerofoil/rl/trained_model/{self.model_name}_{self.gauss_init_ctr}")
                #self.logger_reward_prev = np.mean(self.design_rewards_avg)
                #print(
                #    f"Model saved, reward: {self.logger_reward_prev}, iteration: {self.gauss_init_ctr}, best design: {self.best_design_gauss}")

            # Reset episode reward accumulator
            self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
            self.design_iteration = [1 for _ in range(self.n_envs_train)]
            self.episode_rewards = {}
            self.episode_times = {}
            self.episode_length = {}
            self.convergence_steps = {}

            output_data = {
                "design_params": np.array(self.mat_design_params),
                "reward": np.array(self.mat_reward),
                "convergence_time": np.array(self.mat_time),
                "iteration": np.array(self.mat_iteration),
                "best_design": np.array(self.mat_best_design_gauss),
                "design_space_lb": np.array(self.mat_design_lower_bound),
                "design_space_ub": np.array(self.mat_design_upper_bound),
            }

            print("saving matlab data...")

            file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/aerofoil/rl/trained_model/{self.mat_file_name}.mat"

            savemat(file_path, output_data)

        if self.sampling_hebo_ctr >= self.sampling_hebo_time_period:
            self.state = "gauss_init"
            self.sampling_hebo_ctr = 0

        elif self.sampling_gauss_ctr >= self.sampling_gauss_time_period:
            self.state = "hebo_init"
            self.sampling_gauss_ctr = 0

        return True

    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'infos' in self.locals:
            info = self.locals['infos']
            for i, times in enumerate(info):
                self.episode_times[i] = self.episode_times.get(i, 0) + times['convergence_time']
                self.convergence_steps[i] = self.convergence_steps.get(i, 0) + times['convergence_steps']

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done:
                    self.design_iteration[i] += 1
        return True


def main():
    # training parameters
    use_sde = False
    hidden_sizes_train = 64
    REWARD = np.array([0.90, 0.10])
    design_params_limits = np.array([0.01, 100.0])
    learning_rate_train = 0.0001
    n_epochs_train = 10


    LOAD_OLD_MODEL = True
    n_update_dist = 25
    n_number_dist = 8
    n_steps_train = 10000
    n_envs_train = 50
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 1000
    batch_size_train = 128
    global_iteration = 0
    TRAIN = False
    callback_func_name = "Hebo_callback"
    lr_std_schaff = 0.01
    lr_mean_schaff = 0.01

    # initialise the model PPO
    learning_rate_train = learning_rate_train

    onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                           net_arch=dict(vf=[hidden_sizes_train, hidden_sizes_train],
                                         pi=[hidden_sizes_train, hidden_sizes_train]))

    global_iteration += 1

    # Define unique initialization variables for each environment
    env_configs = [{'REWARD': REWARD, 'env_id': i, 'call_back': f"random_design"} for i in range(n_envs_train)]
    # Ensure we have configurations for each environment instance
    assert len(env_configs) == n_envs_train

    # Create function for each environment instance with its unique configuration
    env_fns = [lambda config=config: LQREnv(**config) for config in env_configs]

    # Create the vectorized environment using SubprocVecEnv directly
    vec_env = SubprocVecEnv(env_fns, start_method='fork')
    # model_name = f"Trial58_random_AerofoilEnv_length_random_design_start_upd_{n_update_dist}lr_mean{lr_std_schaff}_std_{lr_std_schaff}_sde{use_sde}_Tanh_Tsteps_{total_timesteps_train}_lr_{learning_rate_train}_hidden_sizes_{hidden_sizes_train}_CLpenalty_{REWARD[0]}_ACTIONpenalty_{REWARD[1]}_var_design"
    model_name = f"Trial189_RL"
    log_dir = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/AerofoilEnv/AerofoilEnv_tensorboard/TB_{model_name}"

    if LOAD_OLD_MODEL:
        new_model = []
        old_model = PPO.load(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/random_design/{model_name}.zip",
            env=vec_env)
        # Create a new model with the desired configuration
        new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train,
                        batch_size=batch_size_train, n_epochs=n_epochs_train,
                        use_sde=use_sde, ent_coef=entropy_coeff_train,
                        learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs,
                        device='cpu', verbose=1, tensorboard_log=log_dir)

        # Load the weights from the old model
        new_model.set_parameters(old_model.get_parameters())
    else:
        new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train, batch_size=batch_size_train,
                        n_epochs=n_epochs_train, use_sde=use_sde, ent_coef=entropy_coeff_train,
                        learning_rate=learning_rate_train,
                        policy_kwargs=onpolicy_kwargs, device='cpu', verbose=1, tensorboard_log=log_dir)
        print("New model created")

    # Train the new model
    print("Model training...")
    # Now you can continue training with the new model
    callback_funcs = {
        "Hebo_callback": (lambda : Hebo_callback(model_name=model_name, n_steps_train=n_steps_train,
                                                 n_envs_train=n_envs_train, batch_size_opt=1,
                                                 design_params_limits=design_params_limits, verbose=1)),
        "Schaff_callback": (lambda : Schaff_callback(model_name=model_name, n_update_dist=n_update_dist,
                                                     n_steps_train=n_steps_train, n_envs_train=n_envs_train,
                                                     lr_mean_schaff=lr_mean_schaff, lr_std_schaff=lr_std_schaff,
                                                     n_number_dist=n_number_dist, verbose=1)),
        "random_design": (lambda : random_design(model_name=model_name, n_steps_train=n_steps_train,
                                                 n_envs_train=n_envs_train, verbose=1)),
        "constant_design": (lambda : constant_design(model_name=model_name, n_steps_train=n_steps_train,
                                                     n_envs_train=n_envs_train, verbose=1)),
        "Hebo_Gauss_callback": (lambda: Hebo_Gauss_callback(model_name=model_name, model=new_model, n_steps_train=n_steps_train,
                                        n_envs_train=n_envs_train, verbose=1))
    }

    param_changer = None
    try:
        param_changer = callback_funcs[callback_func_name]()
    except:
        print(f"No valid callback function specified. Try any of the following: {list(callback_funcs.keys())}")
        return

    new_model.learn(total_timesteps=total_timesteps_train, progress_bar=True, callback=param_changer)
    print("Model trained, saving...")
    new_model.save(
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/HEBO/{model_name}")
    print("Model saved")
    vec_env.close()


if __name__ == '__main__':
    main()
