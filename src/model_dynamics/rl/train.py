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
from env_quad_traj_circle import QuadcopterEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from gaussMix_design_opt import DesignDistribution_log as DesignDistribution
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from scipy.io import loadmat, savemat
from stable_baselines3.common.policies import ActorCriticPolicy

import torch.nn as nn
import torch
import math

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

    def __init__(self, observation_space, net_arch, activation_fn, design_obs=4, trajectory_obs=13):
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

        self.policy_net = self.build_mlp(input_dim - (self.design_obs + self.trajectory_obs) + self.encoder_layer_size_traj + self.encoder_layer_size_design, net_arch['pi'], activation_fn)
        self.value_net = self.build_mlp(input_dim - (self.design_obs + self.trajectory_obs) + self.encoder_layer_size_traj + self.encoder_layer_size_design, net_arch['vf'], activation_fn)
        
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
        obs_traj = obs[:, -self.design_obs-self.trajectory_obs:-self.design_obs]
        obs_rest = obs[:, :-self.design_obs-self.trajectory_obs]
        
        encoder_design = self.encoder_design(obs_design)
        encoder_traj = self.encoder_traj(obs_traj)

        combined_obs = torch.cat([encoder_design, encoder_traj, obs_rest], dim=1)

        policy_output = self.policy_net(combined_obs)
        value_output = self.value_net(combined_obs)

        return policy_output, value_output

    def forward_actor(self, obs):
        obs_design = obs[:, -self.design_obs:].float()
        obs_traj = obs[:, -self.design_obs-self.trajectory_obs:-self.design_obs].float()
        obs_rest = obs[:, :-self.design_obs-self.trajectory_obs].float()
        
        encoder_design = self.encoder_design(obs_design)
        encoder_traj = self.encoder_traj(obs_traj)

        combined_obs = torch.cat([encoder_design, encoder_traj, obs_rest], dim=1)
        latent_pi = self.policy_net(combined_obs)
        return latent_pi

    def forward_critic(self, obs):
        obs_design = obs[:, -self.design_obs:].float()
        obs_traj = obs[:, -self.design_obs-self.trajectory_obs:-self.design_obs].float()
        obs_rest = obs[:, :-self.design_obs-self.trajectory_obs].float()

        encoder_design = self.encoder_design(obs_design)
        encoder_traj = self.encoder_traj(obs_traj)

        combined_obs = torch.cat([encoder_design, encoder_traj, obs_rest], dim=1)
        latent_vf = self.value_net(combined_obs)
        return latent_vf


def main():
    #training parameters
    use_sde = False
    hidden_sizes_train = 256
    REWARD = np.array([0.0, 1.0, 0.2])
    arm_length_limits = np.array([0.01, 2.0])
    learning_rate_train = 0.0001
    n_epochs_train = 10
    LOAD_OLD_MODEL = True
    n_update_dist = 25
    n_number_dist = 8
    n_steps_train = 512 * 10
    n_envs_train =  500
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 100

    batch_size_train = 512 * 2
    global_iteration = 0
    TRAIN = True
    CALL_BACK_FUNC = f"Hebo_callback"
    lr_std_schaff = 0.01
    lr_mean_schaff = 0.01

    while True:
        #initialise the model PPO
        learning_rate_train = learning_rate_train

        onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                                net_arch=dict(vf=[hidden_sizes_train, hidden_sizes_train], pi=[hidden_sizes_train, hidden_sizes_train]))
        
        global_iteration += 1 

        # Define unique initialization variables for each environment
        env_configs = [{'REWARD': REWARD, 'env_id': i , 'n_steps_train': n_steps_train, 'arm_length_limits' : arm_length_limits} for i in range(n_envs_train)]
        # Ensure we have configurations for each environment instance
        assert len(env_configs) == n_envs_train 

        # Create function for each environment instance with its unique configuration
        env_fns = [lambda config=config: QuadcopterEnv(**config) for config in env_configs]

        # Create the vectorized environment using SubprocVecEnv directly
        vec_env = SubprocVecEnv(env_fns, start_method='fork')
        model_name = f"QuadcopterCircle_length_{CALL_BACK_FUNC}_start_upd_{n_update_dist}lr_mean{lr_std_schaff}_std_{lr_std_schaff}_sde{use_sde}_Tanh_Tsteps_{total_timesteps_train}_lr_{learning_rate_train}_hidden_sizes_{hidden_sizes_train}_POSreward_{REWARD[0]}_VELreward_{REWARD[1]}_omega_pen_{REWARD[2]}_var_design"
        log_dir = f"/home/divij/Documents/quadopter/src/model_dynamics/Quadcopter_tensorboard/TB_{model_name}"
          
        if LOAD_OLD_MODEL is True:
            new_model = []
            old_model = PPO.load("/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/random_design/QuadcopterCircle_length_random_design_start_upd_25lr_mean0.01_std_0.01_sdeFalse_Tanh_Tsteps_256000000_lr_0.0001_hidden_sizes_256_POSreward_0.0_VELreward_1.0_omega_pen_0.2_var_design", env = vec_env)
            # Create a new model with the desired configuration
            new_model = PPO(CustomPolicy, env=vec_env, n_steps=n_steps_train, 
                            batch_size=batch_size_train, n_epochs=n_epochs_train, 
                            use_sde=use_sde, ent_coef=entropy_coeff_train, 
                            learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs, 
                            device='cpu', verbose=1, tensorboard_log=log_dir)
            
            # Load the weights from the old model
            new_model.set_parameters(old_model.get_parameters())
        else:
            new_model = PPO(CustomPolicy, env = vec_env, n_steps = n_steps_train, batch_size = batch_size_train, 
                n_epochs=n_epochs_train, use_sde = use_sde , ent_coef = entropy_coeff_train, learning_rate = learning_rate_train,
                policy_kwargs=onpolicy_kwargs, device ='cpu',verbose=1, tensorboard_log=log_dir)
            print("New model created")


        # Train the new model   
        print("Model training...")        
        # Now you can continue training with the new model
        if CALL_BACK_FUNC is f"Hebo_callback":
            param_changer = Hebo_callback(model_name = model_name, n_steps_train = n_steps_train, n_envs_train = n_envs_train, arm_length_limits=arm_length_limits,verbose=1)
        elif CALL_BACK_FUNC is f"Schaff_callback":
            param_changer = Schaff_callback(model_name = model_name,n_update_dist=n_update_dist, n_steps_train = n_steps_train, n_envs_train = n_envs_train, lr_mean_schaff = lr_mean_schaff, lr_std_schaff = lr_std_schaff,n_number_dist = n_number_dist, verbose=1)
        elif CALL_BACK_FUNC is f"random_design":
            param_changer = random_design(model_name = model_name, n_steps_train = n_steps_train, n_envs_train = n_envs_train, verbose=1)
        else:
            print("No callback function specified")
            break

        new_model.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=param_changer)
        print("Model trained, saving...")
        new_model.save(f"/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/random_design/{model_name}")
        print("Model saved")
        LOAD_OLD_MODEL = True
        vec_env.close()
        break

class random_design(BaseCallback):
    def __init__(self, model_name = f"matfile" ,n_steps_train=512 * 10, n_envs_train=8, verbose=0):

        super(random_design, self).__init__(verbose)
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_arm_length = []
        self.mat_reward = []
        self.mat_iteration = []
        self.model_name = model_name
    
    def _on_rollout_start(self) -> bool:
        #reset the environments
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
            for i, done in enumerate(dones):
                if done or self.episode_length[i] >= self.n_steps_train:
                    current_arm_length = self.training_env.env_method('get_arm_length', indices=[i])[0]
                    dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                    print(f"Env ID: {dist_env_id}, episode reward: {self.episode_rewards[i]},  Mean episode length: {self.episode_length[i]}, arm length: {current_arm_length}")
                    self.logger.record("mean reward", self.episode_rewards[i])
                    self.logger.record("mean episode length", self.episode_length[i])

                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0
        return True
    
    def _on_rollout_end(self) -> bool:
        self.model.save(f"/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/random_design/{self.model_name}")
        return True


class Schaff_callback(BaseCallback):
    def __init__(self,model_name = f"matfile" ,n_update_dist = 1000,n_steps_train=512 * 10, n_envs_train=8,lr_mean_schaff=0.01,lr_std_schaff=0.01, n_number_dist = 8, verbose=0):

        super(Schaff_callback, self).__init__(verbose)
        self.episode_rewards = {}
        self.episode_length = {}
        self.model_name = model_name
        self.mat_file_name = model_name
        # Initialize the distributions
        self.num_distributions = n_number_dist
        self.num_envs = n_envs_train
        self.distributions = []
        self.min_arm_length = [1.0, 1.0, 1.0, 1.0]
        self.max_arm_length = [200.0, 200.0, 200.0, 200.0]
        self.n_steps_train = n_steps_train
        self.steps_update_distribution = n_steps_train * n_envs_train * n_update_dist
        self.steps_chop_distribution = n_steps_train * n_envs_train * 50
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.model_save_interval = n_steps_train * n_envs_train * 1
        np.set_printoptions(precision=2)
        #initialize matlab data
        self.mat_dist_mean = [[] for _ in range(self.num_distributions)]
        self.mat_dist_std = [[] for _ in range(self.num_distributions)]
        self.mat_arm_length = [[] for _ in range(self.num_distributions)]
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
                                    for min_val, max_val in zip(self.min_arm_length, self.max_arm_length)])
            # self.initial_std = np.ones(4, dtype=np.float32) * 0.01  # Initialize std deviation as you prefer
            self.initial_std = np.array([(max_val - min_val) / (2 * z) 
                                    for min_val, max_val, max_val in zip(self.min_arm_length, self.max_arm_length, self.max_arm_length)])
    
            self.design_dist = DesignDistribution(self.initial_mean, self.initial_std, min_parameters = self.min_arm_length, max_parameters = self.max_arm_length, lr_mean=lr_mean_schaff, lr_std=lr_std_schaff)
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

                        current_arm_length = self.training_env.env_method('get_arm_length', indices=[i])[0]
                        print(f"Env ID: {dist_env_id}, Mean episode length: {self.episode_length[i]} , Current arm length: {current_arm_length}, Mean reward: {total_episode_mean_reward}, distribution mean: {self.distributions[(dist_env_id)].get_mean()}, distribution std: {self.distributions[(dist_env_id)].get_std()}")
                        self.logger.record("mean reward", self.episode_rewards[i])
                        self.logger.record("mean episode length", self.episode_length[i])
                        self.Schaffs_batch_reward[dist_env_id].append(torch_reward)
                        self.Schaffs_batch_design[dist_env_id].append(current_arm_length*100)
                        self.Schaffs_batch_size_iter[dist_env_id] += 1

                        # Make batches before updating
                        # Update the distributions based on the episode reward
                        if self.Schaffs_batch_size_iter[dist_env_id] >= self.Schaffs_batch_size:
                            self.distributions[(dist_env_id)].update_distribution(self.Schaffs_batch_design[dist_env_id], self.Schaffs_batch_design[dist_env_id])
                            self.Schaffs_batch_design[dist_env_id] = []
                            self.Schaffs_batch_reward[dist_env_id] = []
                            self.Schaffs_batch_size_iter[dist_env_id] = 0

                        # Modify the environment parameter based on the episode reward
                        new_arm_length = self.distributions[(dist_env_id)].sample_design().detach().numpy()
                        new_arm_length = np.clip(new_arm_length, self.min_arm_length, self.max_arm_length)

                        # Calling the set_arm_length method of the environment
                        self.training_env.env_method('set_arm_length', new_arm_length/100, indices=[i])

                        # Logging
                        self.mat_dist_mean[dist_env_id].append(self.distributions[(dist_env_id)].get_mean())
                        self.mat_dist_std[dist_env_id].append(self.distributions[(dist_env_id)].get_std())
                        self.mat_arm_length[dist_env_id].append(current_arm_length)
                        self.mat_reward[dist_env_id].append(self.episode_rewards[i])  
                        self.mat_mean_reward[dist_env_id].append(total_episode_mean_reward)
                        self.mat_iteration[dist_env_id].append(self.episode_length[i])                     
                    
                        # Reset episode reward accumulator
                        self.episode_rewards[i] = 0
                        self.episode_length[i] = 0
                        total_episode_mean_reward = 0
            
            #save matlab data
            if self.num_timesteps % self.save_recorded_data == 0:
                output_data = {
                    "dist_mean": np.array(self.mat_dist_mean),
                    "dist_std": np.array(self.mat_dist_std),
                    "arm_length": np.array(self.mat_arm_length),
                    "reward": np.array(self.mat_reward),
                    "mean_reward": np.array(self.mat_mean_reward),
                    "iteration": np.array(self.mat_iteration)
                }
                print("saving to output.mat...")
                self.mat_iter += 1 
                file_path = f"/home/divij/Documents/quadopter/MultirotorSim_Vervoorst/{self.mat_file_name}_{self.mat_iter}.mat"
                savemat(file_path, output_data)
                self.model.save(f"/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/{self.model_name}_{self.mat_iter}")
            #chop low performing distributions
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
                    self.mat_arm_length = [self.mat_arm_length[i] for i in top_indices]
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
                            self.training_env.env_method('set_env_id', distribution_id, indices=[i * envs_per_distribution + k])

                    print(f"Kept {len(top_indices)} top-performing distributions.")
                    print(f"New distribution means: {[self.distributions[i].get_mean() for i in range(len(self.distributions))]}")
                    print(f"New distribution stds: {[self.distributions[i].get_std() for i in range(len(self.distributions))]}")
                    print(f"New ditribution env IDs: {[self.training_env.env_method('get_env_id', indices=[i])[0] for i in range(self.num_envs)]}")
                    print(f"New distribution mean rewards: {[np.mean(self.mat_mean_reward[i]) for i in range(len(self.mat_mean_reward))]}")

        else:
            if self.num_timesteps % self.model_save_interval == 0:
                self.model.save(f"/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/{self.model_name}_{self.mat_iter}")
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
                        print(f"Env ID: {i}, Mean reward: {self.episode_rewards[i]}, Mean episode length: {self.episode_length[i]}, arm length: {self.training_env.env_method('get_arm_length', indices=[i])[0]}")
                        # Reset episode reward accumulator
                        self.episode_rewards[i] = 0
                        self.episode_length[i] = 0

    def convert_range(self,x, min_x, max_x, min_y, max_y):
        return (x - min_x) / (max_x - min_x) * (max_y - min_y) + min_y

class Hebo_callback(BaseCallback):

    def __init__(self,model_name = f"matfile" ,n_steps_train=512 * 10, n_envs_train=8, arm_length_limits = np.array([0.01, 2.0]), verbose=0):
        
        super(Hebo_callback, self).__init__(verbose)

        self.batch_iterations = n_steps_train * n_envs_train
        self.steps_update_distribution = self.batch_iterations * 0 # Set to batch_iterations * 1 for clarity
        self.n_envs_train = n_envs_train
        self.model_name = model_name
        self.min_arm_length = arm_length_limits[0]
        self.max_arm_length = arm_length_limits[1]

        self.distributions = []
        self.mat_arm_length = []
        self.mat_reward = []
        self.mat_iteration = []
        self.state = 'propose_design'
        self.design_process = False  # Initialize to False
        self.mat_file_name = model_name
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.reduce_batch_size = n_steps_train * n_envs_train * 1
        self.batch_size_opt = 5
        
        # Initialize 
        self.episode_rewards = {}
        self.episode_length = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train//self.batch_size_opt)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train//self.batch_size_opt)]
        self.best_design = []
        self.best_design_reward = []

        np.set_printoptions(precision=3)

        space = DesignSpace().parse([
            {'name': 'x1', 'type': 'num', 'lb': self.min_arm_length, 'ub': self.max_arm_length},
            {'name': 'x2', 'type': 'num', 'lb': self.min_arm_length, 'ub': self.max_arm_length},
            {'name': 'x3', 'type': 'num', 'lb': self.min_arm_length, 'ub': self.max_arm_length},
            {'name': 'x4', 'type': 'num', 'lb': self.min_arm_length, 'ub': self.max_arm_length}
        ])
        self.opt = HEBO(space)
        print("Initialiastion callback: ")

    def on_training_start(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> bool:
        print("Training started")
        print(self.n_envs_train)
        #set the environment id for each environment
        for i in range(self.n_envs_train):
            self.training_env.env_method('set_env_id', i, indices=[i])

        return True

    def _on_rollout_start(self) -> bool:
            
        #reset the environments
        self.training_env.env_method('reset', indices=range(self.n_envs_train))

        print("...Updating the new lengths...")
        start_time = time.time()
        self.rec = self.opt.suggest(n_suggestions=self.n_envs_train // self.batch_size_opt)
        
        for i in range(self.n_envs_train//self.batch_size_opt):
            for j in range(self.batch_size_opt):
                new_arm_length = self.rec.values[i]
                self.training_env.env_method('set_arm_length', new_arm_length, indices=[i*self.batch_size_opt + j])
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                print(f"env: {i*self.batch_size_opt + j:<1.2f}, real id:{dist_env_id}, pole length: {new_arm_length}")

        print(f"Design proposal took {time.time() - start_time:.2f} seconds")
        
        return True

    def _on_rollout_end(self) -> bool:
        

        if self.num_timesteps >= self.steps_update_distribution-self.n_envs_train:

            print("...Starting design distribution update...")
            start_time = time.time()

            scores = []
            
            self.design_rewards_avg = [0 for _ in range(self.n_envs_train//self.batch_size_opt)]
            self.episode_length_avg = [0 for _ in range(self.n_envs_train//self.batch_size_opt)]
            for i in range(self.n_envs_train//self.batch_size_opt):
                #average batch reward
                sum_reward = 0
                total_episode_length = 0
                for j in range(self.batch_size_opt):
                    sum_reward += self.episode_rewards[i*self.batch_size_opt + j]/self.design_iteration[i*self.batch_size_opt + j]
                    total_episode_length += self.episode_length[i*self.batch_size_opt + j]/self.design_iteration[i*self.batch_size_opt + j]
                self.design_rewards_avg[i] = sum_reward/self.batch_size_opt
                self.episode_length_avg[i] = total_episode_length/self.batch_size_opt
                score_array = np.array(self.design_rewards_avg[i]).reshape(-1, 1)  # Convert to NumPy array
                scores.append(-score_array) # HEBO minimizes, so we need to negate the scores
                
                # Logging
                current_arm_length = self.training_env.env_method('get_arm_length', indices=[i*self.batch_size_opt])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i*self.batch_size_opt])[0]
                print(f"Env ID: {dist_env_id}, mean reward: {self.design_rewards_avg[i]}, Mean episode length: {self.episode_length_avg[i]}, arm length: {current_arm_length}")
                self.logger.record("mean reward", self.design_rewards_avg[i])
                self.logger.record("mean episode length", self.episode_length_avg[i])
                
                # Matlab logging
                self.mat_arm_length.append(current_arm_length)
                self.mat_reward.append(self.design_rewards_avg[i])  
                self.mat_iteration.append(self.episode_length_avg[i])
            
            
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

                current_arm_length = self.training_env.env_method('get_arm_length', indices=[i])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i])[0]
                print(f"Env ID: {dist_env_id}, episode reward: {self.episode_rewards[i]}, mean reward: {self.episode_rewards[i]/self.design_iteration[i]}, design iter: {self.design_iteration[i]}, Mean episode length: {self.episode_length[i]}, arm length: {current_arm_length}")
                self.logger.record("mean reward", self.episode_rewards[i]/self.design_iteration[i])
                self.logger.record("mean episode length", self.episode_length[i])
                
                # Matlab logging
                self.mat_arm_length.append(current_arm_length)
                self.mat_reward.append(self.episode_rewards[i]/self.design_iteration[i])  
                self.mat_iteration.append(self.episode_length[i]/self.design_iteration[i])

        # Reset episode reward accumulator
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.episode_rewards = {}
        self.episode_length = {}   


        if self.num_timesteps % self.save_recorded_data == 0:
            output_data = {
                "arm_length": np.array(self.mat_arm_length),
                "reward": np.array(self.mat_reward),
                "iteration": np.array(self.mat_iteration)
            }
            print("saving matla data...")
            file_path = f"/home/divij/Documents/quadopter/MultirotorSim_Vervoorst/{self.mat_file_name}.mat"
            savemat(file_path, output_data)
            print("saving current model...")
            self.model.save(f"/home/divij/Documents/quadopter/src/model_dynamics/rl/trained_model/{self.model_name}")
            print("Model saved")

        #increasing the batch size with each iteration
        # if self.num_timesteps % self.reduce_batch_size == 0:
        #     print("Increasing batch size...")
        #     self.batch_size_opt = 2 * self.batch_size_opt
        #     if self.batch_size_opt > self.n_envs_train:
        #         self.batch_size_opt = self.n_envs_train

        return True

    def _on_step(self) -> bool:

        if self.num_timesteps >= self.steps_update_distribution-self.n_envs_train:
            
            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                for i, reward in enumerate(rewards):
                    self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                    self.episode_length[i] = self.episode_length.get(i, 0) + 1

            if 'dones' in self.locals:
                dones = self.locals['dones']
                for i, done in enumerate(dones):
                    if done:
                        self.design_iteration[i] += 1
        return True

if __name__ == '__main__':
    main()


# Best design: x1    0.573797
# x2    0.528027
# x3    1.012275
# x4    0.481558
# Name: 549, dtype: float64, best reward: [-3392.278]
# Design distribution update took 0.36 seconds
