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
    {"name": "x2", "def_value": 0.62},
    {"name": "x3", "def_value": 0.25},
    {"name": "x4", "def_value": 1800},
    {"name": "x5", "def_value": 370},
    {"name": "x6", "def_value": 390}
]

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
        self.state = 'propose_design'
        self.design_process = False  # Initialize to False
        self.mat_file_name = model_name
        self.save_recorded_data = n_steps_train * n_envs_train * 1
        self.reduce_batch_size = n_steps_train * n_envs_train * 1
        self.batch_size_opt = batch_size_opt


        # Initialize
        self.episode_rewards = {}
        self.episode_length = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
        self.episode_length_avg = [0 for _ in range(self.n_envs_train // self.batch_size_opt)]
        self.best_design = []
        self.best_design_reward = []
        self.average_length = []
        self.average_reward = []
        self.checker = [False for _ in range(self.n_envs_train)]

        np.set_printoptions(precision=5)

        space = DesignSpace().parse([
            {'name': 'x2', 'type': 'num', 'lb': 0.001, 'ub': 0.8},
            {'name': 'x3', 'type': 'num', 'lb': 0.001, 'ub': 0.8},
            {'name': 'x4', 'type': 'num', 'lb': 0.01, 'ub': 5000},
            {'name': 'x5', 'type': 'num', 'lb': 0.01, 'ub': 5000},
            {'name': 'x6', 'type': 'num', 'lb': 0.01, 'ub': 5000}
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

        # Reset episode reward accumulator
        self.design_rewards_avg = [0 for _ in range(self.n_envs_train)]
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.episode_rewards = {}
        self.episode_length = {}


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
            for i in range(self.n_envs_train // self.batch_size_opt):
                # average batch reward
                sum_reward = 0
                total_episode_length = 0
                for j in range(self.batch_size_opt):
                    sum_reward += self.episode_rewards[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]
                    total_episode_length += self.episode_length[i * self.batch_size_opt + j] / self.design_iteration[
                        i * self.batch_size_opt + j]
                self.design_rewards_avg[i] = sum_reward / self.batch_size_opt
                self.episode_length_avg[i] = total_episode_length / self.batch_size_opt
                self.average_length.append(self.episode_length_avg[i])
                self.average_reward.append(self.design_rewards_avg[i])
                score_array = np.array(self.design_rewards_avg[i]).reshape(-1, 1)  # Convert to NumPy array
                scores.append(-score_array)  # HEBO minimizes, so we need to negate the scores

                # Logging
                current_design_params = \
                self.training_env.env_method('get_design_params', indices=[i * self.batch_size_opt])[0]
                dist_env_id = self.training_env.env_method('get_env_id', indices=[i * self.batch_size_opt])[0]
                print(
                    f"Env ID: {dist_env_id}, mean reward: {self.design_rewards_avg[i]}, Mean episode length: {self.episode_length_avg[i]}, arm length: {current_design_params}")

                # Matlab logging
                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.design_rewards_avg[i])
                self.mat_iteration.append(self.episode_length_avg[i])

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
                    f"Env ID: {dist_env_id}, episode reward: {self.episode_rewards[i]}, mean reward: {self.episode_rewards[i] / self.design_iteration[i]}, design iter: {self.design_iteration[i]}, Mean episode length: {self.episode_length[i]}, arm length: {current_design_params}")
                self.logger.record("mean reward", self.episode_rewards[i] / self.design_iteration[i])
                self.logger.record("mean episode length", self.episode_length[i])

                # Matlab logging
                self.mat_design_params.append(current_design_params)
                self.mat_reward.append(self.episode_rewards[i] / self.design_iteration[i])
                self.mat_iteration.append(self.episode_length[i] / self.design_iteration[i])


        if self.num_timesteps % self.save_recorded_data == 0:
            output_data = {
                "design_params": np.array(self.mat_design_params),
                "reward": np.array(self.mat_reward),
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


def main():
    # training parameters
    use_sde = False
    hidden_sizes_train = 512
    REWARD = np.array([0.95, 0.05])
    design_params_limits = np.array([0.01, 100.0])
    learning_rate_train = 0.0001
    n_epochs_train = 10


    LOAD_OLD_MODEL = False
    n_update_dist = 25
    n_number_dist = 8
    n_steps_train = 10000
    n_envs_train = 20
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 1000
    batch_size_train = 128
    global_iteration = 0
    TRAIN = False
    callback_func_name = "random_design"
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
    env_fns = [lambda config=config: AerofoilEnv(**config) for config in env_configs]

    # Create the vectorized environment using SubprocVecEnv directly
    vec_env = SubprocVecEnv(env_fns, start_method='fork')
    # model_name = f"Trial58_random_AerofoilEnv_length_random_design_start_upd_{n_update_dist}lr_mean{lr_std_schaff}_std_{lr_std_schaff}_sde{use_sde}_Tanh_Tsteps_{total_timesteps_train}_lr_{learning_rate_train}_hidden_sizes_{hidden_sizes_train}_CLpenalty_{REWARD[0]}_ACTIONpenalty_{REWARD[1]}_var_design"
    model_name = f"Trial130_random_design"
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
                                                 n_envs_train=n_envs_train, batch_size_opt=5,
                                                 design_params_limits=design_params_limits, verbose=1))
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
            f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/random_design/{model_name}")
    print("Model saved")
    vec_env.close()


if __name__ == '__main__':
    main()
