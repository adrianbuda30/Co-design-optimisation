import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.sb_quadcopter_env import QuadcopterEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
from gymnasium import spaces
import torch
import numpy as np
from gaussMix_design_opt import DesignDistribution_pytorch as DesignDistribution
import time
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

def main():

    while True:
        #initialise the model PPO
        onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                                net_arch=dict(vf=[256, 256], pi=[256, 256])
                                )
        
        #training parameters
        hidden_sizes_train = 256
        reward_func_train = 10025025025025
        learning_rate_train = 0.0003
        n_epochs_train = 10
        LOAD_OLD_MODEL = True
        n_steps_train = 512 * 10
        n_envs_train = 8
        entropy_coeff_train = 0.01
        total_timesteps_train = n_steps_train * n_envs_train * 100
        batch_size_train = 512 * 5
        sampled_design_train = []
        global_iteration = 0



        global_iteration += 1 

        # Define unique initialization variables for each environment
        env_configs = [{'arm_length': np.array([0.17, 0.17, 0.17, 0.17])} for _ in range(n_envs_train)]

        # # Create function for each environment instance with its unique configuration
        # env_fns = [lambda config=config: QuadcopterEnv(**config) for config in env_configs]
        
        # # Create the vectorized environment using SubprocVecEnv directly
        # vec_env = SubprocVecEnv(env_fns, start_method='fork')
        
        # # Wrap the environment with VecCheckNan for debugging
        # vec_env = VecCheckNan(vec_env, raise_exception=True)


        # Ensure we have configurations for each environment instance
        assert len(env_configs) == n_envs_train 

        # Create function for each environment instance with its unique configuration
        env_fns = [lambda config=config: QuadcopterEnv(**config) for config in env_configs]

        # Create the vectorized environment using SubprocVecEnv directly
        vec_env = SubprocVecEnv(env_fns, start_method='fork')

        log_dir = f"quadcopter_tensorboard/lr_{learning_rate_train}_bs_{batch_size_train}_epochs_{n_epochs_train}_n_steps_{n_steps_train}_n_envs_{n_envs_train}_hidden_sizes_{hidden_sizes_train}_reward_{reward_func_train}_freq_10hz_entropy_coeff_{entropy_coeff_train}"
        model_name = f"quadcopter_PPO_Hovering_complex_propeller_lr_{learning_rate_train}_bs_{batch_size_train}_epochs_{n_epochs_train}_n_steps_{n_steps_train}_n_envs_{n_envs_train}_hidden_sizes_{hidden_sizes_train}_reward_{reward_func_train}_freq_10hz_entropy_coeff_{entropy_coeff_train}"
        new_model = PPO("MlpPolicy", env = vec_env, n_steps = n_steps_train, batch_size = batch_size_train, 
                        n_epochs=n_epochs_train, ent_coef = entropy_coeff_train, learning_rate = learning_rate_train,policy_kwargs=onpolicy_kwargs,
                        device ='auto',verbose=1, tensorboard_log=log_dir)
        
        # if LOAD_OLD_MODEL is True:
        #     trained_model = PPO.load("quadcopter_PPO_Hovering_complex_propeller_lr_0.0003_bs_2560_epochs_10_n_steps_2560_n_envs_8_hidden_sizes_256_reward_10025025025025_freq_10hz_entropy_coeff_0.01", env = vec_env)
        #     new_model.policy.load_state_dict(trained_model.policy.state_dict())
        #     new_model.policy.value_net.load_state_dict(trained_model.policy.value_net.state_dict())
        #     print("orignal saved model loaded")
        if LOAD_OLD_MODEL is True:
            trained_model = PPO.load("quadcopter_tube_new_reward", env = vec_env)
            new_model.policy.load_state_dict(trained_model.policy.state_dict())
            new_model.policy.value_net.load_state_dict(trained_model.policy.value_net.state_dict())
            print("New saved model loaded")

        # Train the new model   
        print("Model training...")        
        # Now you can continue training with the new model
        param_changer = RewardBasedHeboCallback(n_steps_train = n_steps_train, n_envs_train = n_envs_train,verbose=1)
        new_model.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=param_changer)
        print("Model trained, saving...")
        new_model.save(f"quadcopter_tube_new_reward")
        print("Model saved")
        LOAD_OLD_MODEL = True
        vec_env.close()

class RewardBasedParameterChangerCallback(BaseCallback):
    def __init__(self, verbose=0):

        super(RewardBasedParameterChangerCallback, self).__init__(verbose)
        self.episode_rewards = {}
        self.rewards_iteration = {}
        # Initialize the 10 multi-variate distributions
        num_distributions = 2
        self.distributions = []
        min_arm_length = [10, 10, 10, 10]
        max_arm_length = [40, 40, 40, 40]
        self.steps_update_distribution = 0
        np.set_printoptions(precision=2)

        for i in range(num_distributions):
            self.initial_mean = np.array([min_val + (max_val - min_val) * np.random.rand() 
                                    for min_val, max_val in zip(min_arm_length, max_arm_length)])
             # Initialize the covariance matrix with random values for variance in each dimension
            # You might also wish to add random correlation terms (off-diagonal elements), but
            # remember that the resulting matrix must be positive-definite.
            self.initial_cov = np.diag(np.random.rand(4) * 0.5) + 0.001

            self.design_dist = DesignDistribution(self.initial_mean, self.initial_cov, 
                                                alpha_mean=0.01, alpha_cov=0.01,
                                                min_values=min_arm_length, max_values=max_arm_length)
            self.distributions.append(self.design_dist)


    def _on_step(self) -> bool:


        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.rewards_iteration[i] = self.rewards_iteration.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done:
                    if self.num_timesteps > self.steps_update_distribution:
                        total_episode_mean_reward = self.episode_rewards[i] / self.rewards_iteration[i]
                        current_arm_length = self.training_env.env_method('get_arm_length', indices=[i])[0]
                        # Update the distribution based on the episode reward
                        print("arm :",current_arm_length)
                        self.distributions[(i // 4)].update_distribution([current_arm_length], [total_episode_mean_reward])
                        # Modify the environment parameter based on the episode reward
                        new_arm_length = self.distributions[(i // 4)].sample_design()
                        # Calling the set_arm_length method of the environment
                        self.training_env.env_method('set_arm_length', new_arm_length, indices=[i])
                        updated_arm_length = self.training_env.env_method('get_arm_length', indices=[i])[0]
                        # print(f"env: {i:<1.2f} iter: {self.rewards_iteration[i]:<1.2f}  oldArmlength: {current_arm_length} Meanreward: {total_episode_mean_reward:<1.2f} New arm length {updated_arm_length} Mean: {(self.distributions[(i // 4)].mean)}")
                        # print(f"{self.distributions[(i // 4)].cov}")

                    # Reset episode reward accumulator
                    self.episode_rewards[i] = 0
                    self.rewards_iteration[i] = 0
        
        return True

class RewardBasedHeboCallback(BaseCallback):
    def __init__(self, n_steps_train=512 * 10, n_envs_train=8, verbose=0):
        super(RewardBasedHeboCallback, self).__init__(verbose)

        self.batch_iterations = n_steps_train * n_envs_train
        self.steps_update_distribution = self.batch_iterations * 800 # Set to batch_iterations * 1 for clarity

        # Initialize 
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.design_rewards = {}
        self.design_counter = {}
        self.distributions = []
        self.state = 'propose_design'
        self.design_iteration = 0  # Renamed for typo correction
        self.design_process = False  # Initialize to False
        self.design_process_iteration = 0  # Renamed for typo correction
        self.tube_radius = 0.3
        self.tube_radius_reward_decay = 0.90
        np.set_printoptions(precision=2)

        space = DesignSpace().parse([
            {'name': 'x1', 'type': 'num', 'lb': 10, 'ub': 40},
            {'name': 'x2', 'type': 'num', 'lb': 10, 'ub': 40},
            {'name': 'x3', 'type': 'num', 'lb': 10, 'ub': 40},
            {'name': 'x4', 'type': 'num', 'lb': 10, 'ub': 40}
        ])
        self.opt = HEBO(space)
        print("Initialiastion callback: ")

    def _on_step(self) -> bool:

        # if self.num_timesteps % (self.batch_iterations*10) == 0:
        #     self.tube_radius = self.tube_radius * self.tube_radius_reward_decay
        #     for i in range(8):
        #         self.training_env.env_method('set_tube_radius', self.tube_radius, indices=[i])


        if self.num_timesteps >= self.steps_update_distribution-8:
            
            if self.state == 'observe_design':
                if 'rewards' in self.locals:
                    rewards = self.locals['rewards']
                    for i, reward in enumerate(rewards):
                        self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                        self.rewards_iteration[i] = self.rewards_iteration.get(i, 0) + 1

                if 'dones' in self.locals:
                    dones = self.locals['dones']
                    for i, done in enumerate(dones):
                        if done:
                            self.design_rewards[i] = self.design_rewards.get(i, 0) + self.episode_rewards.get(i, 0)
                            self.design_counter[i] = self.design_counter.get(i, 0) + 1

                            # Logging
                            self.logger.record("mean reward", self.episode_rewards[i])
                            self.logger.record("mean episode length", self.rewards_iteration[i])

                            # Reset
                            self.episode_rewards[i] = 0
                            self.rewards_iteration[i] = 0

                if self.design_process_iteration == self.batch_iterations:
                    self.state = 'update_design_distribution'
                    self.design_process_iteration = 0
            if self.state == 'update_design_distribution':
                print("Starting design distribution update...")
                start_time = time.time()

                print("Design used for update: ")
                for i in range(8):
                    updated_arm_length = self.training_env.env_method('get_arm_length', indices=[i])[0]
                    print(f"env: {i:<1.2f} arm length: {updated_arm_length}")

                scores = []
                for j in range(8):
                    self.design_rewards[j] /= self.design_counter.get(j, 1)
                    score_array = np.array(self.design_rewards[j]).reshape(-1, 1)  # Convert to NumPy array
                    scores.append(-score_array) # HEBO minimizes, so we need to negate the scores
                scores = np.array(scores)  # Make sure the outer list is also a NumPy array
                self.opt.observe(self.rec, scores)
                self.state = 'propose_design'
                print(f"Design distribution update took {time.time() - start_time:.2f} seconds")

            if self.state == 'propose_design':
                print("New proposed design: ")
                start_time = time.time()
                self.rec = self.opt.suggest(n_suggestions=8)
                for i in range(8):
                    new_arm_length = self.rec.values[i]
                    self.training_env.env_method('set_arm_length', new_arm_length, indices=[i])
                    print(f"env: {i:<1.2f} arm length: {new_arm_length/100}")
                self.state = 'observe_design'
                self.design_rewards = {}
                self.design_counter = {}
                print(f"Design proposal took {time.time() - start_time:.2f} seconds")

            # if ((self.num_timesteps % self.batch_iterations) == 0):

            self.design_process_iteration += 8   

        return True


if __name__ == '__main__':
    main()
