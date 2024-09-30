from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.sb_quadcopter_env import QuadcopterEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from gymnasium import spaces
import torch
import numpy as np
from gaussMix_design_opt import DesignDistribution_log as DesignDistribution
import time



def main():

    # Initialize the 10 multi-variate distributions
    num_distributions = 4
    distributions = []
    min_arm_length = [0.1, 0.1, 0.1, 0.1]
    max_arm_length = [1, 1, 1, 1]
    for i in range(num_distributions):
        initial_mean = 0.17*np.random.rand(4)
        initial_cov = np.eye(4) * 1 + 0.001
        design_dist = DesignDistribution(initial_mean, initial_cov, alpha_mean=0.01, alpha_cov=0.01,
                                         min_values=min_arm_length, max_values=max_arm_length)
        distributions.append(design_dist)

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
    n_envs_train = 4
    entropy_coeff_train = 0.01
    total_timesteps_train = n_steps_train * n_envs_train
    batch_size_train = 512 * 5
    sampled_design_train = []

    eval_train = 0
    global_iteration = 0
    sample_distribution_time = 0
    num_iterations_train = 5
    update_iteration = 0
    evaluate_interval = 10

    while True:

        global_iteration += 1 
        # sample a batch of designs from each distribution for training
        # for start model training
        for iteration in range(num_iterations_train):
            sampled_design_train = []
            for design_dist in distributions:
                sampled_design_train_inter = design_dist.sample_design()
                sampled_design_train.append(sampled_design_train_inter)
            # Define unique initialization variables for each environment
            if global_iteration > sample_distribution_time:
                env_configs = [{'arm_length': value} for value in sampled_design_train]
            else:
                env_configs = [{'arm_length': np.array([0.17, 0.17, 0.17, 0.17])} for _ in sampled_design_eval]

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
            
            tensorboard_callback = TensorboardCallback(verbose=1)

            if LOAD_OLD_MODEL is True:
                # trained_model = PPO.load(model_name, env = vec_env)
                # new_model.policy.load_state_dict(trained_model.policy.state_dict())
                # new_model.policy.value_net.load_state_dict(trained_model.policy.value_net.state_dict())
                # print("orignal saved model loaded")
                trained_model = PPO.load("quadcopter_design_optmisation", env = vec_env)
                new_model.policy.load_state_dict(trained_model.policy.state_dict())
                new_model.policy.value_net.load_state_dict(trained_model.policy.value_net.state_dict())
                print("New saved model loaded")

            # Train the new model   
            print("Model training...")        
            # Now you can continue training with the new model
            new_model.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=tensorboard_callback)
            print("Model trained, saving...")
            new_model.save(f"quadcopter_design_optmisation")
            print("Model saved")
            LOAD_OLD_MODEL = True
            vec_env.close()


        if global_iteration > update_iteration: 
            # Start design optimisation algorithm
            print("Model evaluation...")
            eval_model = new_model
            batch_size = 1
            update_iterations = 5
            batch_env = 4
            env_steps = 5000
            

            for design_dist in distributions:
                start_time = time.time()
                intermidiate_rewards_mean = 0
                for _ in range(update_iterations):
                    for _ in range(batch_size):

                        sampled_design_eval = np.zeros((batch_env, 4))
                        for q in range(batch_env):
                            sampled_design_eval[q] = design_dist.sample_design()
                        eval_env_configs = [{'arm_length': value} for value in sampled_design_eval]
                        # Create function for each environment instance with its unique configuration
                        eval_env_fns = [lambda config=config: QuadcopterEnv(**config) for config in eval_env_configs]
                        # Create the vectorized environment using SubprocVecEnv directly
                        vec_eval_env = SubprocVecEnv(eval_env_fns, start_method='fork')
                        obs = vec_eval_env.reset()
                        # Initialize an empty list of lists for intermediate rewards
                        intermidiate_rewards = [[] for _ in range(batch_env)]
                        for i in range(env_steps):
                            action, _states = eval_model.predict(obs)
                            obs, rewards, dones, info = vec_eval_env.step(action)
                            for s in range(batch_env):
                                intermidiate_rewards[s].append(rewards[s])

                        idx_rewards = 0
                        for value in sampled_design_eval:
                            intermidiate_rewards_mean = np.mean(intermidiate_rewards[idx_rewards])
                            print(f"mean reward: {intermidiate_rewards_mean} for design: {value}")
                            design_dist.update_distribution([value], [intermidiate_rewards_mean])
                            idx_rewards += 1

                    vec_eval_env.close()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time} seconds")
                print(f"design distribution covariance: {design_dist.cov}, and mean: {design_dist.mean}")

        # env_steps_ranking = 20000
        # batch_size_ranking = 10

        # # Evaluate and select the best distributions every evaluate_interval iterations
        # if ((global_iteration + 1) % evaluate_interval == 0) and not(len(distributions) == 1):
        #     scores = []
        #     for design_dist in distributions:
        #         intermidiate_rewards_sum = 0
        #         intermidiate_rewards = []
        #         for i in range(batch_size_ranking):
        #             sampled_design_eval = design_dist.sample_design()
        #             chop_env_configs = [{'arm_length': sampled_design_eval}]
        #             # Create function for each environment instance with its unique configuration
        #             chop_env_fns = [lambda config=config: QuadcopterEnv(**config) for config in chop_env_configs]
        #             # Create the vectorized environment using SubprocVecEnv directly
        #             vec_chop_env = SubprocVecEnv(chop_env_fns, start_method='fork')
        #             obs = vec_chop_env.reset()

        #             for i in range(env_steps_ranking):
        #                 action, _states = eval_model.predict(obs)
        #                 obs, rewards, dones, info = vec_chop_env.step(action)
        #                 intermidiate_rewards.append(rewards)
        #             intermidiate_rewards_sum += np.sum(intermidiate_rewards)
        #         scores.append(intermidiate_rewards_sum)
        #     sorted_indices = np.argsort(scores)[::-1]
        #     # Keep only half of the best performing distributions
        #     distributions = [distributions[idx] for idx in sorted_indices[:len(distributions) // 2]]
        #     # Print the iteration and the best performing design
        #     print(f"Iteration: {global_iteration}, Best Mean: {distributions[0].mean}")
        
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # value = np.random.random()
        # self.logger.record("random_value", value)
        return True

if __name__ == '__main__':
    main()
