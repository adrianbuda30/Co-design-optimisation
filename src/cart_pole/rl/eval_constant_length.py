from stable_baselines3 import PPO
from env_cartpole import CartPoleEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from scipy.io import savemat
import time

def main():
    pole_length = 1.0
    episode_length_loop = 5120
    batch_size = 50
    mat_episode_length = []
    mat_rewards = []
    mat_pole_length = []


    # Training parameters
    while True:

        REWARD = np.array([1.0, 0.2, 0.5])

        n_envs = 1
        model_name = f"CartPole_constant_design_Tanh_Tsteps_8192000_lr_0.0001_hidden_sizes_256_reward_1.00.20.5_pole_length_{pole_length}_best_model"
        EVAL_MODEL_PATH = f"/home/divij/Documents/quadopter/src/cart_pole/rl/trained_model/constant_mass/{model_name}"
        matlab_model_name = f"CartPole_callback_constant_design_Tanh_Tsteps_8192000_lr_0.0001_hidden_sizes_256_reward_1.00.20.5_best_model"
        matlab_model_path = f"/home/divij/Documents/quadopter/src/cart_pole/rl/trained_model/constant_mass/{matlab_model_name}.mat"
        # Load the model once outside of the loop
        eval_env = create_env(n_envs, REWARD,pole_length)
        model = PPO.load(EVAL_MODEL_PATH, env=eval_env)
                

        for _ in range(batch_size):
            rewards = 0
            episode_length = 0
            obs = model.env.reset()
            for _ in range(episode_length_loop):
                actions, _states = model.predict(obs)
                obs, rewards_env, dones, infos = model.env.step(actions)
                rewards += rewards_env 
                episode_length += 1
                if dones:
                    break

            print(f"Episode length: {episode_length}, Rewards: {rewards}, Pole length: {pole_length}")
            mat_episode_length.append(episode_length)
            mat_rewards.append(rewards)
            mat_pole_length.append(pole_length)
        pole_length += 1.0

        # Save data in a MATLAB file
        output_data = {
            "episode_length": np.array(mat_episode_length),
            "rewards": np.array(mat_rewards),
            "pole_length": np.array(mat_pole_length)
            }

        savemat(matlab_model_path, output_data)
            

def create_env(n_envs, reward_config, pole_length):
    # Define unique initialization variables for each environment
    env_configs = [{'REWARD': reward_config, 'env_id': i, 'pole_length': pole_length, 'call_back' : f"constant_design"} for i in range(n_envs)]
    
    # Create function for each environment instance with its unique configuration
    env_fns = [lambda config=config: CartPoleEnv(**config) for config in env_configs]
    
    # Create the vectorized environment using SubprocVecEnv
    return SubprocVecEnv(env_fns, start_method='fork')


if __name__ == '__main__':
    main()
