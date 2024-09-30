from stable_baselines3 import PPO
from env_cartpole import CartPoleEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from scipy.io import savemat
import time

def main():
    # Training parameters
    REWARD = np.array([1.0, 0.2, 0.5])
    n_envs_train = 8
    global_iteration = 0
    EVAL = True
    model_name = f"CartPoleEnv_Tanh_Tsteps_122880000_lr0.0001_hidden_sizes_256_lay2_rewardUpRight_1_obsLen_min1002_polemass_GaussMix_9"
    EVAL_MODEL_PATH = f"trained_model/{model_name}"
    
    # Load the model once outside of the loop
    eval_env = create_env(n_envs_train, REWARD)
    eval_model = PPO.load(EVAL_MODEL_PATH, env=eval_env)
    
    while True:
        global_iteration += 1 
        
        if EVAL:
            evaluate(eval_model, n_envs_train, model_name, global_iteration)

def create_env(n_envs, reward_config):
    # Define unique initialization variables for each environment
    env_configs = [{'REWARD': reward_config} for _ in range(n_envs)]
    
    # Create function for each environment instance with its unique configuration
    env_fns = [lambda config=config: CartPoleEnv(**config) for config in env_configs]
    
    # Create the vectorized environment using SubprocVecEnv
    return SubprocVecEnv(env_fns, start_method='fork')

def evaluate(model, n_envs, model_name, global_iteration):
    LOOP_COUNT = 1000000
    cart_position = [[] for _ in range(n_envs)]
    pole_position = [[] for _ in range(n_envs)]
    rewards = [[] for _ in range(n_envs)]
    effort = [[] for _ in range(n_envs)]
    avg_effort = [[] for _ in range(n_envs)]
    step = [[] for _ in range(n_envs)]
    average_effort_mat, steps, ep_reward_mat, pole_length,pole_position_mat,cart_position_mat = [], [], [], [], [], []
    ep_reward = np.zeros(n_envs)
    step = np.zeros(n_envs)
    obs = model.env.reset()
    start_time = time.time()
    print("starting evaluation")
    
    for num in range(LOOP_COUNT):
        if num % 10000 == 0:
            end_time = time.time()
            print(f"iteration: {num}, iteration per sec: {10000/(end_time-start_time)}")
            start_time = time.time()

        actions, _states = model.predict(obs)
        obs, rewards_batch, dones, infos = model.env.step(actions)
        
        for i in range(n_envs):
            
            cart_position[i].append(np.array(infos[i]['cart_pos']))        
            pole_position[i].append(np.array(infos[i]['pole_pos']))
            rewards[i].append(np.array(rewards_batch[i]))
            effort[i].append(np.array(infos[i]['effort']))
            avg_effort[i].append(np.array(infos[i]['effort']))
            ep_reward[i] += rewards_batch[i]
            step[i] += 1

            if dones[i]:
                average_effort_mat.append(np.mean(avg_effort[i]))
                cart_position_mat.append(np.array(cart_position[i]))
                pole_position_mat.append(np.array(pole_position[i]))
                steps.append(np.array(step[i]))
                pole_length.append(np.array(infos[i]['pole_length']))
                ep_reward_mat.append(np.array(ep_reward[i]))
                avg_effort[i] = []
                ep_reward[i] = 0
                step[i] = 0

        if num % 50000 == 0:
            # Save data in a MATLAB file
            output_data = {
                "cart_position": np.array(cart_position_mat),
                "pole_position": np.array(pole_position_mat),
                "pole_length": np.array(pole_length),
                "average_effort": np.array(average_effort_mat),
                "steps": np.array(steps),
                "ep_reward": np.array(ep_reward_mat)
            }
            print("saving to output.mat...")
            file_path = f"/home/divij/Documents/quadopter/cartpole_system/EVAL_{model_name}_{global_iteration}.mat"
            savemat(file_path, output_data)
            print("saved to output.mat")

if __name__ == '__main__':
    main()
