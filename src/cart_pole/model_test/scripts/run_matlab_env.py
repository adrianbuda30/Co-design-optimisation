from stable_baselines3 import PPO
from env_cartpole import CartPoleEnv
import numpy as np
from scipy.io import savemat
import time
import os

def main():
    # Training parameters
    REWARD = np.array([1.0, 0.2])
    global_iteration = 0
    EVAL = True
 
    # Load the model once outside of the loop
    eval_env = create_env(REWARD)
    
    evaluate(eval_env)


def create_env(reward_config):
    return CartPoleEnv(REWARD=reward_config)

def evaluate(eval_env):
    LOOP_COUNT = 1000000
    cart_position = []
    cart_vel = []
    pole_position = []
    pole_vel = []

    start_time = time.time()
    obs = eval_env.reset()
    print("starting evaluation")
    
    for num in range(LOOP_COUNT):
        if num % 10000 == 0:
            end_time = time.time()
            print(f"iteration: {num}, iteration per sec: {10000/(end_time-start_time)}")
            start_time = time.time()

        actions, _states = eval_env.predict(obs)
        obs, _, done, info = eval_env.step(actions)
        
        cart_position.append(obs[0])
        cart_vel = np.array(obs[1])       
        pole_position.append(obs[2])
        pole_vel.append(obs[3])

        if num % 10000 == 0:
            # Save data in a MATLAB file
            output_data = {
                "cart_position": np.array(cart_position),
                "cart_vel": np.array(cart_vel),
                "pole_position": np.array(pole_position),
                "pole_vel": np.array(pole_vel)
            }
            print("saving to output.mat...")

            file_path = f"data/manual_simulink.mat"

            savemat(file_path, output_data)
            print("saved to output.mat")

if __name__ == '__main__':
    main()
