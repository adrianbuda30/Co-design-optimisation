import os
import time
from collections import OrderedDict
import pickle
import sys
import gym
from gym import wrappers
import numpy as np
from gym.envs.registration import register
from scipy.io import loadmat, savemat

from src.quadcopter_env import QuadcopterEnv

def main():
    register(
        id='Quadcopter-v0',
        entry_point='src.quadcopter_env:QuadcopterEnv',
    )

    env = gym.make('Quadcopter-v0')
    ob = env.reset()

    obs, acs, rewards, position, velocity, omega, R = [], [], [], [], [], [], []

    # Load .mat file
    data = loadmat('/home/divij/tum/input_cmd.mat')

    # Assume the key you're interested in is 'input_cmd'.
    # Adjust this to be the actual key for the data you want to use.
    data_key = 'w'

    # Check if the key exists in the .mat file and is 2D
    if data_key in data and len(data[data_key].shape) == 2:
        data_list = data[data_key].tolist()
    else:
        print(f"{data_key} not found in .mat file or it is not 2D")
        return

    for i in range(len(data_list)):

        acs = np.array(data_list[i])
        ob, rew, done, _ = env.step(acs)
        position.append(ob[0:3])        
        velocity.append(ob[3:6])
        omega.append(ob[6:9])
        R.append(ob[9:18])  # Reshape into 3x3 matrix before appending
        rewards.append(rew)
    # R = np.array(R)  # Convert list of matrices into a 3D array
    # R = np.transpose(R, (1, 2 , 0))

    print("step: ", i, "  " ,data_list[i])

    # Save the data
    output_data = {
        "rewards_python": np.array(rewards),
        "position_python": np.array(position),
        "velocity_python": np.array(velocity),
        "omega_python": np.array(omega),
        "R_python": np.array(R),

    }

    print("saving to output.mat...")
    savemat('/home/divij/tum/output.mat', output_data)
    

if __name__ == "__main__":
    main()
