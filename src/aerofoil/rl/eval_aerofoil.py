from stable_baselines3 import PPO
from env_aerofoil import AerofoilEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from scipy.io import loadmat, savemat
import time

def main():


    # Load the joint torque data from the input.mat file
    mat_data = loadmat("/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/delta_ddot.mat")
    delta_ddot_input_values = mat_data["delta_ddot"]

    # Load the model once outside of the loop
    eval_env = create_env()
    obs, info = eval_env.reset()
    MatrixA = info['MatrixA']
    MatrixB = info['MatrixB']
    MatrixC = info['MatrixC']

    pitch, plunge, delta, CL, delta_ddot_input_py = [], [], [], [], []
    plunge_dot, pitch_dot, delta_dot = [], [], []
    start_time = time.time()


    for delta_ddot_input in delta_ddot_input_values:
        
        obs, reward, done, early_Stop, info = eval_env.step(delta_ddot_input)

        pitch.append(info['pitch'])
        plunge.append(info['plunge'])
        delta.append(info['delta'])
        CL.append(info['CL'])
        delta_ddot_input_py.append(info['delta_ddot_input'])
        plunge_dot.append(info['plunge_dot'])
        pitch_dot.append(info['pitch_dot'])
        delta_dot.append(info['delta_dot'])


    end_time = time.time()
    print(f"iteration per sec: {len(delta_ddot_input_values) / (end_time - start_time)}")

    output_data = {"pitch_py": np.array(pitch), 
                   "plunge_py": np.array(plunge), 
                   "delta_py": np.array(delta), 
                   "CL_py": np.array(CL),
                   "delta_ddot_input_py": np.array(delta_ddot_input_py),
                   "MatrixA": MatrixA,
                   "MatrixB": MatrixB,
                   "MatrixC": MatrixC,
                   "plunge_dot_py": np.array(plunge_dot),
                   "pitch_dot_py": np.array(pitch_dot),
                   "delta_dot_py": np.array(delta_dot)}

    print("saving to output.mat...")
    file_path =f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/results.mat"
    savemat(file_path, output_data)
    print("saved to output.mat")


def create_env():
    # Initialize the environment with its configuration if needed
    env = AerofoilEnv(env_id=0)  # assuming env_id is a required argument
    
    return env


if __name__ == '__main__':
    main()
