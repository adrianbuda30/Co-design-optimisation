from stable_baselines3 import PPO
from env_LQR import LQREnv
from env_MPC_HEBO import MPCEnv
from env_aerofoil import AerofoilEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from scipy.io import savemat
import time


def main():
    # Training parameters
    REWARD = np.array([0.95, 0.05])
    n_envs_train = 1
    global_iteration = 0
    EVAL = True
    model_name = f"Trial185_random_design.zip"
    EVAL_MODEL_PATH = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/rl/trained_model/HEBO/{model_name}"

    # Load the model once outside of the loop
    eval_env = create_env(n_envs_train, REWARD)
    eval_model = PPO.load(EVAL_MODEL_PATH, env=eval_env)

    while True:
        global_iteration += 1


        if EVAL:
            evaluate(eval_model, n_envs_train, model_name, global_iteration)

        if global_iteration == 10:
            EVAL = False


def create_env(n_envs, reward_config):
    # Define unique initialization variables for each environment
    env_configs = [{'REWARD': reward_config} for _ in range(n_envs)]

    # Create function for each environment instance with its unique configuration
    env_fns = [lambda config=config: AerofoilEnv(**config) for config in env_configs]

    # Create the vectorized environment using SubprocVecEnv
    return SubprocVecEnv(env_fns, start_method='fork')


def evaluate(model, n_envs, model_name, global_iteration):
    LOOP_COUNT = 10001
    pitch = [[] for _ in range(n_envs)]
    plunge = [[] for _ in range(n_envs)]
    delta = [[] for _ in range(n_envs)]
    delta_ddot_input = [[] for _ in range(n_envs)]
    CL = [[] for _ in range(n_envs)]

    rewards = [[] for _ in range(n_envs)]

    step = [[] for _ in range(n_envs)]
    steps, ep_reward_mat, pitch_mat, plunge_mat, delta_mat, delta_ddot_input_mat, CL_mat = [], [], [], [], [], [], []
    ep_reward = np.zeros(n_envs)
    step = np.zeros(n_envs)
    obs = model.env.reset()
    start_time = time.time()
    print("starting evaluation")

    for num in range(1, LOOP_COUNT):
        if num % 10000 == 0:
            end_time = time.time()
            print(f"iteration: {num}, iteration per sec: {10000 / (end_time - start_time)}")
            start_time = time.time()

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards_batch, dones, info = model.env.step(action)

        for i in range(n_envs):

            pitch[i].append(np.array(info[i]['pitch']))
            plunge[i].append(np.array(info[i]['plunge']))
            delta[i].append(np.array(info[i]['delta']))
            delta_ddot_input[i].append(np.array(info[i]['delta_ddot_input']))
            CL[i].append(np.array(info[i]['CL']))
            rewards[i].append(np.array(rewards_batch[i]))
            ep_reward[i] += rewards_batch[i]
            step[i] += 1


            if num % 10000 == 0:
                pitch_mat.append(pitch[i])
                plunge_mat.append(plunge[i])
                delta_mat.append(delta[i])
                delta_ddot_input_mat.append(delta_ddot_input[i])
                CL_mat.append(CL[i])
                steps.append(np.array(step[i]))
                ep_reward_mat.append(np.array(ep_reward[i]))
                ep_reward[i] = 0
                step[i] = 0
            #print(pitch_mat)

        if num % 10000 == 0:
            # Save data in a MATLAB file
            output_data = {
                "pitch": np.array(pitch_mat),
                "plunge": np.array(plunge_mat),
                "delta": np.array(delta_mat),
                "delta_ddot_input": np.array(delta_ddot_input_mat),
                "CL": np.array(CL_mat),
                "steps": np.array(steps),
                "ep_reward": np.array(ep_reward_mat)
            }
            print("saving to output.mat...")
            file_path = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/aerofoil/EVAL_{model_name}_{global_iteration}_no_control_1.mat"
            savemat(file_path, output_data)
            print("saved to output.mat")

if __name__ == '__main__':
    main()