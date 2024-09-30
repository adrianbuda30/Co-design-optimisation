import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env_cartpole import CartPoleEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np
from gaussMix_design_opt import DesignDistribution_pytorch as DesignDistribution
import time
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from scipy.io import loadmat, savemat

def main():
    #training parameters
    hidden_sizes_train = 128
    REWARD = np.array([1.0, 0.0])
    LOAD_OLD_MODEL = False
    n_envs = 8
    global_iteration = 0
    OPT = True
    while True:

        global_iteration += 1 

        # Define unique initialization variables for each environment
        env_configs = [{'REWARD': REWARD} for _ in range(n_envs)]

        # Ensure we have configurations for each environment instance
        assert len(env_configs) == n_envs 

        # Create function for each environment instance with its unique configuration
        env_fns = [lambda config=config: CartPoleEnv(**config) for config in env_configs]

        # Create the vectorized environment using SubprocVecEnv directly
        eval_env = SubprocVecEnv(env_fns, start_method='fork')
        model_name = f"CartPoleEnv_hidden_sizes_{hidden_sizes_train}_reward_{1000}_obsLen_min{1010}_polemass"
        eval_model = PPO.load(EVAL_MODEL_PATH, env=eval_env)

        if OPT:
            print("starting optimisation")
            EVAL_MODEL_PATH = model_name
            LOOP_COUNT = 500000
            cart_position, pole_position, rewards, pole_length, effort, avg_effort, average_effort_mat, steps, hebo_effort, hebo_length  = [], [], [], [], [], [], [], [], [], []
            step = 0
            obs = eval_env.reset()
            space = DesignSpace().parse([
                {'name': 'x1', 'type': 'num', 'lb': 0.5, 'ub': 10}
            ])
            opt = HEBO(space)
            rec = opt.suggest(n_suggestions=n_envs)
            for i in range(n_envs):
                eval_env.env_method('set_pole_length', rec.values[i][0])
            for _ in range(LOOP_COUNT):
                actions, _states = eval_model.predict(obs)
                obs, rewards_batch, dones, infos = eval_env.step(actions)
                for i in range(n_envs):
                    cart_position.append(np.array(infos[i]['cart_pos']))        
                    pole_position.append(np.array(infos[i]['pole_pos']))
                    rewards.append(np.array(rewards_batch[i]))
                    effort.append(np.array(infos[i]['effort']))
                    avg_effort.append(np.array(infos[i]['effort']))
                    step += 1.0
                    if dones[i]:
                        average_effort_mat.append(np.mean(avg_effort))
                        hebo_effort.append(np.mean(avg_effort)) 
                        hebo_length.append(np.array(infos[i]['pole_length']))
                        steps.append(np.array(step))
                        print("suggested pole length", rec.values[0][0], "effort", np.mean(avg_effort))
                        eval_env[i].reset()
                        opt.observe(rec, np.array([np.mean(avg_effort)]))
                        rec = opt.suggest(n_suggestions=1)
                        new_pole_length = rec.values[0][0]
                        eval_env[i].env_method('set_pole_length', new_pole_length)
                        step = 0
                        pole_length.append(np.array(infos[i]['pole_length']))
                        env_pole_length = eval_env[i].env_method('get_pole_length')[0]
                        if not(new_pole_length == env_pole_length):
                            print("pole lengths are different", new_pole_length, env_pole_length)
                        avg_effort = []


            #save data in a matlab file
            output_data = {
                "cart_position": np.array(cart_position),
                "pole_position": np.array(pole_position),
                "rewards": np.array(rewards),
                "pole_length": np.array(pole_length),
                "effort": np.array(effort),
                "average_effort": np.array(average_effort_mat),
                "steps": np.array(steps),
                "hebo_effort": np.array(hebo_effort),
                "hebo_length": np.array(hebo_length)

            }
            print("saving to output.mat...")
            file_path = f"/home/divij/Documents/quadopter/cartpole_system/{model_name}opt.mat"
            savemat(file_path, output_data)

if __name__ == '__main__':
    main()
