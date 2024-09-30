from stable_baselines3 import PPO
from env_cartpole import CartPoleEnv
import numpy as np
from scipy.io import savemat
import time
import os

def main():
    # Training parameters
    REWARD = np.array([1.0, 0.2, 0.2])
    global_iteration = 0
    EVAL = True
    model_name = f"matfile"
    EVAL_MODEL_PATH = f"/home/divij/Documents/quadopter/src/cart_pole/rl/trained_model/{model_name}"
    
    # Load the model once outside of the loop
    eval_env = create_env(REWARD)
    eval_model = PPO.load(EVAL_MODEL_PATH, env=eval_env)
    while True:
        global_iteration += 1 
        
        if EVAL:
            evaluate(eval_model, model_name, global_iteration)

        break

def create_env(reward_config):
    return CartPoleEnv(REWARD=reward_config)

def evaluate(model, model_name, global_iteration):
    LOOP_COUNT = 1000000
    cart_position = []
    pole_position = []
    rewards = []
    effort = []
    avg_effort = []
    average_effort_mat, steps, ep_reward_mat, pole_length, pole_position_mat, cart_position_mat = [], [], [], [], [], []
    ep_reward = 0
    step_count = 0
    effort_temp = 0
    start_time = time.time()
    obs = model.env.reset()
    eval_pole_length =  np.array([1.6, 1.6, 1.6, 1.6, 1.8, 1.8, 1.8, 1.8, 2.0, 2.0, 2.0, 2.0, 2.2, 2.2, 2.2, 2.4, 2.4, 2.4])
    k=0
    print("starting evaluation")
    
    for num in range(LOOP_COUNT):
        if num % 10000 == 0:
            end_time = time.time()
            print(f"iteration: {num}, iteration per sec: {10000/(end_time-start_time)}")
            start_time = time.time()

        actions, _states = model.predict(obs)
        obs, reward, done, info = model.env.step(actions)
        
        cart_position.append(np.array(info[0]['cart_pos']))        
        pole_position.append(np.array(info[0]['pole_pos']))
        avg_effort.append(np.array(info[0]['effort']))
        rewards.append(np.array(reward))
        pole_length.append(np.array(info[0]['pole_length']))
        effort_temp = effort_temp + abs(np.array(info[0]['force']))*0.02
        effort.append(np.array(effort_temp))

        ep_reward += reward
        step_count += 1

        if done:
            average_effort_mat.append(np.mean(avg_effort))
            steps.append(np.array(step_count))
            ep_reward_mat.append(np.array(ep_reward))
            avg_effort = []
            ep_reward = 0
            step_count = 0
            effort_temp = 0
            model.env.env_method('set_pole_length', eval_pole_length[k])
            print("steps: ", steps[k])

            k+=1

            # if k % 5 == 0:
            #     eval_pole_length = eval_pole_length + 0.2
        if num % 10000 == 0:
            # Save data in a MATLAB file
            output_data = {
                "cart_position": np.array(cart_position),
                "pole_position": np.array(pole_position),
                "pole_length": np.array(pole_length),
                "average_effort": np.array(average_effort_mat),
                "steps": np.array(steps),
                "ep_reward": np.array(ep_reward_mat),
                "rewards": np.array(rewards),
                "effort": np.array(effort)
            }
            print("saving to output.mat...")

            file_path = f"/home/divij/Documents/quadopter/cartpole_system/EVAL_{model_name}.mat"

            savemat(file_path, output_data)
            print("saved to output.mat")

if __name__ == '__main__':
    main()
