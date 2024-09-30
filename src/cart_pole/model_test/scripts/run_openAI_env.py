from env_cartpole import CartPoleEnv
import numpy as np
from scipy.io import savemat
import time
import gym

def main():
    # Load the model once outside of the loop
    REWARD = np.array([1.0, 0.2])

    eval_env_openAI = gym.make('CartPole-v1')
    eval_env_simulink = create_env(REWARD)

    evaluate(eval_env_openAI, eval_env_simulink)

def create_env(reward_config):
    return CartPoleEnv(REWARD=reward_config)

def evaluate(eval_env_openAI, eval_env_simulink):
    LOOP_COUNT = 10000
    cart_position_openAI = []
    cart_vel_openAI = []
    pole_position_openAI = []
    pole_vel_openAI = []
    pr = False
    cart_position = []
    pole_position = []
    cart_position_simulink = []
    cart_vel_simulink = []
    pole_position_simulink = []
    pole_vel_simulink = []    
    actions_openAI = 0
    start_time = time.time()
    obs_openAI = eval_env_openAI.reset()
    obs_simulink = eval_env_simulink.reset()
    print("starting evaluation")
    
    for num in range(LOOP_COUNT):
        if num % 10000 == 0:
            end_time = time.time()
            print(f"iteration: {num}, iteration per sec: {10000/(end_time-start_time)}")
            start_time = time.time()

        # Randomly select an action (0 or 1) just as an example. Replace this with your policy if needed.
        if num % 10 == 0:
            if actions_openAI == 0:
                actions_openAI = 1
            elif(actions_openAI == 1):
                actions_openAI = 0

        if actions_openAI == 0:
            actions_simulink = 0.0
        elif(actions_openAI == 1):
            actions_simulink = 0.0
        else:
            print("invalid action")
            exit(1)

        if pr:
            print("simulink reset: ", obs_simulink)
            print("open AI reset: ", obs_openAI)
            actions_openAI = 0

        obs_openAI, reward, terminated, _, info = eval_env_openAI.step(actions_openAI)
        obs_simulink, _, _, _, _ = eval_env_simulink.step(actions_simulink)
        eval_env_openAI.render()  # <-- Add this line to render the OpenAI environment

        if pr:
            pr = False

        # if terminated:
        #     obs_openAI = eval_env_openAI.reset()
        #     obs_simulink,_ = eval_env_simulink.reset()
        #     pr = True


        cart_position_openAI.append(np.array(obs_openAI[0]))
        cart_vel_openAI.append(np.array(obs_openAI[1]))
        pole_position_openAI.append(np.array(obs_openAI[2]))
        pole_vel_openAI.append(np.array(obs_openAI[3]))


        cart_position_simulink.append(np.array(obs_simulink[0]))
        cart_position.append(np.array(obs_simulink[0]))
        cart_vel_simulink.append(np.array(obs_simulink[1]))
        pole_position_simulink.append(np.array(obs_simulink[2]))
        pole_vel_simulink.append(np.array(obs_simulink[3]))
        pole_position.append(np.array([obs_simulink[4], obs_simulink[5]]))

    # Save data in a MATLAB file
    output_data = {
        "cart_position_openAI": np.array(cart_position_openAI),
        "cart_vel_openAI": np.array(cart_vel_openAI),
        "pole_position_openAI": np.array(pole_position_openAI),
        "pole_vel_openAI": np.array(pole_vel_openAI),
        "cart_position_simulink": np.array(cart_position_simulink),
        "cart_vel_simulink": np.array(cart_vel_simulink),
        "pole_position_simulink": np.array(pole_position_simulink),
        "pole_vel_simulink": np.array(pole_vel_simulink),
        "pole_position": np.array(pole_position),
        "cart_position": np.array(cart_position)

    }
    print("saving to output.mat...")
    file_path = f"/home/divij/Documents/quadopter/cartpole_system/results.mat"
    savemat(file_path, output_data)
    print("saved to output.mat")

if __name__ == '__main__':
    main()
