
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import mujoco
import gymnasium as gym
from walker2d_v4 import Walker2dEnv

ENV_ID = 10

def main():

    # Create the vectorized environment
    #env = create_env() #gym.make('Ant-v4')

    model_name = f"walker_constant_2"
    EVAL_MODEL_PATH = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/trained_model/random_design/{model_name}.zip"
    log_dir = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/walker_tensorboard/TB_{model_name}"

    # Initialize the PPO model
    #model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)


    # Train the model
    #total_timesteps = 4096 * 1000  # You can adjust this number based on your needs
    #model.learn(total_timesteps=total_timesteps)
    #model.save(f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/trained_model/random_design/{model_name}")

    #model = PPO.load(EVAL_MODEL_PATH)
    # Evaluate the model
    #eval_env = DummyVecEnv([lambda: gym.make('Walker2d-v4')])
    #mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)

    #print(f"Mean reward: {mean_reward} +/- {std_reward}")


    mujoco_file_folder = f"/Users/adrianbuda/Downloads/master_thesis-aerofoil/src/walker/assets/"

    XML_PATH = f"{mujoco_file_folder}walker2d.xml"

    # Load the model and data
    xml_model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(xml_model)

    batch_size = 10
    episode_length_loop = 1024


    for iter in range(batch_size):
        eval_env = create_env() #gym.make('Walker2d-v4', render_mode='human') #create_env()
        model = PPO.load(EVAL_MODEL_PATH, env=eval_env)

        rewards = 0
        episode_length = 0
        obs = model.env.reset()

        for _ in range(episode_length_loop):
            actions, _states = model.predict(obs, deterministic=True)


            obs, rewards_env, dones, infos = model.env.step(actions)
            rewards += rewards_env
            episode_length += 1
            if dones:
                break

        print("Reward: ", rewards)

def create_env():
    return Walker2dEnv(env_id=ENV_ID,render_mode='human')


if __name__ == '__main__':
    main()
