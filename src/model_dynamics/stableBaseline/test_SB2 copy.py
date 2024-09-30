import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

from gym import wrappers
import numpy as np
from gym.envs.registration import register

register(
    id='Quadcopter-v0',
    entry_point='src.quadcopter_env:QuadcopterEnv',
)

# env = gym.make('Quadcopter-v0')
# ob = env.reset()

env = gym.make('Quadcopter-v0')
ob = env.reset()

# multiprocess environment
env = make_vec_env('Quadcopter-v0', n_envs=50)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000000)
model.save("Quadcopter")

# del model # remove to demonstrate saving and loading

model = PPO2.load("Quadcopter")

# Enjoy trained agent
ob = env.reset()
while True:
    action, _states = model.predict(ob)
    ob, rewards, dones, info = env.step(action)
    print(f"ac: {action[0][0]:<25} ac: {action[0][1]:<25} ac: {action[0][2]:<25} ac: {action[0][3]:<25} ob: {ob[0][0]:<25} {ob[0][1]:<25} {ob[0][2]:<25}")    
    if dones[0]:
        break
    # env.render()