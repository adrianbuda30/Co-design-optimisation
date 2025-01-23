
import numpy as np
from stable_baselines3 import PPO
from walker2d_v4_test import Walker2dEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from scipy.io import loadmat, savemat

import torch




def main():
    #training parameters
    use_sde = False
    hidden_sizes_train = 256
    REWARD = np.array([1.0, 0.0])
    learning_rate_train = 0.0001
    n_epochs_train = 10
    LOAD_OLD_MODEL = False
    n_steps_train = 512 * 2
    n_envs_train = 256
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 10000
    batch_size_train = 64
    global_iteration = 0
    TRAIN = True
    CALL_BACK_FUNC = f"constant_design"


    while True:

        learning_rate_train = learning_rate_train

        onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                               net_arch=dict(vf=[hidden_sizes_train, hidden_sizes_train],
                                             pi=[hidden_sizes_train, hidden_sizes_train]))

        global_iteration += 1


        env_configs = [{'env_id': i, 'ctrl_cost_weight': 0.5} for i in range(n_envs_train)]

        assert len(env_configs) == n_envs_train


        env_fns = [lambda config=config: Walker2dEnv(**config) for config in env_configs]

        vec_env = SubprocVecEnv(env_fns, start_method='fork')

        n_envs_eval = 1
        env_configs_eval = [{'env_id': i, 'ctrl_cost_weight': 0.5, 'render_mode': 'human'} for i in range(n_envs_eval)]

        assert len(env_configs_eval) == n_envs_eval

        env_fns_eval = [lambda config=config: Walker2dEnv(**config) for config in env_configs_eval]

        vec_env_eval = DummyVecEnv(env_fns_eval)


        model_name = f"walker_constant_test_gpu"
        log_dir = f"/home/ab2419/Co-design-optimisation/src/walker/walker_tensorboard/TB_{model_name}"

        if LOAD_OLD_MODEL is True:
            new_model = []
            old_model = PPO.load(f"/home/ab2419/Co-design-optimisation/src/walker/trained_model/constant_design/walker_constant_sprint_test.zip", env = vec_env)

            new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train,
                            batch_size=batch_size_train, n_epochs=n_epochs_train,
                            use_sde=use_sde, ent_coef=entropy_coeff_train,
                            learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs,
                            device='cuda', verbose=1, tensorboard_log=log_dir)

            new_model_eval = PPO("MlpPolicy", env=vec_env_eval, n_steps=n_steps_train,
                            batch_size=batch_size_train, n_epochs=n_epochs_train,
                            use_sde=use_sde, ent_coef=entropy_coeff_train,
                            learning_rate=learning_rate_train, policy_kwargs=onpolicy_kwargs,
                            device='cuda', verbose=1, tensorboard_log=log_dir)


            new_model.set_parameters(old_model.get_parameters())
            new_model_eval.set_parameters(old_model.get_parameters())

        else:
            new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train, batch_size=batch_size_train,
                n_epochs=n_epochs_train, use_sde=use_sde, ent_coef=entropy_coeff_train,
                learning_rate=learning_rate_train,
                policy_kwargs=onpolicy_kwargs, device='cuda', verbose=1, tensorboard_log=log_dir)
            print("New model created")

        print("Model training...")
        if CALL_BACK_FUNC is f"constant_design":
            param_changer = constant_design(model_name = model_name, model = new_model, n_steps_train = n_steps_train, n_envs_train = n_envs_train, verbose=1)
        else:
            print("No callback function specified")
            break


        if TRAIN is True:
            new_model.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=param_changer)
            print("Model trained, saving...")
            new_model.save(f"/home/ab2419/Co-design-optimisation/src/walker/trained_model/random_design/{model_name}")
            print("Model saved")
            LOAD_OLD_MODEL = True
            vec_env.close()
        else:
            new_model_eval.learn(total_timesteps = total_timesteps_train ,progress_bar=True, callback=param_changer)
            print("Model trained, saving...")
            LOAD_OLD_MODEL = True
            vec_env_eval.close()

        break


class constant_design(BaseCallback):
    def __init__(self, model_name=f"matfile", model=None, n_steps_train=512 * 10, n_envs_train=8, verbose=0):

        super(constant_design, self).__init__(verbose)
        self.model = model
        self.n_envs_train = n_envs_train
        self.n_steps_train = n_steps_train
        self.episode_rewards = {}
        self.rewards_iteration = {}
        self.design_iteration = [1 for _ in range(self.n_envs_train)]
        self.design_rewards = [0 for _ in range(self.n_envs_train)]
        self.episode_length = {}
        self.mat_limb_length = []
        self.mat_reward = []
        self.mat_iteration = []
        self.average_reward = []
        self.average_episode_length = []
        self.model_name = model_name
        self.mat_file_name = model_name
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

    def _on_rollout_start(self) -> bool:

        # reset the environments
        for i in range(self.n_envs_train):
            self.training_env.env_method('__init__', i, indices=[i])
            self.training_env.env_method('reset', indices=[i])
        return True

    def _on_step(self) -> bool:

        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            for i, reward in enumerate(rewards):
                self.episode_rewards[i] = self.episode_rewards.get(i, 0) + reward
                self.episode_length[i] = self.episode_length.get(i, 0) + 1

        if 'dones' in self.locals:
            dones = self.locals['dones']
            for i, done in enumerate(dones):
                if done or self.episode_length[i] >= self.n_steps_train:
                    self.average_episode_length.append(self.episode_length[i])
                    self.average_reward.append(self.episode_rewards[i])
                    self.design_iteration[i] += 1

                    self.episode_rewards[i] = 0
                    self.episode_length[i] = 0
        return True

    def _on_rollout_end(self) -> bool:

        self.model.save(
            f"/home/ab2419/Co-design-optimisation/src/walker/trained_model/constant_design/{self.model_name}")


        for i in range(self.n_envs_train):
            self.mat_reward.append(self.episode_rewards[i])
            self.mat_iteration.append(self.episode_length[i])

        self.logger.record("mean episode length", np.sum(self.average_episode_length) / np.sum(self.design_iteration))
        self.logger.record("mean reward", np.sum(self.average_reward) / np.sum(self.design_iteration))

        output_data = {
            "reward": np.array(self.mat_reward),
            "iteration": np.array(self.mat_iteration),
        }

        print("saving matlab data...")
        file_path = f"/home/ab2419/Co-design-optimisation/src/walker/rl/trained_model/{self.mat_file_name}.mat"
        savemat(file_path, output_data)
        self.average_episode_length = []
        self.average_reward = []
        self.design_iteration = [0 for _ in range(self.n_envs_train)]

        return True



if __name__ == '__main__':
    main()