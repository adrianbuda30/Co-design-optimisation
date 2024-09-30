import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gym import make

def main():
    # Training parameters
    hidden_sizes_train = 128
    learning_rate_train = 0.0003
    n_epochs_train = 10
    n_steps_train = 512 * 5
    n_envs_train = 8
    entropy_coeff_train = 0.0
    total_timesteps_train = n_steps_train * n_envs_train * 2000
    batch_size_train = 512 * 1
    global_iteration = 0
    TRAIN = True

    while True:
        learning_rate_train *= 0.6
        onpolicy_kwargs = dict(activation_fn=torch.nn.Tanh,
                               net_arch=dict(vf=[hidden_sizes_train, hidden_sizes_train],
                                             pi=[hidden_sizes_train, hidden_sizes_train])
                               )
        
        global_iteration += 1 

        # Create vectorized environment
        env_fns = [lambda: make("CartPole-v1") for _ in range(n_envs_train)]
        vec_env = SubprocVecEnv(env_fns, start_method='fork')

        log_dir = f"CartPole_Tanh_Tsteps_{total_timesteps_train}_lr{0.003}_hidden_sizes_{hidden_sizes_train}_lay{2}"
        model_name = f"CartPole_Tanh_Tsteps_{total_timesteps_train}_lr{0.003}_hidden_sizes_{hidden_sizes_train}_lay{2}"

        if TRAIN:
            new_model = PPO("MlpPolicy", env=vec_env, n_steps=n_steps_train, batch_size=batch_size_train, 
                            n_epochs=n_epochs_train, ent_coef=entropy_coeff_train, learning_rate=learning_rate_train,
                            policy_kwargs=onpolicy_kwargs, device='auto', verbose=1, tensorboard_log=log_dir)
            
            if global_iteration != 1:  # We can use this condition as a replacement for LOAD_OLD_MODEL
                trained_model = PPO.load(model_name, env=vec_env)
                new_model.policy.load_state_dict(trained_model.policy.state_dict())
                new_model.policy.value_net.load_state_dict(trained_model.policy.value_net.state_dict())
                print("Saved model loaded")

            # Train the new model   
            print("Model training...")
            new_model.learn(total_timesteps=total_timesteps_train, progress_bar=True)
            print("Model trained, saving...")
            new_model.save(model_name)
            print("Model saved")
            vec_env.close()

if __name__ == '__main__':
    main()
