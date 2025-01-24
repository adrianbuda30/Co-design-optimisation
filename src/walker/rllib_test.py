from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig()
config.environment("CartPole-v1")
config.env_runners(num_env_runners=1)
config.training(
    gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256
)

# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build()
algo.train()