from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy, register_policy
import torch as th
import torch.nn as nn
import gym

class CustomCnnPolicy(BasePolicy):
    """
    Policy class that implements a custom CNN policy, using similar architecture as DQN NatureCNN.
    """

    def __init__(self, *args, **kwargs):
        super(CustomCnnPolicy, self).__init__(*args, **kwargs,
                                              features_extractor_class=CustomCNN,
                                              features_extractor_kwargs=dict(features_dim=512))

class CustomCNN(nn.Module):
    """
    Custom CNN as feature extractor.
    You might want to change it to fit your needs (number of channels/features).
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__()
        # Compute number of input features for the fully connected net
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 256),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Register the policy, it will check that the name is not already taken
# register_policy('CustomCnnPolicy', CustomCnnPolicy)
