import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from final_project.code.src.policy.base import CNNFeatureExtractor, MLPPolicy
from stable_baselines3 import PPO
from gymnasium.wrappers import LazyFrames
import numpy as np

import torch
import torch.nn as nn


class SB3CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = None,
    ):
        # Create your CNN extractor
        cnn_extractor = CNNFeatureExtractor(state_dim=observation_space.shape)

        # If features_dim not provided, use the output size from your CNN extractor
        if features_dim is None:
            features_dim = cnn_extractor.get_output_size()
        else:
            assert (
                cnn_extractor.get_output_size() == features_dim
            ), "The CNN output dimension must be equal to features_dim"

        super().__init__(observation_space, features_dim)
        self.cnn_extractor = cnn_extractor

    def forward(self, observations):
        return self.cnn_extractor(observations)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        pretrained_weights_path=None,
        **kwargs
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=SB3CustomCNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=None),
            net_arch=dict(pi=[512], vf=[512]),
            activation_fn=nn.ReLU,
            **kwargs
        )

    def obs_to_tensor(self, observation):
        if isinstance(observation, LazyFrames):
            observation = np.array(observation)
        observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]
        return super().obs_to_tensor(observation)


class PPOPolicy:
    """A wrapper policy class to match API calls from others"""

    def __init__(self, checkpoint_path):
        self.model = PPO.load(checkpoint_path)

    def sample_action(self, state, deterministic=False):
        if type(state) != torch.Tensor:
            state = state.__array__()

        action, _states = self.model.predict(state, deterministic=deterministic)
        return int(action)
