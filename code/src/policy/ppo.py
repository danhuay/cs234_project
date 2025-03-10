import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from final_project.code.src.policy.base import CNNFeatureExtractor, MLPPolicy
from stable_baselines3 import PPO


class SB3CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = None,
        transform_fn=None,
    ):
        # Extract height and width from observation_space (assumes (channels, height, width))
        _, input_height, input_width = observation_space.shape

        # Create your CNN extractor
        cnn_extractor = CNNFeatureExtractor(input_height, input_width)
        # If features_dim not provided, use the output size from your CNN extractor
        if features_dim is None:
            features_dim = cnn_extractor.get_output_size()
        else:
            assert (
                cnn_extractor.get_output_size() == features_dim
            ), "The CNN output dimension must be equal to features_dim"

        super().__init__(observation_space, features_dim)
        self.cnn_extractor = cnn_extractor
        # Set the state transformation; if none is provided, use identity
        if transform_fn is None:
            self.transform_fn = lambda x: x
        else:
            self.transform_fn = transform_fn

    def forward(self, observations):
        x = self.transform_fn(observations)
        return self.cnn_extractor(x)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=SB3CustomCNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=None, transform_fn=None),
            net_arch=dict(pi=[512], vf=[512]),
            **kwargs
        )


class PPOPolicy:
    """A wrapper policy class to match API calls from others"""

    def __init__(self, checkpoint_path):
        self.model = PPO.load(checkpoint_path)

    def sample_action(self, state):
        action, _states = self.model.predict(state, deterministic=False)
        return int(action)
