import gymnasium as gym
import numpy as np
import torch
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.wrappers import LazyFrames
from stable_baselines3 import PPO
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from final_project.code.src.policy.base import CNNFeatureExtractor


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
        **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=SB3CustomCNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=None),
            net_arch=dict(pi=[512], vf=[512]),
            activation_fn=nn.ReLU,
            **kwargs,
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


class CustomPPO(PPO):
    def __init__(self, *args, ref_model=None, kl_coeff=0.01, **kwargs):
        """
        Args:
            ref_model: CNN-based reference model (PyTorch neural network)
            kl_coeff: Coefficient for KL divergence penalty
            discrete: Whether the action space is discrete (default=True)
        """
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model  # CNN Model for reference policy
        self.kl_coeff = kl_coeff

    def compute_kl_divergence(self, rollout_data):
        """
        Compute KL divergence between the current policy distribution and CNN reference policy.
        """
        obs = rollout_data.observations

        with torch.no_grad():
            ref_logits = self.ref_model(obs)

        ref_dist = torch.distributions.Categorical(logits=ref_logits)
        policy_dist = self.policy.get_distribution(obs).distribution

        kl_div = torch.distributions.kl.kl_divergence(policy_dist, ref_dist)
        return kl_div.mean()  # Mean over batch

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # =================== CUSTOM PART! KL penalty ===================
                if self.ref_model is not None:
                    kl_loss = self.compute_kl_divergence(rollout_data)
                    loss = loss + self.kl_coeff * kl_loss
                # =================== CUSTOM PART! KL penalty ===================

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if self.ref_model is not None:
            self.logger.record("train/bc_kl_loss", kl_loss.item())
