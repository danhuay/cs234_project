import gzip
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from final_project.code.src.actions import meaningful_actions

logger = logging.getLogger(__name__)


def get_x_pos(info):
    """
    From env returned info calculate x position of Mario.
    Parameters
    ----------
    info: dict of info returned by env.step()

    Returns x_pos: int
    -------

    """
    xscroll_hi = info.get("xscrollHi", 0)
    xscroll_lo = info.get("xscrollLo", 0)
    x_pos = (xscroll_hi << 8) + xscroll_lo
    return x_pos


def convert_lazyframe_to_tensor(lazyframe):
    return torch.tensor(lazyframe.__array__(), dtype=torch.float32)


def action_to_string(action_arr):
    return "".join([str(x) for x in action_arr])


def load_trajectories(traj_folder, skip_frame=4):
    """
    Load human trajectories from a folder. Union
    all trajectories into a single list.

    Args:
        traj_folder: folder containing pickle file of trajectories

    Returns:
        trajectories: a list of trajectories, where each trajectory is a list of
            observations

    """
    states = list()
    actions = list()
    info = list()
    included_actions = [action_to_string(x) for x in meaningful_actions]

    for file in os.listdir(traj_folder):
        if file.endswith(".pkl"):
            traj_path = os.path.join(traj_folder, file)

            with open(traj_path, "rb") as f:
                _t = pickle.load(f)
                for n, i in enumerate(_t):
                    if n % skip_frame == 0:
                        if action_to_string(i["action"]) in included_actions:
                            states.append(convert_lazyframe_to_tensor(i["observation"]))
                            actions.append(i["action"])
                            info.append(i["info"])

    assert len(states) == len(actions) == len(info)

    return states, actions, info


def pretrained_weights_breakdown_for_ppo(pretrained_weights_path):
    checkpoint = torch.load(pretrained_weights_path)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    # get the feature extractor part
    feature_extractor_state_dict = {
        k.replace("features.", "cnn_extractor.", 1): v  # Remove 'features.' prefix
        for k, v in checkpoint.items()
        if k.startswith("features.")
    }

    # get the mlp part
    w = {
        k.replace("classifier.", ""): v
        for k, v in checkpoint.items()
        if k.startswith("classifier.")
    }

    # get the mlp_extractor part
    mlp_extractor_state_dict = {k: v for k, v in w.items() if k.startswith("0.")}

    action_net_state_dict = {
        k.replace("2.", ""): v for k, v in w.items() if k.startswith("2.")
    }

    return feature_extractor_state_dict, mlp_extractor_state_dict, action_net_state_dict


def load_retrained_weights_to_ppo(
    model, pretrained_weights_path, load_feat=False, load_mlp=False
):
    """model is a PPO policy from baseline3.
    pretrained_weights_path is the path to the pretrained weights
    from CNNPolicy
    """
    (feature_extractor_state_dict, mlp_extractor_state_dict, action_net_state_dict) = (
        pretrained_weights_breakdown_for_ppo(pretrained_weights_path)
    )

    if load_feat:
        model.policy.features_extractor.load_state_dict(feature_extractor_state_dict)
    if load_mlp:
        model.policy.mlp_extractor.policy_net.load_state_dict(mlp_extractor_state_dict)
        model.policy.mlp_extractor.value_net.load_state_dict(mlp_extractor_state_dict)
        model.policy.action_net.load_state_dict(action_net_state_dict)
    return model


class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=5, verbose=1):
        super(EvaluationCallback, self).__init__(verbose)
        self.eval_env = eval_env  # Separate environment for evaluation
        self.eval_freq = eval_freq  # Frequency of evaluation
        self.n_eval_episodes = n_eval_episodes  # Number of episodes per evaluation

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:  # Run evaluation every eval_freq steps
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
                deterministic=False,
            )

            # Log cumulative reward (per trajectory) in TensorBoard
            self.logger.record("train/mean_episode_reward", np.mean(mean_reward))
            self.logger.record("train/std_episode_reward", np.mean(std_reward))

            if self.verbose:
                logger.info(
                    f"Step {self.n_calls}: Mean Episode Reward = {np.mean(mean_reward):.2f} ± {np.mean(std_reward):.2f}"
                )

        return True  # Continue training


class DummySummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self):
        pass


class ExpertTraj:
    def __init__(self, folder):
        self.folder = folder
        self.state_files = self.get_list_of_files()

    def get_list_of_files(self):
        return sorted([p for p in Path(self.folder).iterdir() if p.suffix == ".state"])

    def get_file_name(self, idx):
        return self.state_files[idx]

    def __len__(self):
        return len(self.state_files)

    def __getitem__(self, idx):
        return self.load_emulator_state(self.get_file_name(idx))

    @staticmethod
    def load_emulator_state(file):
        with gzip.open(file, "rb") as f:
            _em_state = f.read()
        return _em_state
