import gzip
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from torch.backends.cudnn import deterministic
from torch.utils.tensorboard import SummaryWriter
from final_project.code.src.utils import DummySummaryWriter, ExpertTraj

from final_project.code.src.wrapper import create_game_env, ExpertTrajResetEnv
import final_project.code.src.actions as actions
import final_project.code.src.policy.dqn as dqn
from final_project.code.src.policy.base import CNNPolicy
from final_project.code.src.policy.dataset import HumanTrajectoriesDataLoader
from final_project.code.src.policy.ppo import (
    CustomActorCriticPolicy,
    PPOPolicy,
    CustomPPO,
)
from final_project.code.src.policy.trainer import ModelTrainer
from final_project.code.src.utils import (
    get_x_pos,
    load_retrained_weights_to_ppo,
    EvaluationCallback,
)

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_game(
    env,
    num_games=10,
    action_policy=None,
    epsilon=0,
    experiment_name=None,
    deterministic=True,
    *args,
    **kwargs,
):
    writer = SummaryWriter(experiment_name) if experiment_name else DummySummaryWriter()
    n_game_end_rewards = list()
    n_game_end_positions = list()
    finishline_positions = 13.0 * 256.0  # 13 screens * 256 pixels per screen

    for n in range(num_games):
        logger.info(f"Starting new game {n + 1} of {num_games}")
        cum_rewards = 0

        # move the first action always to the right (action 1)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(1)

        step = 0
        while not terminated or truncated:
            if action_policy is None:
                action = env.action_space.sample()
            else:
                # if epsilon-greedy, sample random action with probability epsilon
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = action_policy.sample_action(observation, deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            # writer.add_scalar("Game Cumulative Reward", cum_rewards, step)
            step += 1
            cum_rewards += reward
            env.render()

            writer.add_scalar(f"Game Cumulative Reward/{n}", cum_rewards, step)
            writer.add_scalar(f"Game X Position/{n}", get_x_pos(info), step)
            writer.add_scalar(f"Current Reward/{n}", reward, step)

        # end-game statistics
        n_game_end_rewards.append(cum_rewards)
        n_game_end_positions.append(get_x_pos(info) / finishline_positions)

    # log the end files to csv
    df = pd.DataFrame(
        {"cum_rewards": n_game_end_rewards, "completion": n_game_end_positions}
    )
    if experiment_name:
        df.to_csv(experiment_name + "_log.csv", index=False)

    writer.close()
    return


def load_trainer_policy(checkpoint_path):
    policy = ModelTrainer(
        model=CNNPolicy(state_dim=(4, 84, 84), action_dim=len(actions.SIMPLE_MOVEMENT)),
        train_dataloader=None,
        dev_dataloader=None,
        optimizer=None,
        criterion=None,
    )
    policy.load_checkpoint("checkpoints/bc_policy.pt")
    return policy


def run_policy(checkpoint_path, ppo=False, *args, **kwargs):
    env = create_game_env(state="Level1-1")
    if ppo:
        policy = PPOPolicy(checkpoint_path)
    else:
        policy = load_trainer_policy(checkpoint_path)

    # run game using the trained policy
    run_game(env, action_policy=policy, *args, **kwargs)
    env.close()
    return


def train_bc_policy(
    human_traj_folder="human_demon", num_epochs=500, eval_interval=5, *args, **kwargs
):
    train_dl, dev_dl = HumanTrajectoriesDataLoader(
        human_traj_folder, split=True, train_fraction=0.7, batch_size=32, shuffle=True
    )

    model = CNNPolicy(state_dim=(4, 84, 84), action_dim=len(actions.SIMPLE_MOVEMENT))
    trainer = ModelTrainer(
        model=model,
        train_dataloader=train_dl,
        dev_dataloader=dev_dl,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        criterion=nn.CrossEntropyLoss(),
        *args,
        **kwargs,
    )

    trainer.train(num_epochs=num_epochs, eval_interval=eval_interval)
    return


def train_dqn_policy(*args, **kwargs):
    agent = dqn.DQNAgent(
        state_dim=(4, 84, 84),
        action_dim=len(actions.SIMPLE_MOVEMENT),
        buffer_capacity=2048,
        batch_size=32,
        gamma=0.99,
        lr=1e-4,
    )

    env = create_game_env(state="Level1-1")
    dqn.train(agent, env, num_episodes=250, *args, **kwargs)
    env.close()


def train_ppo_policy(
    training_steps=100000,  # Total environment steps
    checkpoint_freq=500,
    eval_freq=500,
    model_name="ppo",
    n_envs=2,  # Number of parallel environments
    warm_start=False,
    pretrained_weights_path=None,
    env_args=None,
    ref_model=None,
    kl_coeff=0.1,
):
    # Create multiple environments using SubprocVecEnv for parallel processing
    env_args = {} if env_args is None else env_args
    env = SubprocVecEnv(
        [lambda: create_game_env(state="Level1-1", **env_args) for _ in range(n_envs)]
    )
    eval_env = Monitor(create_game_env(state="Level1-1"))

    # Ensure checkpoint directory exists
    checkpoint_dir = f"checkpoints/{model_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up checkpoint callback (saves model every checkpoint_freq steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    # Reward logging callback
    reward_logging_callback = EvaluationCallback(eval_env, eval_freq=eval_freq)
    # Initialize PPO with multiple environments
    if warm_start:
        model = PPO.load(f"{checkpoint_dir}/final_model")
        model.set_env(env)
    else:

        def _clip_schedule(progress_remaining):
            return 0.05 + (0.1 * (1 - progress_remaining))

        model = CustomPPO(
            CustomActorCriticPolicy,
            env,
            n_steps=512,  # Number of steps per rollout (per environment)
            batch_size=32,
            n_epochs=10,  # Number of optimization epochs per rollout batch
            learning_rate=3e-5,
            clip_range=0.02,
            ent_coef=0.01,
            tensorboard_log=f"runs/{model_name}",
            verbose=1,
            ref_model=ref_model,
            kl_coeff=kl_coeff,
        )

    if pretrained_weights_path:
        # loading BC feature extractor to PPO model
        load_retrained_weights_to_ppo(model, pretrained_weights_path)

    # Train the model with both evaluation and checkpoint callbacks
    model.learn(
        total_timesteps=training_steps,
        callback=[checkpoint_callback, reward_logging_callback],
    )

    # Save the final model
    model.save(f"{checkpoint_dir}/final_model")

    env.close()
    eval_env.close()


def main():
    env = create_game_env(state="Level1-1")

    # only use one-life setting
    run_game(env, num_games=10)


if __name__ == "__main__":
    set_seed(12345)
    # main()

    # ================= DQN =================
    # train_dqn_policy(
    #     log_dir="runs/dqn_base",
    #     model_save_path="checkpoints/dqn_base_policy.pt",
    # )

    # ================= BEHAVIORAL CLONING =================
    # train_bc_policy(
    #     human_traj_folder="human_demon",
    #     checkpoint_name=f"bc_policy.pt",
    #     num_epochs=500,
    #     eval_interval=5,
    #     early_stopping_patience=10,
    #     log_dir="runs",
    #     experiment_name="bc_policy",
    # )

    # ================= PPO =================
    ref_model = load_trainer_policy("checkpoints/bc_policy.pt").model

    train_ppo_policy(
        training_steps=512 * 200,
        checkpoint_freq=512 * 20,
        eval_freq=512 * 2,
        model_name="hrl_ppo_policy_kl_0.1_ws_mlp",
        n_envs=2,
        warm_start=False,
        ref_model=ref_model,
        kl_coeff=0.1,
        # env_args={
        #     "expert_traj": ExpertTraj("human_demon_w_states"),
        #     "total_reset_steps": 50,  # number of episodes reset
        #     "decay_factor": 0.9,
        # },
        pretrained_weights_path="checkpoints/bc_policy.pt",
    )

    train_ppo_policy(
        training_steps=512 * 200,
        checkpoint_freq=512 * 20,
        eval_freq=512 * 2,
        model_name="hrl_ppo_policy_kl_0.1_ws_mlp_er",
        n_envs=2,
        warm_start=False,
        ref_model=ref_model,
        kl_coeff=0.1,
        env_args={
            "expert_traj": ExpertTraj("human_demon_w_states"),
            "total_reset_steps": 50,  # number of episodes reset
            "decay_factor": 0.9,
        },
        pretrained_weights_path="checkpoints/bc_policy.pt",
    )

    train_ppo_policy(
        training_steps=512 * 200,
        checkpoint_freq=512 * 20,
        eval_freq=512 * 2,
        model_name="hrl_ppo_policy_kl_0.1_er",
        n_envs=2,
        warm_start=False,
        ref_model=ref_model,
        kl_coeff=0.1,
        env_args={
            "expert_traj": ExpertTraj("human_demon_w_states"),
            "total_reset_steps": 50,  # number of episodes reset
            "decay_factor": 0.9,
        },
        # pretrained_weights_path="checkpoints/bc_policy.pt",
    )

    #
    # train_ppo_policy(
    #     training_steps=512 * 2000,
    #     checkpoint_freq=512 * 100,
    #     eval_freq=512 * 20,
    #     model_name="ppo_base_policy_2k_rollouts",
    #     n_envs=2,
    #     warm_start=False,
    #     # pretrained_weights_path="checkpoints/bc_policy.pt",
    # )

    # ================= RUN POLICY =================
    for model in [
        # "bc_policy",
        # "dqn_base_policy",
        # "ppo_base_policy",
        # "hrl_bc_ppo_policy_ws_mlp_dyn_clip",
        # "hrl_bc_ppo_policy_ws_feat_dyn_clip",
        # "hrl_bc_ppo_policy_ws_all_dyn_clip",
        # "hrl_exp_traj_reverse_ppo",
        # "hrl_exp_traj_reverse_ppo_ws_mlp_dyn_clip",
        # "hrl_ppo_policy_kl_0.1",
        "hrl_ppo_policy_kl_0.1_ws_mlp",
        "hrl_ppo_policy_kl_0.1_ws_mlp_er",
        "hrl_ppo_policy_kl_0.1_er",
    ]:
        if "ppo" in model:
            run_policy(
                f"checkpoints/{model}/final_model",
                num_games=50,
                # epsilon=0,
                experiment_name=f"runs/policy_runs/run_{model}_stochastic",
                ppo=True,
                deterministic=False,
            )
        else:
            run_policy(
                f"checkpoints/{model}.pt",
                num_games=50,
                # epsilon=0,
                experiment_name=f"runs/policy_runs/run_{model}_stochastic",
                ppo=False,
                deterministic=False,
            )
