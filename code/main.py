import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import retro
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.backends.cudnn import deterministic
from torch.utils.tensorboard import SummaryWriter

import final_project.code.src.actions as actions
import final_project.code.src.policy.dqn as dqn
import final_project.code.src.wrapper as wrapper
from final_project.code.src.policy.base import CNNPolicy
from final_project.code.src.policy.dataset import (
    HumanTrajectoriesDataLoader,
    DataTransformer,
)
from final_project.code.src.policy.ppo import CustomActorCriticPolicy, PPOPolicy
from final_project.code.src.policy.trainer import ModelTrainer
from final_project.code.src.utils import get_x_pos, load_retrained_weights_to_ppo

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


def create_game_env(*args, **kwargs):
    env = retro.make(game="SuperMarioBros-Nes", *args, **kwargs)
    # Wrap the environment to use discrete, simple action space
    # and custom termination and reward functions
    env = wrapper.JoypadSpace(env, actions.SIMPLE_MOVEMENT)
    env = wrapper.CustomTerminationEnv(env)
    env = wrapper.CustomRewardEnv(env)

    return env


def run_game(
    env,
    num_games=10,
    action_policy=None,
    epsilon=0,
    experiment_name="runs/run_game",
    deterministic=True,
    *args,
    **kwargs,
):
    # writer = SummaryWriter(experiment_name)
    n_game_end_rewards = list()
    n_game_end_positions = list()
    finishline_positions = (12.0 * 256.0) + 70.0  # from the human gameplay end state

    for n in range(num_games):
        logger.info(f"Starting new game {n + 1} of {num_games}")
        cum_rewards = 0

        terminated = truncated = False

        # move the first action always to the right (action 2)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(2)

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

        # end-game statistics
        n_game_end_rewards.append(cum_rewards)
        n_game_end_positions.append(get_x_pos(info) / finishline_positions)

    # log the end files to csv
    df = pd.DataFrame(
        {"cum_rewards": n_game_end_rewards, "completion": n_game_end_positions}
    )
    df.to_csv(experiment_name + "_log.csv", index=False)

    # # for plots
    # """Logs a box plot image to TensorBoard"""
    # fig, ax = plt.subplots()
    # ax.boxplot(np.array(n_game_end_rewards), vert=True, patch_artist=True)
    # ax.set_title("Reward Distribution per Episode")
    # ax.set_xlabel("Episodes")
    # ax.set_ylabel("Reward")
    #
    # # Convert figure to TensorBoard image format
    # writer.add_figure("Boxplot/Reward Distribution", fig)
    # plt.close(fig)
    #
    # fig, ax = plt.subplots()
    # ax.boxplot(np.array(n_game_end_positions), vert=True, patch_artist=True)
    # ax.set_title("Level Completion per Episode")
    # ax.set_xlabel("Episodes")
    # ax.set_ylabel("Level Completion")
    #
    # # Convert figure to TensorBoard image format
    # writer.add_figure("Boxplot/Level Completion Distribution", fig)
    # plt.close(fig)

    return


def run_policy(checkpoint_path, ppo=False, *args, **kwargs):
    env = create_game_env(state="Level1-1")
    if ppo:
        policy = PPOPolicy(checkpoint_path)
    else:
        policy = ModelTrainer(
            model=CNNPolicy(
                input_height=224,
                input_width=240,
                action_dim=len(actions.SIMPLE_MOVEMENT),
            ),
            train_dataloader=None,
            dev_dataloader=None,
            optimizer=None,
            criterion=None,
        )
        policy.load_checkpoint(checkpoint_path)

    # run game using the trained policy
    run_game(env, action_policy=policy, *args, **kwargs)
    return


def train_bc_policy(
    human_traj_folder="human_demon", num_epochs=500, eval_interval=5, *args, **kwargs
):
    train_dl, dev_dl = HumanTrajectoriesDataLoader(
        human_traj_folder, split=True, train_fraction=0.7, batch_size=32, shuffle=True
    )

    model = CNNPolicy(
        input_height=224, input_width=240, action_dim=len(actions.SIMPLE_MOVEMENT)
    )
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


def train_dqn_policy():
    agent = dqn.DQNAgent(
        state_dim_height=224,
        state_dim_width=240,
        action_dim=len(actions.SIMPLE_MOVEMENT),
        buffer_capacity=10000,
        batch_size=32,
        gamma=0.99,
        lr=1e-4,
    )

    env = create_game_env(state="Level1-1")
    dqn.train(agent, env, num_episodes=2500, log_dir="runs/dqn_experiment_new_reward")


def train_ppo_policy(
    training_steps=100000,  # Total environment steps
    checkpoint_freq=500,
    model_name="ppo",
    n_envs=2,  # Number of parallel environments
    warm_start=False,
    pretrained_weights_path=None,
):
    # Create multiple environments using SubprocVecEnv for parallel processing
    env = SubprocVecEnv(
        [lambda: create_game_env(state="Level1-1") for _ in range(n_envs)]
    )

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

    # Initialize PPO with multiple environments
    if warm_start:
        model = PPO.load(f"{checkpoint_dir}/final_model")
        model.set_env(env)
    else:
        model = PPO(
            CustomActorCriticPolicy,
            env,
            n_steps=1024,  # Number of steps per rollout (per environment)
            batch_size=32,
            n_epochs=10,  # Number of optimization epochs per rollout batch
            learning_rate=1e-4,
            tensorboard_log=f"runs/{model_name}",
            verbose=1,
        )

    if pretrained_weights_path:
        # loading BC feature extractor to PPO model
        load_retrained_weights_to_ppo(model, pretrained_weights_path)

    # Train the model with both evaluation and checkpoint callbacks
    model.learn(total_timesteps=training_steps, callback=checkpoint_callback)

    # Save the final model
    model.save(f"{checkpoint_dir}/final_model")


def main():
    env = create_game_env(state="Level1-1")

    # only use one-life setting
    run_game(env, num_games=10)


if __name__ == "__main__":
    # main()

    # Example usage
    set_seed(12345)
    # train_dqn_policy()
    # train_bc_policy(
    #     human_traj_folder="human_demon",
    #     checkpoint_name=f"{model_name}.pt",
    #     num_epochs=500,
    #     eval_interval=5,
    #     early_stopping_patience=10,
    # )

    model_name = "hrl_bc_ppo_model"
    train_ppo_policy(
        training_steps=1000000,
        checkpoint_freq=50000,
        model_name=model_name,
        n_envs=2,
        warm_start=False,
        pretrained_weights_path=f"checkpoints/best_checkpoint_reload_new_arch.pt",
    )

    # # final eval
    # epsilon = 0.0
    # model_name = "best_checkpoint_reload_new_arch"
    # run_policy(
    #     f"checkpoints/{model_name}.pt",
    #     num_games=10,
    #     epsilon=epsilon,
    #     experiment_name=f"runs/run_{model_name}_policy_{epsilon}",
    # )

    # run_policy(
    #     f"checkpoints/best_checkpoint_reload_new_arch.pt",
    #     num_games=10,
    #     epsilon=0,
    #     experiment_name=f"runs/run_bc_policy_0",
    #     ppo=False,
    # )

    # run_policy(
    #     f"checkpoints/best_checkpoint_reload_new_arch.pt",
    #     num_games=50,
    #     epsilon=0.01,
    #     experiment_name=f"runs/run_bc_policy_0.01",
    #     ppo=False,
    # )

    # # train_dqn_policy()
