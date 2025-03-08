import pandas as pd
import retro
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import final_project.code.src.actions as actions
import final_project.code.src.wrapper as wrapper
from final_project.code.src.utils import get_x_pos
from final_project.code.src.policy.dataset import (
    HumanTrajectoriesDataLoader,
    DataTransformer,
)
from final_project.code.src.policy.base import CNNPolicy
from final_project.code.src.policy.trainer import ModelTrainer
import final_project.code.src.policy.dqn as dqn
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def create_game_env(*args, **kwargs):
    env = retro.make(game="SuperMarioBros-Nes", *args, **kwargs)
    # Wrap the environment to use discrete, simple action space
    # and custom termination and reward functions
    env = wrapper.JoypadSpace(env, actions.SIMPLE_MOVEMENT)
    env = wrapper.CustomTerminationEnv(env)
    env = wrapper.CustomRewardEnv(env)

    return env


def run_game(
    env, num_games=10, action_policy=None, epsilon=0, experiment_name="runs/run_game"
):
    writer = SummaryWriter(experiment_name)
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
                    _observation = DataTransformer.transform_state(observation)
                    action = action_policy.sample_action(_observation)
            observation, reward, terminated, truncated, info = env.step(action)
            writer.add_scalar("Game Cumulative Reward", cum_rewards, step)
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

    # for plots
    """Logs a box plot image to TensorBoard"""
    fig, ax = plt.subplots()
    ax.boxplot(np.array(n_game_end_rewards), vert=True, patch_artist=True)
    ax.set_title("Reward Distribution per Episode")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")

    # Convert figure to TensorBoard image format
    writer.add_figure("Boxplot/Reward Distribution", fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.boxplot(np.array(n_game_end_positions), vert=True, patch_artist=True)
    ax.set_title("Level Completion per Episode")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Level Completion")

    # Convert figure to TensorBoard image format
    writer.add_figure("Boxplot/Level Completion Distribution", fig)
    plt.close(fig)

    return


def run_policy(checkpoint_path, *args, **kwargs):
    env = create_game_env(state="Level1-1")
    policy = ModelTrainer(
        model=CNNPolicy(
            input_height=224, input_width=240, action_dim=len(actions.SIMPLE_MOVEMENT)
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


def train_bc_policy(human_traj_folder="human_demon", *args, **kwargs):
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

    trainer.train(num_epochs=500, eval_interval=5)
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
    dqn.train(agent, env, num_episodes=2500, log_dir="runs/dqn_experiment_v2")


def main():
    env = create_game_env(state="Level1-1")

    # only use one-life setting
    run_game(env, num_games=10)


if __name__ == "__main__":
    # main()
    # train_bc_policy(
    #     human_traj_folder="human_demon",
    #     checkpoint_name="best_checkpoint_2_traj.pt",
    # )
    run_policy(
        "checkpoints/dqn_model.pth",
        num_games=10,
        epsilon=0,
        experiment_name="runs/run_dqn_policy_0",
    )
    # train_dqn_policy()
