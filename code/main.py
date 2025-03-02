import retro
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sympy.physics.paulialgebra import epsilon

import src.actions as actions
import src.wrapper as wrapper
from src.utils import get_x_pos
from src.policy.dataset import HumanTrajectoriesDataLoader, DataTransformer
from src.policy.bc import BehaviorCloningPolicy
from src.policy.trainer import ModelTrainer
import src.policy.dqn as dqn
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


def run_game(env, num_games=1, action_policy=None, epsilon=0):
    for n in range(num_games):
        logger.info(f"Starting new game {n + 1} of {num_games}")
        reward_stats = list()
        x_pos = list()
        terminated = truncated = False

        # move the first action always to the right (action 2)
        env.reset()
        observation, reward, terminated, truncated, info = env.step(2)

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
            reward_stats.append(reward)
            x_pos.append(get_x_pos(info))
            env.render()

        plt.figure()
        plt.plot(np.cumsum(reward_stats), label="cumulative reward")
        plt.plot(np.array(x_pos), label="x_pos")
        plt.legend()
        plt.show()


def run_bc_policy(checkpoint_path, *args, **kwargs):
    env = create_game_env(state="Level1-1")
    policy = ModelTrainer(
        model=BehaviorCloningPolicy(
            input_height=224, input_width=240, action_dim=len(actions.SIMPLE_MOVEMENT)
        ),
        train_dataloader=None,
        dev_dataloader=None,
        optimizer=None,
        criterion=None,
    )
    policy.load_checkpoint(checkpoint_path)

    # run game using the trained policy
    run_game(env, num_games=10, action_policy=policy, *args, **kwargs)
    return


def train_bc_policy(human_traj_folder="human_demon", *args, **kwargs):
    train_dl, dev_dl = HumanTrajectoriesDataLoader(
        human_traj_folder, split=True, train_fraction=0.7, batch_size=32, shuffle=True
    )

    model = BehaviorCloningPolicy(
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
    # run_bc_policy("checkpoints/best_checkpoint.pt", epsilon=0.01)
    train_dqn_policy()
