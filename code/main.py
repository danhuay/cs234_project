import retro
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import src.actions as actions
import src.wrapper as wrapper
from src.utils import get_x_pos
from src.policy.dataset import HumanTrajectoriesDataLoader
from src.policy.bc import BehaviorCloningPolicy
from src.policy.trainer import ModelTrainer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def create_game_env(*args, **kwargs):
    env = retro.make(game="SuperMarioBros-Nes", *args, **kwargs)
    env.reset()

    # Wrap the environment to use discrete, simple action space
    # and custom termination and reward functions
    env = wrapper.JoypadSpace(env, actions.SIMPLE_MOVEMENT)
    env = wrapper.CustomTerminationEnv(env)
    env = wrapper.CustomRewardEnv(env)

    return env


def main():
    env = create_game_env(state="Level1-1")

    # only use one-life setting
    games_to_play = 10
    for n in range(games_to_play):
        logger.info(f"Starting new game {n + 1} of {games_to_play}")
        reward_stats = list()
        x_pos = list()
        terminated = truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            reward_stats.append(reward)
            x_pos.append(get_x_pos(info))
            env.render()

        plt.figure()
        plt.plot(np.cumsum(reward_stats), label="cumulative reward")
        plt.plot(np.array(x_pos), label="x_pos")
        plt.legend()
        plt.show()

        env.reset()


if __name__ == "__main__":
    # main()
    train_dl, dev_dl = HumanTrajectoriesDataLoader(
        "human_demon", split=True, train_fraction=0.7, batch_size=32, shuffle=True
    )

    model = BehaviorCloningPolicy(input_height=224, input_width=240, action_dim=9)
    trainer = ModelTrainer(
        model=model,
        train_dataloader=train_dl,
        dev_dataloader=dev_dl,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        criterion=nn.CrossEntropyLoss(),
    )

    trainer.train(num_epochs=500, eval_interval=5)
