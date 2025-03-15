import logging

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)

from final_project.code.src.actions import SIMPLE_MOVEMENT
from final_project.code.src.utils import load_trajectories
from final_project.code.src.wrapper import JoypadSpace

logger = logging.getLogger(__name__)


class DataTransformer:
    def __init__(self):
        self.action_env = JoypadSpace(
            None, SIMPLE_MOVEMENT
        )  # dummy space for action wrapper

    @staticmethod
    def transform_state(state):
        raise NotImplementedError

    def transform_action(self, action):
        act = self.action_env.get_discrete_action_from_array(action)
        return torch.tensor(act, dtype=torch.long)

    @staticmethod
    def transform_rewards(*args, **kwargs):
        raise NotImplementedError


class HumanTrajectories(Dataset):
    def __init__(self, traj_folder):
        """
        Initialize a new dataset of human trajectories.

        Args:
            traj_folder: path to the folder containing the human trajectories

        Returns:
            None
        """
        self.states, self.actions, self.info = load_trajectories(traj_folder)
        self.transformer = DataTransformer()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, item):
        state = self.states[item]
        action = self.transformer.transform_action(self.actions[item])
        return state, action


def split_dataloader(
    dataset, sampler=None, train_fraction=0.8, batch_size=32, shuffle=True
):
    """
    Splits the dataset into train and dev sets based on the given fraction.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_fraction (float): Fraction of data to use for the training set.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        dev_loader (DataLoader): DataLoader for the development (dev) set.
    """
    # Calculate sizes for train and dev sets
    train_size = int(len(dataset) * train_fraction)
    dev_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, dev_dataset = random_split(dataset, [train_size, dev_size])

    # Optionally shuffle the data
    if sampler:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader


def HumanTrajectoriesDataLoader(
    traj_folder, split=False, train_fraction=0.8, batch_size=32, shuffle=True
):
    dataset = HumanTrajectories(traj_folder)
    if not split:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_data, dev_data = split_dataloader(
            dataset,
            train_fraction=train_fraction,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return train_data, dev_data
