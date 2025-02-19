import gymnasium as gym
from gymnasium.core import ActionWrapper, Env
import numpy as np


class JoypadSpace(ActionWrapper):
    """An environment wrapper to convert binary to discrete action space."""

    _button_map = {
        "B": 0b100000000,
        "NOOP": 0b010000000,
        "SELECT": 0b001000000,
        "START": 0b000100000,
        "UP": 0b000010000,
        "DOWN": 0b000001000,
        "LEFT": 0b000000100,
        "RIGHT": 0b000000010,
        "A": 0b000000001,
    }

    @staticmethod
    def array_to_bitmap(action_array):
        binary_string = "0b"
        for b in action_array:
            binary_string += str(b)
        return binary_string

    @staticmethod
    def bitmap_to_array(action_bitmap):
        binary_string = bin(action_bitmap)[2:]
        return np.array([int(b) for b in binary_string], dtype=np.int8)

    def __init__(self, env: Env, allowed_actions: list):
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            allowed_actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(allowed_actions))
        self._action_map = {}  # discrete action number to array of binary actions
        self._action_meanings = {}  # discrete action number to action meaning

        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(allowed_actions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_map[button]
            # set this action maps value to the byte action value
            self._action_map[action] = byte_action
            self._action_meanings[action] = ",".join(button_list)

    def action(self, action: int) -> np.ndarray:
        """
        Convert a discrete action to a binary action array.

        Args:
            action: the discrete action to convert

        Returns:
            the binary action array

        """
        return self.bitmap_to_array(self._action_map[action])
