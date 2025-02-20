from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import ActionWrapper, Env, WrapperActType, WrapperObsType
from src.utils import get_x_pos
import numpy as np
import logging

logger = logging.getLogger(__name__)


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


class CustomTerminationEnv(gym.Wrapper):
    LOSE_PENALTY = -1000

    def __init__(self, env, max_no_movement_time=10):
        super().__init__(env)
        self.max_no_movement_time = max_no_movement_time  # seconds
        self.fps = 60  # Assuming 60 FPS for SMB
        self.max_no_movement_frames = self.max_no_movement_time * self.fps
        self.prev_x_pos = None
        self.no_movement_frames = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = None
        self.no_movement_frames = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = get_x_pos(info)

        # Check if Mario is moving forward:
        if self.prev_x_pos is not None and x_pos <= self.prev_x_pos:
            self.no_movement_frames += 1
        else:
            self.no_movement_frames = 0  # Reset counter if Mario moves forward

        self.prev_x_pos = x_pos

        # Terminate if Mario hasn't moved forward for too long
        if self.no_movement_frames >= self.max_no_movement_frames:
            logger.info(
                "Mario has stopped moving forward for too long. Terminating episode."
            )
            terminated = True
            reward = self.LOSE_PENALTY

        # Terminate if Mario loses his only life
        lives = info.get("lives", 2)  # Adjust default if needed
        if lives < 1:
            logger.info("Mario lost his only life. Terminating episode.")
            terminated = True
            reward = self.LOSE_PENALTY

        return obs, reward, terminated, truncated, info


class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize the last_score using the score from info (if available)
        self.last_score = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # reward = score + x_pos
        score_diff = info.get("score", 0) - self.last_score
        self.last_score = info.get("score", 0)
        reward += score_diff
        return obs, reward, terminated, truncated, info
