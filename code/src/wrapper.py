import logging

import gymnasium as gym
import numpy as np
from gymnasium.core import ActionWrapper, Env
from final_project.code.src.utils import get_x_pos

logger = logging.getLogger(__name__)


class JoypadSpace(ActionWrapper):
    """An environment wrapper to convert binary to discrete action space."""

    _button_to_bitmap = {"NULL": 0b000000000}  # INIT
    _button_to_bitmap.update(
        {
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
    )

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
        # initialize the action maps
        self._bitmap_to_dsct_action = {}
        self._dsct_action_to_bitmap = {}
        # map the allowed actions
        self.map_allowed_actions(allowed_actions)

    @staticmethod
    def array_to_bitmap(action_array):
        """action is list of int and return the bitmap string"""
        binary_string = "0b"
        for b in action_array:
            binary_string += str(b)
        return int(binary_string, 2)

    @staticmethod
    def bitmap_to_array(action_bitmap):
        """convert action_bitmap to numpy int array"""
        binary_string = f"{action_bitmap:09b}"  # 9 bits
        return np.array([int(b) for b in binary_string], dtype=np.int8)

    def map_allowed_actions(self, allowed_actions):
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(allowed_actions):
            # the value of this action's bitmap
            byte_action = 0
            # iterate over the buttons in this button list
            for button in button_list:
                byte_action |= self._button_to_bitmap[button]

            # set this action maps value to the byte action value
            self._dsct_action_to_bitmap[action] = byte_action
            self._bitmap_to_dsct_action[byte_action] = action

    def get_discrete_action_from_array(self, action_array):
        """Convert a binary action array to a discrete action."""
        bitmap = self.array_to_bitmap(action_array)
        if bitmap in self._bitmap_to_dsct_action:
            return self._bitmap_to_dsct_action[bitmap]
        else:
            # sample a random action if the action is not allowed
            logger.warning(f"Action {action_array} not allowed. Moving right.")
            return self._bitmap_to_dsct_action[2]

    def action(self, action: int) -> np.ndarray:
        """Convert a discrete action to a binary action array."""
        return self.bitmap_to_array(self._dsct_action_to_bitmap[action])


class CustomTerminationEnv(gym.Wrapper):

    def __init__(self, env, max_no_movement_time=10):
        super().__init__(env)
        self.max_no_movement_time = max_no_movement_time  # seconds
        self.fps = 60  # Assuming 60 FPS for SMB
        self.max_no_movement_frames = self.max_no_movement_time * self.fps
        self.prev_x_pos = None
        self.prev_life = None
        self.no_movement_frames = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = None
        self.prev_life = None
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

        # Terminate if Mario loses his only life
        if self.prev_life is not None:
            lives = info.get("lives")
            if lives < self.prev_life:
                logger.info("Mario lost a life. Terminating episode.")
                terminated = True
        else:
            self.prev_life = info.get("lives")

        return obs, reward, terminated, truncated, info


class CustomRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_info = dict()
        self.milestone_reached = list()

    def get_game_completion(self, info):
        finishline_positions = (12.0 * 256.0) + 70.0
        x_pos = get_x_pos(info)

        return round(x_pos / finishline_positions, 2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize the last_score using the score from info (if available)
        self.last_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated
        new_reward = self.reward_function(reward, info, self.last_info, done)

        self.last_info = info
        return obs, new_reward, terminated, truncated, info

    def reward_function(
        self,
        current_reward,
        current_info,
        last_info,
        done=False,
        death_penalty=-500,  # Still penalize dying but not too extreme
        time_penalty_per_step=-0.1,  # Small penalty per step to prevent stalling
        progress_reward_weight=0.1,  # scaling x_pos
    ):
        """Modified reward function for better learning."""

        # Extract game state info
        score_diff = current_info.get("score", 0) - last_info.get("score", 0)
        curr_xpos = get_x_pos(current_info)
        xpos_diff = curr_xpos - get_x_pos(last_info)
        current_completion_level = self.get_game_completion(current_info)

        # game completion milestones (every 10% of the game)
        milestones = [round(0.1 * x, 1) for x in range(11)]
        milestone_reached = list(
            filter(lambda x: x <= current_completion_level, milestones)
        )
        recent_milestone = milestone_reached[-1] if milestone_reached else 0.0
        if recent_milestone not in self.milestone_reached:
            milestone_bonus = recent_milestone * 1000
        else:
            milestone_bonus = 0

        self.milestone_reached = milestone_reached

        # Encourage moving right (progress reward) if left then negative
        progress_reward = xpos_diff * progress_reward_weight

        # Reward function
        if not done:
            new_reward = (
                score_diff + progress_reward + time_penalty_per_step + milestone_bonus
            )
        else:
            new_reward = death_penalty

        return new_reward


# def reward_function(
#     current_reward, current_info, last_info, done=False, death_penalty=-10000
# ):
#     """current reward is delta_x_pos"""
#     if not done:
#         score_diff = current_info.get("score") - last_info.get("score", 0)
#         time_diff = current_info.get("time") - last_info.get("time", 400)
#         new_reward = current_reward + score_diff + time_diff
#     else:
#         new_reward = death_penalty
#     return new_reward
