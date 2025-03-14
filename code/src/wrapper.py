import logging
import math

import numpy as np
from gymnasium.core import ActionWrapper, Env
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack

from final_project.code.src.utils import get_x_pos
import retro
import torch
from torchvision import transforms as T


logger = logging.getLogger(__name__)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


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
        info["is_timeout"] = False
        info["is_winning"] = False
        self.prev_x_pos = None
        self.prev_life = None
        self.no_movement_frames = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if Mario is moving forward:
        x_pos = get_x_pos(info)
        if self.prev_x_pos is not None and x_pos <= self.prev_x_pos:
            self.no_movement_frames += 1
        else:
            self.no_movement_frames = 0  # Reset counter if Mario moves forward

        self.prev_x_pos = x_pos

        # check if winning (moved to the next round)
        if info["levelLo"] > 0:
            info["is_winning"] = True
            terminated = True
            logger.info("WIN! Mario has moved to the next round. Terminating episode.")

        # Termination case 1: not moving forward
        # Terminate if Mario hasn't moved forward for too long
        if self.no_movement_frames >= self.max_no_movement_frames:
            logger.info(
                "Mario has stopped moving forward for too long. Terminating episode."
            )
            terminated = True
            info["is_timeout"] = True

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
        self.cum_reward = 0

    def get_game_completion(self, info):
        finishline_positions = 13.0 * 256.0
        x_pos = get_x_pos(info)

        return round(x_pos / finishline_positions, 2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["cum_reward"] = 0
        # Initialize the last_score using the score from info (if available)
        self.last_info = info
        self.cum_reward = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.cum_reward += reward
        info["cum_reward"] = self.cum_reward

        new_reward = self.reward_function(info, self.last_info, terminated, truncated)

        self.last_info = info
        return obs, new_reward, terminated, truncated, info

    def reward_function(
        self,
        current_info,
        last_info,
        terminated,
        truncated,
        death_penalty=-1000,  # Still penalize dying but not too extreme
        win_reward=1000,
        timeout_penalty=-100,
        time_penalty_per_step=-0.1,  # Small penalty per step to prevent stalling
        progress_reward_weight=0.1,  # scaling x_pos
    ):
        """Modified reward function for better learning."""

        # Extract game state info
        score_diff = current_info.get("score", 0) - last_info.get("score", 0)
        curr_xpos = get_x_pos(current_info)
        xpos_diff = curr_xpos - get_x_pos(last_info)
        current_completion_level = self.get_game_completion(current_info)

        # game completion milestones (every 10% of the game, staring from 20%)
        milestones = [round(0.1 * x, 1) for x in range(2, 11)]
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
        if not (truncated or terminated):
            new_reward = (
                score_diff + progress_reward + time_penalty_per_step + milestone_bonus
            )
        elif current_info.get("is_winning", False):
            new_reward = win_reward
        elif current_info.get("is_timeout", False):
            new_reward = timeout_penalty
        else:
            new_reward = death_penalty

        return new_reward


class ExpertTrajResetEnv(gym.Wrapper):
    def __init__(self, env, expert_traj, total_reset_steps=10, decay_factor=0.8):
        super().__init__(env)
        self.expert_traj = expert_traj  #  states
        self.t = len(expert_traj) - 1  # start from the end
        self.holding_steps_per_state = self.exponential_decay(
            N=total_reset_steps, T=len(expert_traj), r=decay_factor
        )
        self.current_traj_counter = 0

    @staticmethod
    def exponential_decay(N, T, r=0.8):
        """
        Generate a list of T values that decay exponentially such that the sum is N.

        Parameters:
        - N: total sum we want to approximate.
        - T: number of steps (T must be > 0).
        - r: decay factor (0 < r < 1). Smaller r gives faster decay.

        Returns:
        - A list of length T where the t-th element is F(t) = A * r^(t-1)
          and sum(F) is approximately N.
        """
        # Compute the scaling factor A so that the sum equals N.
        A = N * (1 - r) / (1 - r**T)

        # Generate the list of F(t) values.
        values = [math.ceil(A * r ** (t - 1)) for t in range(1, T + 1)]
        return values

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.t >= 0:

            self.env.get_wrapper_attr("em").set_state(self.expert_traj[self.t])
            # just to get the obs and info without
            # taking any action
            obs, _, _, _, info = self.env.step(0)
            self.env.get_wrapper_attr("em").set_state(self.expert_traj[self.t])

            self.current_traj_counter += 1
            if self.current_traj_counter >= self.holding_steps_per_state[self.t]:
                self.t = max(-1, self.t - 1)
                if self.t >= 0:
                    logger.warning(
                        f"Next reset will move to expert trajectory"
                        f"{self.expert_traj.get_file_name(self.t)} for "
                        f"{self.holding_steps_per_state[self.t]} steps."
                    )
                else:
                    logger.warning(
                        "Expert trajectory exhausted, will resume normal reset."
                    )
                self.current_traj_counter = 0

            # take no-op action
        return obs, info


def create_game_env(*args, **kwargs):
    from final_project.code.src.actions import SIMPLE_MOVEMENT

    env = retro.make(game="SuperMarioBros-Nes", state=kwargs.get("state", "Level1-1"))
    # Wrap the environment to use discrete, simple action space
    # and custom termination and reward functions
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomTerminationEnv(env)
    env = CustomRewardEnv(env)

    # Reduce the space to grayscale and resize
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    expert_traj = kwargs.get("expert_traj", None)
    if expert_traj is not None:
        env = ExpertTrajResetEnv(
            env,
            expert_traj,
            total_reset_steps=kwargs.get("total_reset_steps", 10),
            decay_factor=kwargs.get("decay_factor", 0.8),
        )

    return env


def create_game_env_human_demon(*args, **kwargs):
    from final_project.code.src.actions import SIMPLE_MOVEMENT

    env = retro.make(game="SuperMarioBros-Nes", *args, **kwargs)
    # Wrap the environment to use discrete, simple action space
    # and custom termination and reward functions
    # env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env = CustomTerminationEnv(env)
    env = CustomRewardEnv(env)

    # Reduce the space to grayscale and resize
    # env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env
