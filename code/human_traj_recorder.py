"""
Interact with Gym environments using the keyboard

An adapter object is defined for each environment to map keyboard commands to actions and extract observations as pixels.
"""

import abc
import argparse
import ctypes
import sys
import time
import pickle

import numpy as np
import pyglet
from pyglet import gl
from pyglet.window import key as keycodes

import retro
from triton.language import dtype


class Interactive(abc.ABC):
    """
    Base class for making gym environments interactive for human use.
    This version records each transition into a list and, once a terminated (or truncated) status is reached,
    exits the interactive loop and saves the trajectory to a file.
    """

    def __init__(
        self, env, sync=True, tps=60, aspect_ratio=None, traj_path="trajectory.pkl"
    ):
        obs = env.reset()
        self._image = self.get_image(obs, env)
        assert (
            len(self._image.shape) == 3 and self._image.shape[2] == 3
        ), "must be an RGB image"
        image_height, image_width = self._image.shape[:2]

        if aspect_ratio is None:
            aspect_ratio = image_width / image_height

        # Compute a window size that is not too small or too large.
        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.9
        max_win_height = screen.height * 0.9
        win_width = image_width
        win_height = int(win_width / aspect_ratio)

        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            image_width,
            image_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )

        self._env = env
        self._win = win

        self._key_previous_states = {}

        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

        # Initialize an empty list to record trajectory transitions.
        self._trajectory = []
        # Save trajectory path from argument.
        self._traj_path = traj_path

    def _update(self, dt):
        # Cap dt to avoid huge jumps.
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)
                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            # For synchronous mode, only use keys that were just clicked.
            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            # Step the environment if an action is available.
            if not self._sync or act is not None:
                obs, rew, terminated, truncated, _info = self._env.step(act)
                done = terminated or truncated
                self._image = self.get_image(obs, self._env)
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1

                # Record the transition (only record the first level)
                if _info["levelHi"] == 0 and _info["levelLo"] == 0:
                    self._trajectory.append(
                        {
                            "step": self._steps,
                            "episode_steps": self._episode_steps,
                            "observation": obs,
                            "action": act,
                            "reward": rew,
                            "done": done,
                            "info": _info,
                        }
                    )

                np.set_printoptions(precision=2)
                if self._sync:
                    done_int = int(done)
                    mess = f"steps={self._steps} episode_steps={self._episode_steps} rew={rew} episode_returns={self._episode_returns} done={done_int}"
                    print(mess)
                elif self._steps % self._tps == 0 or done:
                    episode_returns_delta = (
                        self._episode_returns - self._prev_episode_returns
                    )
                    self._prev_episode_returns = self._episode_returns
                    mess = f"steps={self._steps} episode_steps={self._episode_steps} episode_returns_delta={episode_returns_delta} episode_returns={self._episode_returns}"
                    print(mess)

                # Check if level completion flag is set.
                if _info.get("flag_get", False):
                    print(
                        "Level finished. Exiting interactive mode and saving trajectory..."
                    )
                    self._on_close()

                # If terminated/truncated status is reached, exit and save.
                if done:
                    print(
                        "Episode terminated. Exiting interactive mode and saving trajectory..."
                    )
                    self._on_close()

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(
            self._image.tobytes(),
            ctypes.POINTER(ctypes.c_short),
        )
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self._image.shape[1],
            self._image.shape[0],
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            video_buffer,
        )

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ("v2f", [x, y, x + w, y, x + w, y + h, x, y + h]),
            ("t2f", [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        # Save the recorded trajectory before closing.
        with open(self._traj_path, "wb") as f:
            pickle.dump(self._trajectory, f)
        print("Trajectory saved to", self._traj_path)
        # print(self._trajectory[-1])
        self._env.close()
        sys.exit(0)

    @abc.abstractmethod
    def get_image(self, obs, venv):
        """
        Given an observation and the Env object, return an RGB array to display.
        """
        pass

    @abc.abstractmethod
    def keys_to_act(self, keys):
        """
        Given a list of keys that the user has input, produce a gym action to pass to the environment.
        For sync environments, keys is a list of keys that have been pressed since the last step.
        For async environments, keys is a list of keys currently held down.
        """
        pass

    def run(self):
        """
        Run the interactive window until the user quits.
        """
        prev_frame_time = time.time()
        while True:
            self._win.switch_to()
            self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            self._draw()
            self._win.flip()


class RetroInteractive(Interactive):
    """
    Interactive setup for retro games.
    """

    def __init__(self, game, state, scenario, record, traj_path="trajectory.pkl"):
        env = retro.make(
            game=game,
            state=state,
            scenario=scenario,
            record=record,
            render_mode="rgb_array",
        )
        self._buttons = env.buttons
        super().__init__(
            env=env, sync=False, tps=60, aspect_ratio=4 / 3, traj_path=traj_path
        )

    def get_image(self, _obs, env):
        return env.render()

    def keys_to_act(self, keys):
        inputs = {
            None: False,
            "BUTTON": "Z" in keys,
            "A": "Z" in keys,
            "B": "X" in keys,
            "C": "C" in keys,
            "X": "A" in keys,
            "Y": "S" in keys,
            "Z": "D" in keys,
            "L": "Q" in keys,
            "R": "W" in keys,
            "UP": "UP" in keys,
            "DOWN": "DOWN" in keys,
            "LEFT": "LEFT" in keys,
            "RIGHT": "RIGHT" in keys,
            "MODE": "TAB" in keys,
            "SELECT": "TAB" in keys,
            "RESET": "ENTER" in keys,
            "START": "ENTER" in keys,
        }
        return np.array([inputs[b] for b in self._buttons], dtype=np.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SuperMarioBros-Nes")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--record", default=False, nargs="?", const=True)
    parser.add_argument(
        "--traj_path",
        default=f"human_demon/trajectory_{time.time_ns()}.pkl",
        help="Path to save the recorded trajectory",
    )
    args = parser.parse_args()

    ia = RetroInteractive(
        game=args.game,
        state=args.state,
        scenario=args.scenario,
        record=args.record,
        traj_path=args.traj_path,
    )
    ia.run()


if __name__ == "__main__":
    main()
