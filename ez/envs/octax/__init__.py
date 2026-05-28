"""Gymnasium-style adapter around octax's JAX-functional CHIP-8 env.

Octax ships an `OctaxEnv` whose `step(state, action)` is JAX-functional (returns
new state) and does not register with gymnasium. This wrapper hides the JAX
state behind a stateful, classic-gym (4-tuple) interface that EZv2's data
pipeline consumes, and emits grayscale `uint8` frames at CHIP-8's native 32x64.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # keep JAX off the GPU; PyTorch owns it

import gym
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces

from octax.environments import create_environment

from ez.utils.format import arr_to_str


class OctaxWrapper(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, game, seed=0, obs_to_string=False, clip_reward=False,
                 frame_skip=4, max_episode_steps=4500):
        super().__init__()
        self.game = game
        self.obs_to_string = obs_to_string
        self.clip_reward = clip_reward

        env, _meta = create_environment(
            game,
            render_mode="rgb_array",
            frame_skip=frame_skip,
            max_num_steps_per_episodes=max_episode_steps,
        )
        self.env = env

        self._key = jax.random.PRNGKey(int(seed))
        self._state = None

        # CHIP-8 display is 64x32 (width x height); EmulatorState.display is (H, W).
        self.action_space = spaces.Discrete(self.env.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(32, 64, 1), dtype=np.uint8,
        )

    def _frames_to_obs(self, frames):
        # frames: (frame_skip, W=64, H=32) bool. Octax stores display width-major
        # (see octax.constants.SCREEN_WIDTH/HEIGHT). Max-pool over time, transpose
        # to (H, W), scale to uint8, add channel dim → standard HWC.
        frame = np.asarray(jnp.max(frames, axis=0))      # (64, 32)
        frame = (frame.astype(np.uint8) * 255).T         # (32, 64)
        return frame[..., None]                          # (32, 64, 1)

    def _format(self, obs):
        if self.obs_to_string:
            return arr_to_str(obs.astype(np.uint8))
        return obs

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self._key = jax.random.PRNGKey(int(seed))
        self._key, sub = jax.random.split(self._key)
        self._state, frames, _info = self.env.reset(sub)
        return self._format(self._frames_to_obs(frames))

    def step(self, action):
        self._state, frames, reward, terminated, truncated, info = self.env.step(
            self._state, int(action)
        )
        obs = self._format(self._frames_to_obs(frames))
        reward = float(reward)
        done = bool(terminated) or bool(truncated)
        info = dict(info)
        info["raw_reward"] = reward
        if self.clip_reward:
            reward = float(np.sign(reward))
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return self.env.render(self._state)

    def close(self):
        return None
