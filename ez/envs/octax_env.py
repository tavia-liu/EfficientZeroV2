import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cv2
import gym
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces

from octax.environments import create_environment


class OctaxGymEnv(gym.Env):
    """Adapter from Octax (JAX, gymnax-style) to classic gym 0.22 (4-tuple step)."""

    metadata = {"render.modes": []}

    def __init__(self, game: str = "brix", img_size: int = 96, seed: int = 0):
        self._env, self._meta = create_environment(game, frame_skip=1)
        self._img_size = int(img_size)
        self._rng = jax.random.PRNGKey(int(seed))

        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)

        self.action_space = spaces.Discrete(int(self._env.num_actions))
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, self._img_size, self._img_size),
            dtype=np.uint8,
        )
        self._state = None

    def _format_obs(self, obs):
        # Octax obs is (frame_skip, W, H) bool/uint8. With frame_skip=1 -> (1, W, H).
        frame = np.asarray(obs)[-1]                 # (W, H)
        frame = (frame.astype(np.uint8) * 255)      # binary -> 0/255
        frame = cv2.resize(frame, (self._img_size, self._img_size),
                           interpolation=cv2.INTER_NEAREST)
        return np.repeat(frame[None, :, :], 3, axis=0)  # (3, H, W)

    def seed(self, seed=None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(int(seed))
        return [seed]

    def reset(self, **kwargs):
        self._rng, sub = jax.random.split(self._rng)
        self._state, obs, _ = self._reset_fn(sub)
        return self._format_obs(obs)

    def step(self, action):
        self._state, obs, reward, terminated, truncated, info = self._step_fn(
            self._state, jnp.asarray(action, dtype=jnp.int32)
        )
        done = bool(terminated) or bool(truncated)
        return self._format_obs(obs), float(reward), done, {"score": float(info.get("score", 0.0))}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


_REGISTERED = False


def _register_octax_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    for g in ["brix", "pong", "tetris", "blinky", "worm", "squash",
              "missile", "tank", "ufo", "airplane"]:
        try:
            gym.envs.registration.register(
                id=f"Octax-{g.capitalize()}-v0",
                entry_point="ez.envs.octax_env:OctaxGymEnv",
                kwargs={"game": g},
            )
        except gym.error.Error:
            pass
    _REGISTERED = True


_register_octax_envs()
