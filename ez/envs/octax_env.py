import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cv2
import gym
import numpy as np
from gym import spaces


class OctaxGymEnv(gym.Env):
    """Adapter from Octax (JAX) to classic gym 0.22 (4-tuple step).

    JAX/octax are imported LAZILY in __init__ to avoid parent-process imports
    that would deadlock when Ray forks workers (JAX is multithreaded, fork-unsafe).
    """

    metadata = {"render.modes": []}

    # Hardcoded num_actions per game (read once via probe; avoids importing jax in parent).
    # Octax games use a subset of CHIP-8's 16 keys + 1 no-op.
    # num_actions = len(action_set) + 1 (no-op). Verified against octax/environments/*.py.
    _NUM_ACTIONS = {
        "airplane": 2, "blinky": 5, "brix": 3, "cavern": 5, "deep": 4,
        "filter": 3, "flight_runner": 5, "missile": 2, "pong": 3,
        "rocket": 2, "shooting_stars": 5, "space_flight": 3, "spacejam": 5,
        "squash": 3, "submarine": 2, "tank": 6, "target_shooter": 6,
        "tetris": 5, "ufo": 4, "vertical_brix": 3, "wipe_off": 3, "worm": 5,
    }

    def __init__(self, game: str = "brix", img_size: int = 64, seed: int = 0):
        self._game = game
        self._img_size = int(img_size)
        self._seed = int(seed)
        self._initialized = False
        self._jax = self._jnp = None
        self._env = self._state = self._reset_fn = self._step_fn = None

        n_actions = self._NUM_ACTIONS.get(game)
        if n_actions is None:
            raise ValueError(f"Unknown octax game '{game}'; add to _NUM_ACTIONS table.")
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, self._img_size, self._img_size),
            dtype=np.uint8,
        )

    def _lazy_init(self):
        if self._initialized:
            return
        import jax
        import jax.numpy as jnp
        from octax.environments import create_environment
        self._jax, self._jnp = jax, jnp
        self._env, self._meta = create_environment(self._game, frame_skip=1)
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)
        self._rng = jax.random.PRNGKey(self._seed)
        # Verify hardcoded action count matches runtime
        assert int(self._env.num_actions) == self.action_space.n, (
            f"{self._game}: hardcoded {self.action_space.n}, runtime {self._env.num_actions}"
        )
        self._initialized = True

    def _format_obs(self, obs):
        frame = np.asarray(obs)[-1]                      # last frame
        frame = (frame.astype(np.uint8) * 255)           # binary -> 0/255
        frame = cv2.resize(frame, (self._img_size, self._img_size),
                           interpolation=cv2.INTER_NEAREST)
        return np.repeat(frame[None, :, :], 3, axis=0)   # (3, H, W)

    def seed(self, seed=None):
        self._seed = int(seed) if seed is not None else self._seed
        if self._initialized:
            self._rng = self._jax.random.PRNGKey(self._seed)
        return [seed]

    def reset(self, **kwargs):
        self._lazy_init()
        self._rng, sub = self._jax.random.split(self._rng)
        self._state, obs, _ = self._reset_fn(sub)
        return self._format_obs(obs)

    def step(self, action):
        self._lazy_init()
        self._state, obs, reward, terminated, truncated, info = self._step_fn(
            self._state, self._jnp.asarray(action, dtype=self._jnp.int32)
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
    for g in ["airplane", "blinky", "brix", "cavern", "deep", "filter",
              "flight_runner", "missile", "pong", "rocket", "shooting_stars",
              "space_flight", "spacejam", "squash", "submarine", "tank",
              "target_shooter", "tetris", "ufo", "vertical_brix", "wipe_off", "worm"]:
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
