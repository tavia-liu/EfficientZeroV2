import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gym
import numpy as np
from gym import spaces


SCREEN_WIDTH = 64
SCREEN_HEIGHT = 32

GAME_IDS = (
    "airplane", "blinky", "brix", "cavern", "deep", "filter",
    "flight_runner", "missile", "pong", "rocket", "shooting_stars",
    "space_flight", "spacejam", "squash", "submarine", "tank",
    "target_shooter", "tetris", "ufo", "vertical_brix", "wipe_off", "worm",
)

# From octax/octax/environments/*.py action_set definitions.
# OctaxEnv.num_actions is len(action_set) + 1 because the final action is no-op.
ACTION_COUNTS = {
    "airplane": 2,
    "blinky": 5,
    "brix": 3,
    "cavern": 5,
    "deep": 4,
    "filter": 3,
    "flight_runner": 5,
    "missile": 2,
    "pong": 3,
    "rocket": 2,
    "shooting_stars": 5,
    "space_flight": 3,
    "spacejam": 5,
    "squash": 3,
    "submarine": 2,
    "tank": 6,
    "target_shooter": 6,
    "tetris": 5,
    "ufo": 4,
    "vertical_brix": 3,
    "wipe_off": 3,
    "worm": 5,
}


class OctaxGymEnv(gym.Env):
    """Gym adapter for Octax JAX environments."""

    metadata = {"render.modes": []}

    def __init__(self, game="brix", frame_skip=4, seed=0):
        if game not in ACTION_COUNTS:
            raise ValueError(f"Unknown Octax game: {game}")

        self.game = game
        self.frame_skip = int(frame_skip)
        self._seed = int(seed)
        self._initialized = False
        self._jax = None
        self._jnp = None
        self._env = None
        self._state = None
        self._reset_fn = None
        self._step_fn = None
        self._rng = None

        self.action_space = spaces.Discrete(ACTION_COUNTS[game])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 1),
            dtype=np.uint8,
        )

    def _lazy_init(self):
        if self._initialized:
            return

        import jax
        import jax.numpy as jnp
        from octax.environments import create_environment

        self._jax = jax
        self._jnp = jnp
        self._env, _ = create_environment(self.game, frame_skip=self.frame_skip)
        self._reset_fn = jax.jit(self._env.reset)
        self._step_fn = jax.jit(self._env.step)
        self._rng = jax.random.PRNGKey(self._seed)

        if int(self._env.num_actions) != self.action_space.n:
            raise ValueError(
                f"{self.game}: ACTION_COUNTS has {self.action_space.n}, "
                f"Octax runtime has {self._env.num_actions}"
            )

        self._initialized = True

    def _format_obs(self, obs):
        frame = np.asarray(obs)[-1].T
        return (frame.astype(np.uint8) * 255)[:, :, None]

    def seed(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
        if self._initialized:
            self._rng = self._jax.random.PRNGKey(self._seed)
        return [self._seed]

    def reset(self, **kwargs):
        self._lazy_init()
        self._rng, rng = self._jax.random.split(self._rng)
        self._state, obs, info = self._reset_fn(rng)
        return self._format_obs(obs), {"score": float(info["score"])}

    def step(self, action):
        self._lazy_init()
        self._state, obs, reward, terminated, truncated, info = self._step_fn(
            self._state,
            self._jnp.asarray(action, dtype=self._jnp.int32),
        )
        return (
            self._format_obs(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            {"score": float(info["score"])},
        )

    def render(self, mode="human"):
        return None

    def close(self):
        pass


def _register_octax_envs():
    for game in GAME_IDS:
        try:
            gym.envs.registration.register(
                id=f"Octax-{game.capitalize()}-v0",
                entry_point="ez.envs.octax_env:OctaxGymEnv",
                kwargs={"game": game},
            )
        except gym.error.Error:
            pass


_register_octax_envs()
