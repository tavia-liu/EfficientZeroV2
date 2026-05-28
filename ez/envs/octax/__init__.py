from ez.envs.base import BaseWrapper


class OctaxWrapper(BaseWrapper):
    """Convert Octax's Gymnasium-style API to EZv2's classic Gym API."""

    def __init__(self, env, obs_to_string=False):
        super().__init__(env, obs_to_string, False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.format_obs(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.format_obs(obs)
        info["raw_reward"] = reward
        return obs, reward, terminated or truncated, info

