import copy
import time

from omegaconf import open_dict

from ez.agents.ez_atari import EZAtariAgent
from ez.envs import make_octax
from ez.utils.format import DiscreteSupport


class EZOctaxAgent(EZAtariAgent):
    """EZv2 agent for Octax (gym-registered image env). Same network as Atari,
    but uses make_gym to build the env so it doesn't append 'NoFrameskip-v4'."""

    def update_config(self):
        assert not self._update

        env = make_octax(self.config.env.game, seed=0, save_path=None, **self.config.env)
        action_space_size = env.action_space.n

        obs_channel = 1 if self.config.env.gray_scale else 3

        reward_support = DiscreteSupport(self.config)
        reward_size = reward_support.size

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size

        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-{}-seed={}-{}/'.format(self.config.tag, self.config.env.game, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.mcts.num_top_actions = min(action_space_size, self.config.mcts.num_top_actions)
            self.config.env.obs_shape[0] = obs_channel
            self.config.rl.discount **= self.config.env.n_skip
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size

            if action_space_size < 4:
                self.config.mcts.num_top_actions = 2
                self.config.mcts.num_simulations = 4
            elif action_space_size < 16:
                self.config.mcts.num_top_actions = 4
            else:
                self.config.mcts.num_top_actions = 8

            if not self.config.mcts.use_gumbel:
                self.config.mcts.num_simulations = 50
            print(f'env={self.config.env.env}, game={self.config.env.game}, |A|={action_space_size}, '
                  f'top_m={self.config.mcts.num_top_actions}, N={self.config.mcts.num_simulations}')
            self.config.save_path += tag

        self.obs_shape = copy.deepcopy(self.config.env.obs_shape)
        self.input_shape = copy.deepcopy(self.config.env.obs_shape)
        self.input_shape[0] *= self.config.env.n_stack
        self.action_space_size = self.config.env.action_space_size

        self._update = True
        
