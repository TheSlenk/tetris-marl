import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from tetris_game import TetrisGame, NUM_DISTINCT_ACTIONS, OBS_SHAPE

def env(render_mode=None)
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)

    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper()

    env = wrappers.AssertOutOfBoundsWrapper(env)


    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "tetris_v1"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_1"]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {agent: Discrete(NUM_DISTINCT_ACTIONS) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Discrete(OBS_SHAPE) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(OBS_SHAPE)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(NUM_DISTINCT_ACTIONS, seed=self.np_random_seed)

    def render(self):

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 1:
            string = "Current state: Agent1: {}".format(
                self.state[self.agents[1]]
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        return np.array(self.observations[agent])

    def reset(self, seed = None, options = None):

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()