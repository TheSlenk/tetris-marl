import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from tetris import Tetris
from tetris_game import NUM_DISTINCT_ACTIONS, ACTION_MAPPING, OBS_SHAPE

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)

    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {
        "render_mode": ["human"],
        "name": "tetris_v1"
    } 

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_1"]

        self.env = None
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
        return Discrete(NUM_DISTINCT_ACTIONS)

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )

            return

        if self.env.done:
            print("Gameover")
        else:
            print(self.env)
    
    def observe(self, agent):
        return np.array(self.observation_spaces[agent])
    
    def close(self):
        pass

    def reset(self, seed = None, options = None):
        