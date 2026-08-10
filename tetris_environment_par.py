import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from copy import copy
from tetris import Tetris
from tetris_game import OBS_SHAPE, NUM_DISTINCT_ACTIONS, ACTION_MAPPING

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = TetrisEnvironment(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class TetrisEnvironment(ParallelEnv):
    metadata = {
        "name": "tetris_environment_v0"
    }

    def __init__(self):
        self.timestep = None
        self.env = None
        self.possible_agents = [ "agent_0" ]

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.env = Tetris()

        observations = { 
            a: self.env.get_raw_flat_board()
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        agent_action = actions["agent_0"]

        _, _, reward, done = self.env.play(ACTION_MAPPING[agent_action])

        terminations = {a: done for a in self.agents}
        rewards = {a: reward for a in self.agents}
        truncations = {a: False for a in self.agents}

        self.timestep += 1
        
        observations = {
            a: self.env.get_raw_flat_board()
            for a in self.agents
        }

        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        self.env.print_board()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Discrete(OBS_SHAPE)
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(NUM_DISTINCT_ACTIONS)
    
# Test
if __name__ == "__main__":
    from tetris_environment_par import TetrisEnvironment
    from pettingzoo.test import parallel_api_test

    env = TetrisEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)