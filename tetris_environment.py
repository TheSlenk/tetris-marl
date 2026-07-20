import functools
from copy import copy
from tetris import Tetris
from tetris_game import OBS_SHAPE, NUM_DISTINCT_ACTIONS, ACTION_MAPPING

from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv

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
    from tetris_environment import TetrisEnvironment
    from pettingzoo.test import parallel_api_test

    env = TetrisEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)