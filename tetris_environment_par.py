import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import time

from tetris import Tetris, NUM_DISTINCT_OBSTACLES
import itertools

# Metadata
NUM_PLAYERS = 1
WIDTH = 10
HEIGHT = 20
OBS_SHAPE = WIDTH * HEIGHT

ACTION_MAPPING = list(itertools.product(list(itertools.product(range(-WIDTH // 2, (WIDTH // 2) + 1), (0, 90, 180, 270))), range(NUM_DISTINCT_OBSTACLES)))
NUM_DISTINCT_ACTIONS = len(ACTION_MAPPING)
MAX_GAME_LEN = 50_000

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
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
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "tetris_v1"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(NUM_PLAYERS)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.render_mode = render_mode
        # RLlib queries action_space()/observation_space() before reset() is
        # ever called, so the RNG must exist from construction time.
        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.envs = None

    @functools.cache
    def observation_space(self, agent) -> gymnasium.Space:
        # get_raw_flat_board() returns a flat binary array of length OBS_SHAPE.
        # Use float32 so observations match the torch model's weight dtype.
        return Box(low=0, high=1, shape=(OBS_SHAPE,), dtype=np.float32)

    @functools.cache
    def action_space(self, agent) -> gymnasium.Space:
        return Discrete(NUM_DISTINCT_ACTIONS, seed=int(self.np_random_seed))

    def render(self):
        pass

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        logging_dir = f'{int(time.time())}'
        self.envs = {
            agent: Tetris(i, logging=self.render_mode == 'LOG', logging_dir=logging_dir) 
            for i, agent in enumerate(self.agents)
        }
        self.num_moves = 0
        # the observations should be numpy arrays even if there is only one value
        observations = {agent: self.envs[agent].get_raw_flat_board().astype(np.float32) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        rewards = {}
        terminations = {}

        for agent in self.agents:
            move, obstacle = ACTION_MAPPING[actions[agent]]
            _, _, reward, done = self.envs[agent].play(move, obstacle)
            rewards[agent] = reward
            terminations[agent] = done

        self.num_moves += 1
        env_truncation = self.num_moves >= MAX_GAME_LEN
        truncations = {agent: env_truncation for agent in self.agents}

        observations = {
            agent: self.envs[agent].get_raw_flat_board().astype(np.float32)
            for agent in self.agents
        }
        self.state = observations

        infos = {agent: {} for agent in self.agents}

        if env_truncation or any(terminations.values()):
            self.agents = []
            # Dump logs
            if self.envs is not None:
                for env in self.envs.values():
                    env.dump_log()

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

# Register env into pettingzoo
from pettingzoo import parallel_registry, register
register("parallel", "custom/tetris-v1", parallel_env)

# Confirm the environment is available in the registry
assert "custom/tetris-v1" in parallel_registry