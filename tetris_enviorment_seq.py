import functools

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers

from tetris_game import TetrisGame, NUM_DISTINCT_ACTIONS, OBS_SHAPE

NUM_PLAYERS = 2

def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)

    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)

    env = wrappers.AssertOutOfBoundsWrapper(env)


    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "tetris_v1"}

    def __init__(self, render_mode=None, num_players: int = NUM_PLAYERS):
        self.num_players = num_players
        self.possible_agents: list[str] = ["player_" + str(r) for r in range(num_players)]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self._action_spaces = {agent: Discrete(NUM_DISTINCT_ACTIONS) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: Box(low=0, high=1, shape=(OBS_SHAPE,), dtype=np.int8) for agent in self.possible_agents
        }
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=1, shape=(OBS_SHAPE,), dtype=np.int8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(NUM_DISTINCT_ACTIONS, seed=self.np_random_seed)

    def render(self):

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        for index, (board, done) in enumerate(self.envs.full_game_display()):
            print(f"Player {index}{' (game over)' if done else ''}:")
            print(board)
        print()

    def observe(self, agent):
        return np.array(self.observations[agent], dtype=np.int8)

    def reset(self, seed = None, options = None):

        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.agents = self.possible_agents[:]
        self.envs = TetrisGame(num_players=len(self.agents))

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: self.envs.get_current_state(self.agent_name_mapping[agent]) for agent in self.agents}
        self.observations = {agent: self.envs.get_current_state(self.agent_name_mapping[agent]) for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_id = self.agent_name_mapping[agent]

        # The cumulative reward of the acting agent is reset to zero and
        # re-accumulated below (standard PettingZoo AEC pattern).
        self._cumulative_rewards[agent] = 0

        reward_before = self.envs.total_rewards[player_id]
        self.envs.apply_action(action)
        step_reward = self.envs.total_rewards[player_id] - reward_before

        self.num_moves += 1

        # Rewards are cleared each step; only the acting agent receives its reward.
        self.rewards = {a: 0 for a in self.agents}
        self.rewards[agent] = step_reward

        # The game terminates for every agent once any player tops out.
        terminated = self.envs.is_terminated()
        self.terminations = {a: terminated for a in self.agents}

        # Refresh observations for every agent.
        self.observations = {
            a: self.envs.get_current_state(self.agent_name_mapping[a])
            for a in self.agents
        }

        # Advance to the next agent in the sequential turn order.
        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def close(self):
        pass


# Test
if __name__ == "__main__":
    from pettingzoo.test import api_test

    api_test(env(), num_cycles=1_000, verbose_progress=True)
