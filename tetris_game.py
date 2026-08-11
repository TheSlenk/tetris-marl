from tetris import Tetris
import itertools
import numpy as np

WIDTH = 10
HEIGHT = 20
OBS_SHAPE = WIDTH * HEIGHT

ACTION_MAPPING = list(itertools.product(range(-WIDTH // 2, (WIDTH // 2) + 1), (0, 90, 180, 270)))
NUM_DISTINCT_ACTIONS = len(ACTION_MAPPING)
MAX_GAME_LEN = 50_000

class TetrisGame:
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.envs = [Tetris(height=HEIGHT, width=WIDTH) for _ in range(num_players)]
        self.current_player_idx = 0
        self.total_rewards = [0.0] * num_players
    
    def pass_turn(self):
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
    
    def get_current_state(self, player_id: int | None = None):
        player_id = player_id if player_id is not None else self.current_player_idx
        return self.envs[player_id].get_raw_flat_board()
    
    def legal_actions(self, player_id: int | None = None) -> list[int]:
        player_id = player_id if player_id is not None else self.current_player_idx
        return [ACTION_MAPPING.index(key) for key in self.envs[player_id].get_next_states().keys()]
    
    def apply_action(self, action_id: int):
        _, _, step_reward, _ = self.envs[self.current_player_idx].play(ACTION_MAPPING[action_id])
        self.total_rewards[self.current_player_idx] += step_reward

        self.pass_turn()
    
    def is_terminated(self):
        return any([env.done for env in self.envs])

    def rewards(self):
        return tuple(self.total_rewards)
    
    # Output formated to use in Display class
    def full_game_display(self):
        return [(env.get_current_board(), env.done) for env in self.envs]
