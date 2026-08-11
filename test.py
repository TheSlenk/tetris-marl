from tetris_game import TetrisGame, NUM_DISTINCT_ACTIONS
from tetris_logger import Logger
import random
env = TetrisGame(num_players=1)
logger = Logger()

logger.log(env)
while not env.is_terminated():
    env.apply_action(random.choice(range(NUM_DISTINCT_ACTIONS)))
    logger.log(env)

logger.dump()