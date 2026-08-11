from tetris_enviorment_seq import env
from tetris_logger import Logger

logger = Logger()

game = env(render_mode="human")
game.reset(seed=42)

logger.log(game.envs)
for agent in game.agent_iter():
    observation, reward, termination, truncation, info = game.last()

    if termination or truncation:
        action = None
    else:
        # Sample a random legal action for the current agent.
        action = game.action_space(agent).sample()

    game.step(action)
    logger.log(game.envs)
    print(f"Agent: {agent}, reward: {reward}, terminated: {termination}")

logger.dump()
game.close()