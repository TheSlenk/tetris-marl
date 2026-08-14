import tetris_environment_par
from pettingzoo import make
env = make("parallel", "custom/tetris-v1", render_mode="human")

observations, infos = env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()