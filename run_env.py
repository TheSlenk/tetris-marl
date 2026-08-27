import tetris_environment_par
from pettingzoo import make

env = make("parallel", "custom/tetris-v1", render_mode="LOG")
observations, infos = env.reset()
step = 0
total_reward = 0

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    total_reward += rewards['player_0']
    print(f'steps: {step}, reward: {total_reward}')
    step += 1
env.close()