import tetris_environment_par
from pettingzoo import make
import csv

env = make("parallel", "custom/tetris-v1", render_mode="human")
observations, infos = env.reset()

with open('random_policy_rewards.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(('Episode', 'AVG_Reward'))

NUM_EPISODES = 1000
for episode in range(NUM_EPISODES):
    total_rewards = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        total_rewards += sum(rewards.values()) / len(rewards.values())

    with open('random_policy_rewards.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow((episode, total_rewards))
    observations, infos = env.reset()

env.close()