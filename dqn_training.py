import csv

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from tetris_environment_par import parallel_env

NUM_EPOCHS = 100

# RLlib has its own registry (separate from PettingZoo's). Wrap the PettingZoo
# ParallelEnv in ParallelPettingZooEnv so RLlib sees it as a MultiAgentEnv.
def env_creator(config):
    return ParallelPettingZooEnv(parallel_env())

register_env("tetris-v1", env_creator)

config = (
    DQNConfig()
    .environment("tetris-v1")
    .framework("torch")
    # Every agent ("player_0", ...) shares a single policy.
    .multi_agent(
        policies={"shared_policy"},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
    )
    .training(
        # This env is multi-agent (PettingZoo), so episodes are MultiAgentEpisodes.
        # The default single-agent EpisodeReplayBuffer can't sample them, so use the
        # multi-agent (prioritized) episode buffer.
        replay_buffer_config={
            "type": "MultiAgentPrioritizedEpisodeReplayBuffer",
            "capacity": 60000,
            "alpha": 0.5,
            "beta": 0.5,
        }
    )
    .env_runners(num_env_runners=1)
)

algo = config.build()

history = []
for i in range(NUM_EPOCHS):
    result = algo.train()
    env_runners = result.get("env_runners", {})
    return_mean = env_runners.get("episode_return_mean")
    len_mean = env_runners.get("episode_len_mean")
    history.append((i, return_mean, len_mean))
    print(f"iter {i}: return_mean={return_mean} len_mean={len_mean}")

algo.stop()
ray.shutdown()

# Save the reward progression to CSV for plotting later.
with open("training_rewards.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "episode_return_mean", "episode_len_mean"])
    writer.writerows(history)

# Save a reward-progression graph (skips silently if matplotlib isn't installed).
try:
    import matplotlib.pyplot as plt

    iterations = [row[0] for row in history]
    returns = [row[1] for row in history]
    plt.figure()
    plt.plot(iterations, returns)
    plt.xlabel("Training iteration")
    plt.ylabel("Mean episode return")
    plt.title("Reward progression")
    plt.grid(True)
    plt.savefig("training_rewards.png")
    print("Saved training_rewards.csv and training_rewards.png")
except ImportError:
    print("Saved training_rewards.csv (install matplotlib to also get the graph)")