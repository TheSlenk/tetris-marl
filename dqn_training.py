import csv
from pathlib import Path

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from tetris_environment_par import NUM_PLAYERS, parallel_env

NUM_EPOCHS = 5_000
AGENT_IDS = tuple(f"player_{i}" for i in range(NUM_PLAYERS))
CHECKPOINT_DIR = Path("checkpoints").resolve()
CHECKPOINT_INTERVAL = 250
RESTORE_CHECKPOINT = 'checkpoints/iteration_3000'

# RLlib has its own registry (separate from PettingZoo's). Wrap the PettingZoo
# ParallelEnv in ParallelPettingZooEnv so RLlib sees it as a MultiAgentEnv.
def env_creator(config):
    return ParallelPettingZooEnv(parallel_env(render_mode='LOG'))

register_env("tetris-v1", env_creator)

config = (
    DQNConfig()
    .environment("tetris-v1")
    .framework("torch")
    # Each agent has its own policy and therefore its own Q-network and replay
    # experience. The environment already returns each agent's own reward.
    .multi_agent(
        policies=AGENT_IDS,
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        policies_to_train=list(AGENT_IDS),
        count_steps_by="agent_steps",
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
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[512, 512, 256, 128],
            fcnet_activation="relu",
        )
    )
    .env_runners(num_env_runners=1)
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
algo = config.build_algo()

if RESTORE_CHECKPOINT is not None:
    algo.restore(RESTORE_CHECKPOINT)
    print(f"Restored checkpoint: {RESTORE_CHECKPOINT}")

reward_files = {}
for agent_id in AGENT_IDS:
    reward_file = open(
        f"training_rewards_{agent_id}.csv", "w", newline=""
    )
    csv.writer(reward_file).writerow(
        ["iteration", "episode_return_mean", "episode_len_mean"]
    )
    reward_files[agent_id] = reward_file
    
for i in range(NUM_EPOCHS):
    result = algo.train()
    env_runners = result.get("env_runners", {})
    agent_returns = env_runners.get("agent_episode_returns_mean", {})
    len_mean = env_runners.get("episode_len_mean")
    agent_return_values = {}
    for agent_id, reward_file in reward_files.items():
        agent_return = agent_returns.get(agent_id)
        csv.writer(reward_file).writerow((i, agent_return, len_mean))
        reward_file.flush()
        agent_return_values[agent_id] = agent_return
    print(f"iter {i}: agent_returns={agent_return_values} len_mean={len_mean}")

    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_path = algo.save_to_path(
            CHECKPOINT_DIR / f"iteration_{i + 1}"
        )
        print(f"Saved checkpoint: {checkpoint_path}")

checkpoint_path = algo.save_to_path(CHECKPOINT_DIR / "latest")
print(f"Saved final checkpoint: {checkpoint_path}")

for reward_file in reward_files.values():
    reward_file.close()

algo.stop()
ray.shutdown()