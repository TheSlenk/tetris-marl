import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from tetris_environment_par import parallel_env


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
        replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 60000,
            "alpha": 0.5,
            "beta": 0.5,
        }
    )
    .env_runners(num_env_runners=1)
)

algo = config.build()

for i in range(10):
    result = algo.train()
    env_runners = result.get("env_runners", {})
    print(
        f"iter {i}: "
        f"return_mean={env_runners.get('episode_return_mean')} "
        f"len_mean={env_runners.get('episode_len_mean')}"
    )

algo.stop()
ray.shutdown()