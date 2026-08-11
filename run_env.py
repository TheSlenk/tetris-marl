from tetris_environment_par import TetrisEnvironment

env = TetrisEnvironment()
observations, infos = env.reset()

while env.agents:
    actions = { agent: env.action_space(agent).sample() for agent in env.agents }

    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f'Timestep: {env.timestep}, reward: {rewards}')

env.close()