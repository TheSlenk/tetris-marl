from tetris_environment_par import TetrisEnvironment

env = TetrisEnvironment()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

env.close()