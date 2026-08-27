import matplotlib.pyplot as plt
import csv, math
from matplotlib.ticker import MultipleLocator

with open('dqn_training_rewards.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]

    iterations = []
    returns = []
    batch = 10

    for i in range(0, len(rows), batch):
        iterations.append(int(rows[i + batch - 1][0]))
        returns.append(sum([float(result) for _, result, _ in rows[i:i+batch]]) / batch)

plt.figure()
plt.plot(iterations, returns, label="DQN")

with open('dqn_v2_long_training_rewards.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:2991]

    iterations = []
    returns = []
    batch = 10

    for i in range(0, len(rows), batch):
        iterations.append(int(rows[i + batch - 1][0]))
        returns.append(sum([float(result) for _, result, _ in rows[i:i+batch]]) / batch)

plt.plot(iterations, returns, label="DQN_V2_SECOND_RUN")

with open('dqn_v2_training_rewards.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:931]

    iterations = []
    returns = []
    batch = 10

    for i in range(0, len(rows), batch):
        iterations.append(int(rows[i + batch - 1][0]))
        returns.append(sum([float(result) for _, result, _ in rows[i:i+batch]]) / batch)

plt.plot(iterations, returns, label="DQN_V2")

with open('random_policy_rewards.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)[1:]

    episodes = []
    rewards = []
    for row in rows:
        episodes.append(int(row[0]))
        rewards.append(float(row[1]))

mean = sum(rewards) / len(rewards)
print(f'Mean: {mean}')
standard_dev = math.sqrt(sum([(reward - mean) ** 2 for reward in rewards]) / len(rewards))

plt.plot(episodes, [mean] * len(episodes), label="Random Policy Mean")
plt.plot(episodes, [mean + standard_dev] * len(episodes), label="Random Policy Upper Bound")
plt.plot(episodes, [mean - standard_dev] * len(episodes), label="Random Policy Lower Bound")

plt.legend(loc="lower right")
plt.xlabel("Training iteration")
plt.ylabel("Mean episode return")
plt.title("Reward progression")
plt.xlim(0, 3000)
plt.gca().xaxis.set_major_locator(MultipleLocator(250))
plt.grid(True)
plt.savefig("training_rewards.png")
print("Saved training_rewards.csv and training_rewards.png")