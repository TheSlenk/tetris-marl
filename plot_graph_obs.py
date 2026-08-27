import csv
import matplotlib.pyplot as plt
from tetris_environment_par import NUM_DISTINCT_OBSTACLES
from tetris import Tetris

data = {}
with open('obs_usage.log', 'r') as f:
    reader = csv.reader(f)
    raw_data = list(reader)[1:]
    clean_data = [int(entry[0]) for entry in raw_data]
    total = 0
    for d in clean_data:
        if d not in data.keys():
            data[d] = 0
        data[d] += 1
        total += 1

    print(total)
    for i in range(NUM_DISTINCT_OBSTACLES):
        data[i] = (data[i] / total) * 100

env = Tetris('test')
obs = [env.new_obstacle(obs) for obs in data.keys()]
plt.figure(figsize=(max(8, len(obs) * 0.6), 6))
plt.bar([o.name for o in obs], [data[o.id] for o in obs])
plt.xlabel('Tetris Obstacle Types')
plt.ylabel('Usage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('obs_usage_bar_graph.png')