import json
import numpy as np
from display import DisplayLog
import sys
from pathlib import Path

log_path = 'logs/latest.json'
if len(sys.argv) > 1:
    log_path = f"logs/{sys.argv[1]}"

log_display = DisplayLog()

log_obj = None
if log_path.endswith('.json'):
    with open(log_path, 'r') as f:
        json_str = '\n'.join(f.readlines())
        log_obj = json.loads(json_str)

else:
    logs = {
        p.name.split('_')[1]:
        json.loads('\n'.join(open(f'{log_path}/{p.name}', 'r').readlines()))
        for p in Path(log_path).iterdir() if p.is_file() and p.name.endswith('.json')
    }

    game_length = len(logs['0'].keys())

    log_obj = {
        str(step): {
            f'agent_{agent}': logs[agent][str(step)]
            for agent in logs.keys()
        }
        for step in range(game_length)
    }

if log_obj is None:
    raise Exception('Could not read logs')

steps = list(log_obj.keys())
cursor = steps[0]

while True:
    envs = log_obj[cursor]
    states = list(envs.values())
    clean_states = [(np.array(state['board']), state['done']) for state in states]
    agent_rewards = [(state['last_reward'], state['total_reward']) for state in states]

    move = log_display.show_step(f"{cursor}/{steps[-1]}", agent_rewards, clean_states)
    if move >= -1 and move <= 1:
        if move == 0:
            break
        cursor = int(cursor)
        cursor += move
        if not (str(cursor) in steps):
            cursor -= move
        
        cursor = str(cursor)