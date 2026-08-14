import json
import numpy as np
from display import DisplayLog
import sys

log_path = 'logs/latest.json'
if len(sys.argv) > 1:
    log_path = f"logs/{sys.argv[1]}"

log_display = DisplayLog()

with open(log_path, 'r') as f:
    json_str = '\n'.join(f.readlines())
    log_obj = json.loads(json_str)
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