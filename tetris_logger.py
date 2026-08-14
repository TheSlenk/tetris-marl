import json
from pathlib import Path

LATEST_LOG_PATH = 'logs/latest.json'

class Logger:
    def __init__(self, agent_id: int, logging_dir: str):
        self.timestep = 0
        self.logs = {}
        home_dir = ''
        if logging_dir != '.' and logging_dir != '':
            home_dir = f'{logging_dir}'
            if not Path(f'logs/{home_dir}').exists():
                Path(f'logs/{home_dir}').mkdir(parents=True, exist_ok=True)
        self.log_path = f'logs/{home_dir}/agent_{agent_id}_log.json'
        print(f'Logging onto "{self.log_path}" for agent_{agent_id}...')

    def log(self, board, last_reward, total_reward, done):
        self.logs[self.timestep] = {
            "board": board,
            "last_reward": last_reward,
            "total_reward": total_reward,
            "done": done
        } 

        self.timestep += 1

    def dump(self):
        log_json_str = json.dumps(self.logs)

        with open(self.log_path, 'w') as f:
            f.write(log_json_str)

        with open(LATEST_LOG_PATH, 'w') as f:
            f.write(log_json_str)