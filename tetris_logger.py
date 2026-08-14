from tetris import Tetris
import json, time

LATEST_LOG_PATH = 'logs/latest.json'

class Logger:
    def __init__(self):
        self.timestep = 0
        self.logs = {}
        self.log_path = f'logs/log_{int(time.time())}.json'

    def log(self, envs: list[Tetris]):
        self.logs[self.timestep] = {f"agent_{i}": {
            "board": env.get_current_board().tolist(),
            "last_reward": env.last_reward,
            "total_reward": env.total_reward,
            "done": env.done
        } for i, env in enumerate(envs)}

        self.timestep += 1

    def dump(self):
        log_json_str = json.dumps(self.logs)

        with open(self.log_path, 'a') as f:
            f.write(log_json_str)

        with open(LATEST_LOG_PATH, 'w') as f:
            f.write(log_json_str)