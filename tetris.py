import random
import numpy as np
from enum import Enum
from collections import deque

class BlockColor(Enum):
    BLUE = 1
    PINK = 2
    YELLOW = 3
    RED = 4
    GREEN = 5
    CYAN = 6
    ORANGE = 7

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    ROTATE = 3

class Block:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

class Obstacle:
    def __init__(self, id: int, x: int, y: int, color_value: BlockColor):
        self.name = self.__class__.__name__
        self.id = id
        self.x = x
        self.y = y
        self.color_value = color_value
        self.blocks: list[Block] = []

    def rotate(self, amount: int = 90):
        for _ in range((amount % 360) // 90):
            for block in self.blocks:
                x, y = block.x, block.y
                block.x, block.y = -y, x

    def block_positions(self) -> list[tuple[int, int]]:
        positions = []
        for block in self.blocks:
            positions.append((self.x + block.x, self.y + block.y))
        return positions

class LineObstacle(Obstacle):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, x, y, BlockColor.CYAN)
        self.blocks.append(Block(-1, 0))
        self.blocks.append(Block(0, 0))
        self.blocks.append(Block(1, 0))
        self.blocks.append(Block(2, 0))

class TObstacle(Obstacle):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, x, y, BlockColor.PINK)
        self.blocks.append(Block(-1, 0))
        self.blocks.append(Block(0, 0))
        self.blocks.append(Block(1, 0))
        self.blocks.append(Block(0, -1))

class BoxObstacle(Obstacle):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, x, y, BlockColor.YELLOW)
        self.blocks.append(Block(0, 0))
        self.blocks.append(Block(0, -1))
        self.blocks.append(Block(1, 0))
        self.blocks.append(Block(1, -1))

class LObstacle(Obstacle):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, x, y, BlockColor.BLUE)
        self.blocks.append(Block(0, 0))
        self.blocks.append(Block(0, -1))
        self.blocks.append(Block(0, -2))
        self.blocks.append(Block(1, 0))

class JObstacle(Obstacle):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, x, y, BlockColor.RED)
        self.blocks.append(Block(0, 0))
        self.blocks.append(Block(0, -1))
        self.blocks.append(Block(0, -2))
        self.blocks.append(Block(-1, 0))

class ZObstacle(Obstacle):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, x, y, BlockColor.GREEN)
        self.blocks.append(Block(0, 0))
        self.blocks.append(Block(0, -1))
        self.blocks.append(Block(-1, -1))
        self.blocks.append(Block(1, 0))

class SObstacle(Obstacle):
    def __init__(self, id: int, x: int, y: int):
        super().__init__(id, x, y, BlockColor.ORANGE)
        self.blocks.append(Block(0, 0))
        self.blocks.append(Block(-1, 0))
        self.blocks.append(Block(0, -1))
        self.blocks.append(Block(1, -1))

class Tetris:
    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width))
        self.done = False
        self.step = 0

        self.current_obstacle: Obstacle | None = None
        self.obstacle_queue = deque()

        self.next_states: dict = None

        self.next_obstacle()

    def next_obstacle(self, peice_id: int = None):
        if peice_id is not None or len(self.obstacle_queue) == 0:
            random_obstacle = peice_id if peice_id is not None else random.choice(range(7))
            self.current_obstacle = self.new_obstacle(random_obstacle)
        else:
            self.current_obstacle = self.obstacle_queue.popleft()
    
    def new_obstacle(self, id: int) -> Obstacle:
        x, y = self.width // 2 - 1, 2
        obs = None

        match id:
            case 0:
                obs = TObstacle(id, x, y)
            case 1:
                obs = LineObstacle(id, x, y)
            case 2:
                obs = BoxObstacle(id, x, y)
            case 3:
                obs = LObstacle(id, x, y)
            case 4:
                obs = JObstacle(id, x, y)
            case 5:
                obs = ZObstacle(id, x, y)
            case 6:
                obs = SObstacle(id, x, y)
        
        return obs

    def get_board_block_value(self, x: int, y: int) -> int:
        return int(self.board[y, x])

    def set_board_block_value(self, x: int, y: int, value):
        self.board[y, x] = value

    def move_obstacle(self, direction: tuple[int, int] = (0, 0), rotation: int = 0, validate: bool = True) -> bool:
        new_x, new_y = self.current_obstacle.x + direction[0], self.current_obstacle.y + direction[1]
        old_x, old_y = self.current_obstacle.x, self.current_obstacle.y
        rotation_to_old_rotation = 360 - (rotation % 360)
        self.current_obstacle.x, self.current_obstacle.y = new_x, new_y
        self.current_obstacle.rotate(rotation)

        if validate:
            valid = self.validate_obstacle()
            if not valid:
                self.current_obstacle.x, self.current_obstacle.y = old_x, old_y
                self.current_obstacle.rotate(rotation_to_old_rotation)

            return valid
        return True

    def validate_obstacle(self) -> bool:
        blocks = self.current_obstacle.block_positions()
        for block in blocks:
            x, y = block
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return False
            if self.get_board_block_value(x, y) > 0:
                return False

        return True

    def apply_action(self, action: Action):
        x, y = self.current_obstacle.x, self.current_obstacle.y

        match action:
            case Action.LEFT:
                self.move_obstacle(direction=(-1, 0))
            case Action.RIGHT:
                self.move_obstacle(direction=(1, 0))
            case Action.DOWN:
                self.move_obstacle(direction=(0, 1))
            case Action.ROTATE:
                self.move_obstacle(rotation=90)
    
    def apply_state(self, state: tuple[int, int]) -> bool:
        x, rotation = state
        applied = self.move_obstacle(direction=(x, 0), rotation=rotation)

        if applied:
            self.drop_obstacle()
            self.settle_obstacle()

        return applied
    
    def get_next_states(self) -> dict:

        if self.next_states is not None:
            return self.next_states
        
        return self._gen_next_states()

    def _gen_next_states(self) -> dict:
        
        if self.current_obstacle is None:
            return {}
        
        states = {}

        peice_name = self.current_obstacle.name
        rotations = (0, 90, 180, 270)
        if peice_name == "BoxObstacle":
            rotations = (0,)
        elif peice_name == "LineObstacle":
            rotations = (0, 90)
        
        for rotation in rotations:
            self.move_obstacle(rotation=rotation, validate=False)
            block_x_pos = [block[0] for block in self.current_obstacle.block_positions()]
            min_x = min(block_x_pos)
            max_x = max(block_x_pos)
            self.reset_obstacle()

            for x in range(-min_x, self.width - max_x):
                valid = self.move_obstacle(direction=(x, 0), rotation=rotation, validate=True)
                if valid:
                    self.drop_obstacle()
                    states[(x, rotation)] = self.get_current_board()
                self.reset_obstacle()
        
        return states

    def reset_obstacle(self):
        peice_id = self.current_obstacle.id
        self.next_obstacle(peice_id)

    def drop_obstacle(self):
        while self.move_obstacle(direction=(0, 1)):
            continue
    
    def settle_obstacle(self):
        self.board = self.get_current_board()
        self.next_obstacle()

    def play(self, next_state: tuple[int, int]):
        reward = 0

        if not self.done:
            applied = self.apply_state(next_state)
            if not applied:
                valid_next_states = list(self.get_next_states().keys())
                if len(valid_next_states) <= 0:
                    raise Exception(f'ERR: Invalid action: {next_state}, valid action set: {valid_next_states}')
                valid_next_state = self._find_closest_state(next_state, valid_next_states)
                applied = self.apply_state(valid_next_state)
                if not applied:
                    raise Exception(f'ERR: bad impementation within _find_closest_state()')
                
            cleared = self.clear_rows()
            reward = 2 ** cleared
            self.done = self.is_game_over()
            self.step += 1
            self.next_states = self._gen_next_states()

        return self.get_current_board(), self.get_next_states(), reward, self.done

    def _find_closest_state(self, state: tuple, state_list: list[tuple]) -> tuple:
        same_rot = [s for s in state_list if s[1] == state[1]]
        if len(same_rot) > 0:
            closest = np.argmin(np.array([np.abs(state[0] - s[0]) for s in same_rot]))
            return same_rot[closest]
        
        closest = np.argmin(np.array([np.abs(state[0] - s[0]) for s in state_list]))
        return state_list[closest]

    # Test function
    def fill_row(self, row):
        self.board[row, :] = 1
    
    def clear_rows(self) -> int:
        cleared = 0
        for row in range(3, self.height):
            is_row_clear = all([col > 0 for col in self.board[row]])
            if is_row_clear:
                for above in range(row, 0, -1):
                    self.board[above, :] = self.board[above - 1, :]
                    self.board[above - 1, :] = 0
                cleared += 1
        
        return cleared
    
    def is_game_over(self) -> bool:
        for row in range(0, 3):
            if any([col > 0 for col in self.board[row]]):
                self.current_obstacle = None
                return True
        
        return False

    def print_board(self):
        print(self)

    def get_current_board(self):
        if self.current_obstacle is None:
            return self.board
        
        current_obstacle_position = self.current_obstacle.block_positions()
        copy_board = self.board.copy()
        for (x, y) in current_obstacle_position:
            copy_board[y, x] = self.current_obstacle.color_value.value
        
        return copy_board

    def get_flat_board(self):
        board = self.get_current_board()
        board = board.flatten()
        return np.array(board)
    
    def get_raw_flat_board(self):
        board = self.get_current_board()
        board = board.flatten()
        return np.array([1 if e > 0 else 0 for e in board])

    def __str__(self):
        return str(self.get_current_board())
