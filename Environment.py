import numpy as np
import torch
from Graphics import *
from Constant import *
import random
import time


class Environment:
    def __init__(self):
        self.graphics = Graphics()
        self.board = np.zeros([BOARD_SIZE, BOARD_SIZE])   ### למחוק
        # self.snake = [(5,6),(5,5)]
        # self.mouse = [(5,11)]
        self.init_snake()
        self.init_mouse()
        self.score = 0
        # Bomb is a dict with keys: pos (r,c), spawn_time, explode_time
        # Only one active bomb at a time for now
        self.bomb = None
        # Minimum seconds before a new bomb can appear after previous explosion or spawn
        self._last_bomb_spawn_time = 0
        self._bomb_cooldown = BOMB_COOLDOWN
        self.action = 1


    def to_tensor(self, device=None):
        board = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=float)
        board[self.snake[0]] = 1
        board[tuple(zip(*self.snake[1:]))] = 2
        board[self.mouse[0]] = 3

        tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return tensor
   

    def get_head(self):
        return self.snake[0]    

    def is_about_to_eat(self, action):
        head = self.get_head()
        if action == ACTION_UP:
            new_head = (head[0] - 1, head[1])
        elif action == ACTION_DOWN:
            new_head = (head[0] + 1, head[1])
        elif action == ACTION_LEFT:
            new_head = (head[0], head[1] - 1)
        elif action == ACTION_RIGHT:
            new_head = (head[0], head[1] + 1)

        if new_head in self.mouse:
            return True
        return False


    def is_eat(self):
        head = self.get_head()
        if head in self.mouse[:]:  # If the snake eats a mouse
            self.score += 1
            self.mouse.pop(0)  # Remove the eaten mouse
            # Ensure a new mouse is always generated
            self.init_mouse()

            return True
        return False

    def spawn_bomb(self, min_distance=BOMB_MIN_DISTANCE, timer_seconds=BOMB_TIMER_SECONDS):
        # If there's already an active bomb, don't spawn another
        if self.bomb is not None:
            return False

        # Do not spawn too frequently
        now = time.time()
        if now - self._last_bomb_spawn_time < self._bomb_cooldown:
            return False

        # Attempt to find a valid location
        attempts = 0
        while attempts < 1000:
            attempts += 1
            r = random.randint(0, self.board.shape[0] - 1)
            c = random.randint(0, self.board.shape[1] - 1)
            point = (r, c)
            # not on the snake or mouse
            if point in self.snake or point in self.mouse:
                continue
            # ensure it's at least min_distance away from every snake segment (Chebyshev distance)
            too_close = False
            for s in self.snake:
                if max(abs(s[0] - r), abs(s[1] - c)) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue

            # found a valid spot
            self.bomb = {
                "pos": point,
                "spawn_time": now,
                "explode_time": now + timer_seconds,
                "timer_seconds": timer_seconds,
            }
            self._last_bomb_spawn_time = now
            return True

        return False

    def is_snake_in_explosion(self, bomb_pos):
        for s in self.snake:
            if max(abs(s[0] - bomb_pos[0]), abs(s[1] - bomb_pos[1])) <= 1:
                return True
        return False

    def tick_bomb(self, current_time=None):
        # Call this every environment tick; returns True if the bomb exploded and killed the snake (game over).
        # If the bomb exploded but did NOT hit the snake, it is cleared.
        if current_time is None:
            current_time = time.time()
        if self.bomb is None:
            return False
        if current_time >= self.bomb["explode_time"]:
            # Check for explosion hit
            if self.is_snake_in_explosion(self.bomb["pos"]):
                # Bomb exploded and hit snake -> game over
                return True
            else:
                # Bomb exploded, remove it
                self.bomb = None
                return False
        return False

    def is_self_hit (self):
        head = self.get_head()
        if len(self.snake) == 2:
            pass
        if head in self.snake[1:]:
            return True
        return False
    
    def check_collision_with_walls(self):
        head = self.get_head()
        if head[0] < 0 or head[0] >= self.graphics.size or head[1] < 0 or head[1] >= self.graphics.size:
            return False
        return True
    
    def is_board_full(self):
        # Check if the snake fills the entire board (game won condition)
        board_size = self.graphics.size
        total_cells = board_size * board_size
        snake_length = len(self.snake)
        return snake_length >= total_cells

    def move(self, action):

        head = self.get_head()
        if action == ACTION_UP:
            new_head = (head[0] - 1, head[1])
        elif action == ACTION_DOWN:
            new_head = (head[0] + 1, head[1])
        elif action == ACTION_LEFT:
            new_head = (head[0], head[1] - 1)
        elif action == ACTION_RIGHT:
            new_head = (head[0], head[1] + 1)
        else:
            return False

        # Wall collision
        if new_head[0] < 0 or new_head[0] >= self.graphics.size or new_head[1] < 0 or new_head[1] >= self.graphics.size:
            return True
        # Self collision
        if new_head in self.snake:
            return True

        # Normal move: insert new head, remove tail
        self.snake.insert(0, new_head)
        self.snake.pop(-1)
        return False

    def step2(self, action, current_time=None):
        # Handle a full game tick: eating, board-full check, moving, bomb spawn/removal, and bomb ticking.
        # Returns True if the game should continue, False if game over.
        
        if current_time is None:
            current_time = time.time()

        # Eating: if head is on a mouse, grow (insert duplicate head) and is_eat handles score/mouse respawn
        if self.is_eat():
            head = self.get_head()
            new_head = (head[0], head[1])
            self.snake.insert(0, new_head)

        # Board full -> win (end game)
        if self.is_board_full():
            return False

        # Move the snake (action may be None or falsy)
        if action:
            if not self.move(action):
                return False

        # Bomb logic: spawn bombs only when on screen 2 (score >= SECOND_SCREEN_SCORE)
        on_second_screen = (self.score >= SECOND_SCREEN_SCORE)
        if self.bomb is None and on_second_screen:
            if random.random() < BOMB_SPAWN_PROBABILITY:
                self.spawn_bomb()
        # If a bomb exists but we have left screen 2, remove it
        if self.bomb is not None and not on_second_screen:
            self.bomb = None

        # Check for bomb explosion and if it killed the snake
        if self.tick_bomb(current_time):
            return False

        return True
    
    def closer(self, action):
        reward = REWARD_FARTHER
        head_r, head_c = self.snake[0]
        mouse_r, mouse_c = self.mouse[0]

        #action Up
        if action == ACTION_UP and head_r > mouse_r:
            reward = REWARD_CLOSER
        #action Down
        elif action == ACTION_DOWN and head_r < mouse_r:
            reward = REWARD_CLOSER
        #action Left
        elif action == ACTION_LEFT and head_c > mouse_c:
            reward = REWARD_CLOSER
        #action Right
        elif action == ACTION_RIGHT and head_c < mouse_c:
            reward = REWARD_CLOSER
            
        return reward


    def step(self, action, current_time=None):

        if current_time is None:
            current_time = time.time()

        reward = self.closer(action)

        # 
        
        # Board full (win)
        if self.is_board_full():
            return REWARD_WIN, True

        # Eating
        if self.is_about_to_eat(action):
            reward = REWARD_EAT
            done = False
            self.snake.insert(0, self.mouse[0])
            self.init_mouse()
            self.score += 1
            
        else: # Move snake
            done = self.move(action)
            if done:
                return REWARD_LOSE, done
                        
        # if self.is_eat():
        #     head = self.get_head()
        #     new_head = (head[0], head[1])
        #     self.snake.insert(0, new_head)
        #     reward = REWARD_EAT
        
        # Bomb logic
        on_second_screen = (self.score >= SECOND_SCREEN_SCORE)

        if self.bomb is None and on_second_screen:
            if random.random() < BOMB_SPAWN_PROBABILITY:
                self.spawn_bomb()

        if self.bomb is not None and not on_second_screen:
            self.bomb = None

        # Bomb explosion
        if self.tick_bomb(current_time):
            return REWARD_LOSE, True

        return reward, False

    def move_env(self, action):
        reward, done = self.step(action)
        return reward, done
    
    def init_mouse(self):
        # row = random.randint(1,15)
        # col = random.randint(1,15)
        # self.mouse = [(row, col)]
        while True:
            random_point = (random.randint(MOUSE_INIT_MIN, MOUSE_INIT_MAX), random.randint(MOUSE_INIT_MIN, MOUSE_INIT_MAX))  # Keep within board limits
            if random_point not in self.snake:  # Ensure mouse doesn't spawn inside the snake
                self.mouse = [(random_point)]
                break  # Exit loop once a valid position is found
    
    def init_snake(self):
        head_row = random.randint(SNAKE_INIT_MIN, SNAKE_INIT_MAX)
        head_col = random.randint(SNAKE_INIT_MIN, SNAKE_INIT_MAX)
        body_dir = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        body_row = head_row + body_dir[0]
        body_col = head_col + body_dir[1]
        self.snake = [(head_row, head_col), (body_row, body_col)]
    
    def reset(self):
        self.board = np.zeros([BOARD_SIZE, BOARD_SIZE])
        # self.snake = [(5,6),(5,5)]
        # self.mouse = [(5,11)]
        self.init_snake()
        self.init_mouse()
        self.score = 0
        self.bomb = None
        # Minimum seconds before a new bomb can appear after previous explosion or spawn
        self._last_bomb_spawn_time = 0
        self._bomb_cooldown = BOMB_COOLDOWN
        
            

    
            
    