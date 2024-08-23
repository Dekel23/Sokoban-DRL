from collections import deque
from abc import ABC, abstractmethod
import numpy as np


class RewardGenerator(ABC):
    def __init__(self, loop_size):
        super().__init__()

        self.loop_counter = 0
        self.loop_size = loop_size
        self.accumulated_reward = 0

    @ abstractmethod
    def calculate_reward(self, *arg, **kargs):
        pass

    def reset(self):
        self.loop_counter = 0
        self.accumulated_reward = 0

    def _check_loop(self, state, queue):
        for i in range(min(self.loop_size, len(queue))):
            s = queue[i][3]
            if (s is not None) and (np.reshape(state, (len(state) * len(state[0]),)) == s).all():
                self.loop_counter += 1
                return i
        
        return -1
    
    def calc_accumulated(func):

        def inner(self, *args, **kwargs):
            reward = func(self, *args, **kwargs)
            self.accumulated_reward += reward
            return reward
        return inner

class SimpleAndLoop(RewardGenerator): # for no checking loops set loop_size to 0
    def __init__(self, r_waste, r_done, r_move, r_loop, loop_decay, loop_size):
        super().__init__(loop_size=loop_size)

        self.reward_waste = r_waste
        self.reward_done = r_done
        self.reward_move = r_move
        self.reward_loop = r_loop
        self.loop_decay = loop_decay

    @ RewardGenerator.calc_accumulated
    def calculate_reward(self, state, next_state, done, replay_buffer):
        if done:
            return self.reward_done
        if (state == next_state).all():
            return self.reward_waste
        idx = self._check_loop(next_state, replay_buffer)
        if idx != -1:
            self._change_loop_rewards(idx, replay_buffer)
            return self.reward_loop
        return self.reward_move

    def _change_loop_rewards(self, idx, replay_buffer):
        for i in range(idx):    
            replay_buffer[i][2] = self.reward_loop * (self.loop_decay ** (i + 1))

class DistanceMeasure(RewardGenerator):
    def __init__(self, r_waste, r_done, r_move, r_loop, loop_decay, loop_size):
        super().__init__(loop_size=0)

    @ RewardGenerator.calc_accumulated
    def calculate_reward(self, grid, next_grid, done, buffer):
        if done:
            return 0
        
        if np.array_equal(grid, next_grid):
            return -10
        
        curr_pos = np.argwhere((grid == 6) | (grid == 7))[0]
        curr_keeper_to_box, curr_box_pos = self.possible_path(grid, curr_pos[0], curr_pos[1], 4)        
        curr_box_to_target, _ = self.possible_path(grid, curr_box_pos[0], curr_box_pos[1], 3)

        next_pos = np.argwhere((next_grid == 6) | (next_grid == 7))[0]
        next_keeper_to_box, next_box_pos = self.possible_path(next_grid, next_pos[0], next_pos[1], 4)        
        next_box_to_target, next_target_pos = self.possible_path(next_grid, next_box_pos[0], next_box_pos[1], 3)

        # # curr_value = 1 / (curr_keeper_to_box + curr_box_to_target -1) + len(grid[np.where(grid == 5)])
        # next_value = -3* (next_keeper_to_box + next_box_to_target -1) + len(next_grid[np.where(next_grid == 5)])
        # reward = next_value

        # if next_pos[0] == next_box_pos[0] and next_pos[0] == next_target_pos[0]:
        #     if (next_target_pos[1] < next_box_pos[1] < next_pos[1]) or (next_target_pos[1] > next_box_pos[1] > next_pos[1]):
        #         reward += 5

        
        # if next_pos[1] == next_box_pos[1] and next_pos[1] == next_target_pos[1]:
        #     if (next_target_pos[0] < next_box_pos[0] < next_pos[0]) or (next_target_pos[0] > next_box_pos[0] > next_pos[0]):
        #         reward += 5

        reward = 0

        if curr_keeper_to_box + curr_box_to_target < next_keeper_to_box + next_box_to_target:
            reward -= 10
        elif curr_keeper_to_box + curr_box_to_target > next_keeper_to_box + next_box_to_target:
            reward += 10
        # else:
        #     reward -= 0.1

        return reward

    def possible_path(self, grid, y, x, object_type):
        n, m = grid.shape

        grid = np.array(grid)
        grid[grid == 7] = 3

        q = deque()
        q.append((y, x))
    
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
    
        dis = [[-1 for _ in range(m)] for _ in range(n)]
    
        dis[y][x] = 0
    
        while q:
            p = q.popleft()

            if grid[p[0]][p[1]] == object_type:
                return dis[p[0]][p[1]], [p[0], p[1]]

            for i in range(4):
                x = p[1] + dx[i]
                y = p[0] + dy[i]

                if 0 <= y < n and 0 <= x < m and dis[y][x] == -1:
                    if grid[y][x] not in (1, 5):
                        if grid[y][x] == 4 and object_type == 3:
                            continue
                        
                        dis[y][x] = dis[p[0]][p[1]] + 1
                        q.append((y, x))

        return -1