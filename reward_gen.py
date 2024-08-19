from collections import deque
import zope.interface
import numpy as np


class RewardGen(zope.interface.Interface):
    def calculate_reward(*arg, **kargs):
        pass
    def reset(*arg, **kargs):
        pass


@zope.interface.implementer(RewardGen)
class MoveDoneLoop:
    def __init__(self) -> None:
        self.reward_for_waste = -2
        self.reward_for_done = 10
        self.reward_for_move = -0.5
        self.reward_for_loop = -4
        self.reward_for_loop_decay = 0.75
        self.state_queue_length = 5
        self.state_queue = deque(maxlen=self.state_queue_length)
        self.loop_counter = 0
        self.accumulated_reward = 0

    def calculate_reward(self, state, next_state, done, replay_buffer): # Calculate the reward of a step basted on queue of the next steps
        reward = self.reward_for_move
        if (state == next_state).all(): # If the agent chose wasteful action
            reward = self.reward_for_waste
            self.accumulated_reward += self.reward_for_waste
        elif done:  # If the agent finished the game
            reward = self.reward_for_done
            self.accumulated_reward += self.reward_for_done
        elif self._check_loop(next_state):
            self.loop_counter += 1
            self._change_loop_rewards(replay_buffer)
            self._fill_none()
            reward = self.reward_for_loop
            self.accumulated_reward += self.reward_for_move
        else:
            self.accumulated_reward += self.reward_for_move
            
        self.state_queue.pop()
        self.state_queue.appendleft(next_state)
        return reward
    
    def _check_loop(self, state):
        for s in self.state_queue:
            if (s is not None) and (state == s).all():
                return True
        
        return False

    def _change_loop_rewards(self, replay_buffer):
        loop_length = None
        for i in range(len(self.state_queue)):
            if self.state_queue[i] is None:
                loop_length = i
                break
        
        if loop_length is None:
            return
        
        for i in range(loop_length):    
            replay_buffer[i][2] = self.reward_for_loop * (self.reward_for_loop_decay ** (i + 1))

    def reset(self):
        self.loop_counter = 0
        self.accumulated_reward = 0
        self._fill_none()
    
    def _fill_none(self):
        self.state_queue.clear()
        for _ in range(self.state_queue.maxlen):
            self.state_queue.appendleft(None)

@zope.interface.implementer(RewardGen)
class DistanceMeasure:
    def __init__(self):
        self.loop_counter = 0
        self.accumulated_reward = 0

    def calculate_reward(self, grid, next_grid, done, buffer):
        if done:
            return 10
        
        # curr_pos = np.argwhere((grid == 6) | (grid == 7))[0]
        # curr_keeper_to_box, curr_box_pos = self.possible_path(grid, curr_pos[0], curr_pos[1], 4)        
        # curr_box_to_target, _ = self.possible_path(grid, curr_box_pos[0], curr_box_pos[1], 3)

        next_pos = np.argwhere((next_grid == 6) | (next_grid == 7))[0]
        next_keeper_to_box, next_box_pos = self.possible_path(next_grid, next_pos[0], next_pos[1], 4)        
        next_box_to_target, next_target_pos = self.possible_path(next_grid, next_box_pos[0], next_box_pos[1], 3)

        # curr_value = 1 / (curr_keeper_to_box + curr_box_to_target -1) + len(grid[np.where(grid == 5)])
        next_value = -3* (next_keeper_to_box + next_box_to_target -1) + len(next_grid[np.where(next_grid == 5)])
        reward = next_value

        if next_pos[0] == next_box_pos[0] and next_pos[0] == next_target_pos[0]:
            if (next_target_pos[1] < next_box_pos[1] < next_pos[1]) or (next_target_pos[1] > next_box_pos[1] > next_pos[1]):
                reward += 5

        
        if next_pos[1] == next_box_pos[1] and next_pos[1] == next_target_pos[1]:
            if (next_target_pos[0] < next_box_pos[0] < next_pos[0]) or (next_target_pos[0] > next_box_pos[0] > next_pos[0]):
                reward += 5

        if (grid == next_grid).all():
            reward -= 2

        return reward
    
    def reset(self):
        pass

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