from collections import deque
from abc import ABC, abstractmethod
import numpy as np


class RewardGenerator(ABC):
    def __init__(self, loop_size = 0):
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

class Simple(RewardGenerator): # for no checking loops set loop_size to 0
    def __init__(self, r_waste, r_done, r_move, r_loop, loop_decay, loop_size=0):
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
    def __init__(self):
        super().__init__(loop_size=0)

    def calculate_reward(self, state, next_state, done, replay_buffer):
        if done:
            return 10
        
        # curr_pos = np.argwhere((state == 6) | (state == 7))[0]
        # curr_keeper_to_box, curr_box_pos = self.possible_path(state, curr_pos[0], curr_pos[1], 4)        
        # curr_box_to_target, _ = self.possible_path(state, curr_box_pos[0], curr_box_pos[1], 3)

        next_pos = np.argwhere((next_state == 6) | (next_state == 7))[0]
        next_keeper_to_box, next_box_pos = self.possible_path(next_state, next_pos[0], next_pos[1], 4)        
        next_box_to_target, next_target_pos = self.possible_path(next_state, next_box_pos[0], next_box_pos[1], 3)

        # curr_value = 1 / (curr_keeper_to_box + curr_box_to_target -1) + len(state[np.where(state == 5)])
        next_value = -3* (next_keeper_to_box + next_box_to_target -1) + len(next_state[np.where(next_state == 5)])
        reward = next_value

        if next_pos[0] == next_box_pos[0] and next_pos[0] == next_target_pos[0]:
            if (next_target_pos[1] < next_box_pos[1] < next_pos[1]) or (next_target_pos[1] > next_box_pos[1] > next_pos[1]):
                reward += 5

        
        if next_pos[1] == next_box_pos[1] and next_pos[1] == next_target_pos[1]:
            if (next_target_pos[0] < next_box_pos[0] < next_pos[0]) or (next_target_pos[0] > next_box_pos[0] > next_pos[0]):
                reward += 5

        if (state == next_state).all():
            reward -= 2

        return reward

    def possible_path(self, state, y, x, object_type):
        n, m = state.shape

        state = np.array(state)
        state[state == 7] = 3

        q = deque()
        q.append((y, x))
    
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
    
        dis = [[-1 for _ in range(m)] for _ in range(n)]
    
        dis[y][x] = 0
    
        while q:
            p = q.popleft()

            if state[p[0]][p[1]] == object_type:
                return dis[p[0]][p[1]], [p[0], p[1]]

            for i in range(4):
                x = p[1] + dx[i]
                y = p[0] + dy[i]

                if 0 <= y < n and 0 <= x < m and dis[y][x] == -1:
                    if state[y][x] not in (1, 5):
                        if state[y][x] == 4 and object_type == 3:
                            continue
                        
                        dis[y][x] = dis[p[0]][p[1]] + 1
                        q.append((y, x))

        return -1
    

class HotCold(RewardGenerator):
    def __init__(self, r_hot, r_cold, r_done, r_loop, loop_size=0):
        super().__init__(loop_size=loop_size)

        self.r_hot = r_hot
        self.r_cold = r_cold
        self.r_done = r_done
        self.r_loop = r_loop
    
    @ RewardGenerator.calc_accumulated
    def calculate_reward(self, state, next_state, done, replay_buffer):
        if done:
            return self.r_done
        
        # value is the distance so the smaller the better
        before_val = self.evaluate_state(state)
        after_val = self.evaluate_state(next_state)

        if before_val > after_val:
            return self.r_hot
        if self._check_loop(next_state, replay_buffer) != -1:
            return self.r_loop
        return self.r_cold
    
    def evaluate_state(self, state):
        state = np.array(state)
        kepper_y, kepper_x = np.argwhere((state == 6) | (state == 7))[0]
        state[state == 7] = 3

        _, box_y, box_x = self.path_to_type(state, kepper_y, kepper_x, 4)
        box_to_target, target_y, target_x = self.path_to_type(state, box_y, box_x, 3)

        kepper_to_box = np.inf
        sign_y = np.sign(box_y-target_y)
        if sign_y:
            kepper_to_box = min(kepper_to_box, self.path_to_pos(state, kepper_y, kepper_x, box_y+sign_y, box_x))
        sign_x = np.sign(box_x-target_x)
        if sign_x:
            kepper_to_box = min(kepper_to_box, self.path_to_pos(state, kepper_y, kepper_x, box_y, box_x+sign_x))

        return kepper_to_box + 4*box_to_target

    def path_to_type(self, state, start_y, start_x, end_type):
        n, m =  state.shape
        
        q = deque([(start_y, start_x)])

        dist_y = [-1, 0, 1, 0]
        dist_x = [0, -1, 0, 1]
        dist_mat = [[-1 for _ in range(m)] for _ in range(n)]
        dist_mat[start_y][start_x] = 0

        while q:
            cur_y, cur_x = q.pop()

            if state[cur_y][cur_x] == end_type:
                return dist_mat[cur_y][cur_x], cur_y, cur_x
        
            for i in range(4):
                nxt_y, nxt_x = cur_y+dist_y[i], cur_x+dist_x[i]

                if not(0 <= nxt_y < n and 0 <= nxt_x < m and dist_mat[nxt_y][nxt_x] == -1):
                    continue
                if state[nxt_y][nxt_x] in (0, 1, 5):
                    continue
                if end_type == 3 and state[nxt_y][nxt_x] == 4:
                    continue

                dist_mat[nxt_y][nxt_x] = dist_mat[cur_y][cur_x] + 1
                q.appendleft((nxt_y, nxt_x))

        return np.inf, -1, -1
    
    def path_to_pos(self, state, start_y, start_x, end_y, end_x):
        n, m =  state.shape
        
        q = deque([(start_y, start_x)])

        dist_y = [-1, 0, 1, 0]
        dist_x = [0, -1, 0, 1]
        dist_mat = [[-1 for _ in range(m)] for _ in range(n)]
        dist_mat[start_y][start_x] = 0

        while q:
            cur_y, cur_x = q.pop()
        
            for i in range(4):
                nxt_y, nxt_x = cur_y+dist_y[i], cur_x+dist_x[i]

                if nxt_y == end_y and nxt_x == end_x:
                    return dist_mat[cur_y][cur_x] + 1

                if not(0 <= nxt_y < n and 0 <= nxt_x < m and dist_mat[nxt_y][nxt_x] == -1):
                    continue
                if state[nxt_y][nxt_x] in (0, 1, 5):
                    continue
                if state[end_y][end_x] == 3 and state[nxt_y][nxt_x] == 4:
                    continue

                dist_mat[nxt_y][nxt_x] = dist_mat[cur_y][cur_x] + 1
                q.appendleft((nxt_y, nxt_x))

        return np.inf
