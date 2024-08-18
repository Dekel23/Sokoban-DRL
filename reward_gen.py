from collections import deque
import zope.interface

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
        # elif self._check_loop(next_state):
        #     self.loop_counter += 1
        #     self._change_loop_rewards(replay_buffer)
        #     self._fill_none()
        #     reward = self.reward_for_loop
        #     self.accumulated_reward += self.reward_for_move
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
class BFS:
    def calculate_reward(self, _map, target_x, target_y):
        # Find all path lengths to target
        # rows, cols = _map.shape
        # distances = np.full_like(_map, np.inf)
        # distances[target_y, target_x] = 0
        # queue = deque([(target_y, target_x)])

        # while queue:
        #     current_node = queue.popleft()
        #     i, j = current_node

        #     for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        #         new_i, new_j = i + di, j + dj
        #         if (0 <= new_i < rows and 0 <= new_j < cols and _map[new_i, new_j] != 1 and distances[new_i, new_j] > distances[current_node] + 1):
        #             distances[new_i, new_j] = distances[current_node] + 1
        #             queue.append((new_i, new_j))

        # return distances
        pass

    def reset(self):
        pass
