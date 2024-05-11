from game import SokobanGame
from model import Agent
import matplotlib.pyplot as plt

#import queue
import numpy as np
from collections import deque

def process_state(map_info, reshape=True):  # Take the game information and transform it into a stateS
    state = map_info[1:-1]  # Cut the frame
    state = [row[1:-1] for row in state]

    state = np.array(state, dtype=np.float32)  # transform to np.array in 1d
    if reshape:
        state = np.reshape(state, (agent_hyperparameters['input_size'],))
    
    return state

# Find all path lengths to target
def bfs(_map, target_x, target_y):
    rows, cols = _map.shape
    distances = np.full_like(_map, np.inf)
    distances[target_y, target_x] = 0
    queue = deque([(target_y, target_x)])

    while queue:
        current_node = queue.popleft()
        i, j = current_node

        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_i, new_j = i + di, j + dj
            if (0 <= new_i < rows and 0 <= new_j < cols and _map[new_i, new_j] != 1 and distances[new_i, new_j] > distances[current_node] + 1):
                distances[new_i, new_j] = distances[current_node] + 1
                queue.append((new_i, new_j))

    return distances


def calculate_reward(state, action, next_state, done, distances, alpha): # Cuclulate the reward of a step basted on queue of the next steps
    global loop_counter
    if (state == next_state).all(): # If the agent chose wasteful action
        return (reward_for_waste, 'waste')
    if done:  # If the agent finished the game
        return (reward_for_done, 'done')

    if check_loop(next_state, state_queue):
        loop_counter += 1
        # change_loop_rewards(state_queue)
        # fill_none(state_queue)
        # return (reward_for_loop, 'loop')

    #Reward for each step for inefficiency with regard to distance from 
    #distance_relation = alpha * (np.max(distances) / distances[env.cargo_y - 1,env.cargo_x - 1])
    #distance_cargo_keeper = alpha * np.max(distances) / (np.abs(env.cargo_y - env.y) + np.abs(env.cargo_y - env.y))
    return (reward_for_move, 'move')

def check_loop(state, state_queue):
    for s in state_queue:
        if (s is not None) and (state == s).all():
            return True
    
    return False

def change_loop_rewards(state_queue):
    loop_length = None
    for i in range(len(state_queue)):
        if state_queue[i] is None:
            loop_length = i
            break
    
    if loop_length is None:
        return
    
    for i in range(loop_length):    
        agent.replay_buffer[i][2] = reward_for_loop * (reward_for_loop_decay ** (i + 1))
    
    # for i in range(num):
    #     agent.replay_buffer[i][2] += reward_for_loop

def fill_none(state_queue):
    state_queue.clear()
    for _ in range(state_queue_length):
        state_queue.appendleft(None)

# init environment (game)
env = SokobanGame(level=61, graphics_enable=False)

# init agent
agent_hyperparameters = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_min': 0.1,
    'epsilon_decay': 0.9992,
    'input_size': (len(env.map_info) - 2) * (len(env.map_info[0]) - 2),
    'beta': 0.99
}
agent = Agent(**agent_hyperparameters)

# training parameters
max_episodes = 10000
max_steps = 30

successes_before_train = 10
successful_episodes = 0
continuous_successes_goal = 20
continuous_successes = 0
steps_per_episode = []
target_rate = 5

# base reward parameters
reward_for_stuck = 0
reward_for_waste = -2
reward_for_done = 10
reward_for_move = -0.5
reward_for_loop = -4
reward_for_loop_decay = 0.75

state_queue_length = 5
state_queue = deque(maxlen=state_queue_length)
loop_counter = 0
loops_per_episode = []

accumulated_reward_per_epsiode = []

init_state = process_state(env.map_info, reshape=False)
distances = bfs(init_state, env.target_x - 1, env.target_y - 1)
alpha = 0.2 # reward by distance multiplyer

for episode in range(1, max_episodes + 1):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break
    
    loop_counter = 0
    fill_none(state_queue)

    print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
    env.reset_level()

    accumulated_reward = 0

    for step in range(1, max_steps + 1):
        state = process_state(env.map_info)
        action = agent.choose_action(state=state)
        done = env.step_action(action=action)
        next_state = process_state(env.map_info)

        reward = calculate_reward(state, action, next_state, done, distances, alpha)
        if reward[1] != 'loop':
            accumulated_reward += reward[0]

        agent.store_replay(state, action, reward[0], next_state, done)

        state_queue.pop()
        state_queue.appendleft(next_state)

        if successful_episodes >= successes_before_train:
            agent.replay()
            agent.update_target_model()

        if done:
            successful_episodes += 1
            continuous_successes += 1
            print(f"SOLVED! Episode {episode} Steps: {step} Epsilon {agent.epsilon:.4f}")
            print(continuous_successes)
            steps_per_episode.append(step)
            agent.copy_to_prioritized_replay(step)
            break
    
    accumulated_reward_per_epsiode.append(accumulated_reward)
    
    #print(f'number of loops in episode {episode} is {loop_counter}')
    loops_per_episode.append(loop_counter)

    if not done:
        continuous_successes = 0
        steps_per_episode.append(max_steps)


# Plot the step per episode graph
plt.subplot(311)
plt.plot(range(1, len(steps_per_episode) + 1), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')

# Plot loops per episode graph
plt.subplot(312)
plt.plot(range(1, len(loops_per_episode) + 1), loops_per_episode)
plt.xlabel('Episode')
plt.ylabel('Loops')
plt.title('Loops per Episode')

# Plot loops per episode graph
plt.subplot(313)
plt.plot(range(1, len(accumulated_reward_per_epsiode) + 1), accumulated_reward_per_epsiode)
plt.xlabel('Episode')
plt.ylabel('Accumulated Reward')
plt.title('Accumulated Reward per Episode')

plt.tight_layout()
plt.show()

