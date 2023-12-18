from game import SokobanGame
from model import Agent
import matplotlib.pyplot as plt
import queue
import pygame
import numpy as np

agent_hyperparameters = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'batch_size': 10,
    'action_size': 4,
    'epsilon_min': 0.1,
    'epsilon_dec': 0.999,
    'input_size': 9,
    'lr': 0.01,
    'lr_dec': 0.9
}


def proccess_state(map_info):
    state = map_info[1:-1]
    state = [row[1:-1] for row in state]
    state = np.array(state, dtype=np.float32)
    state = np.reshape(state, (agent_hyperparameters['input_size'],))
    return state


def calculate_reward(queue):
    _, _, _, done, stuck = queue.queue[-1]
    prev_state, _, prev_next_state, _, _ = queue.queue[0]
    if (prev_state == prev_next_state).all():
        return -100
    if stuck:  # If the agent does something that doesnt contribute
        return -50 * (beta**step_queue.qsize())
    if done:
        return 20 * (beta**step_queue.qsize())

    return -1  # For each step subtruct 5 points for inefficiency


def check_all_boxes(board):
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 4: # if box
                if is_box_stuck(board, row, col):
                    return True
    
    return False


def is_box_stuck(board, row, col): # if 2 adjacent directions blocked the box is stuck
    board_copy = [row[:] for row in board]
    board_copy[row][col] = 0

    if row > len(board_copy) or col > len(board_copy[0]):
        return False

    down = (board_copy[row+1][col] in [0,1] or (board_copy[row+1][col] in [4,5] and is_box_stuck(board_copy,row+1,col)))
    up = (board_copy[row-1][col] in [0,1] or (board_copy[row-1][col] in [4,5] and is_box_stuck(board_copy,row-1,col)))
    left = (board_copy[row][col-1] in [0,1] or (board_copy[row][col-1] in [4,5] and is_box_stuck(board_copy,row,col-1)))
    right = (board_copy[row][col+1] in [0,1] or (board_copy[row][col+1] in [4,5] and is_box_stuck(board_copy,row,col+1)))

    if (left and up) or (up and right) or (right and down) or (down and left):
        return True
    
    return False


def empty_queue(step_queue):
    while not step_queue.empty():
        reward = calculate_reward(step_queue)
        agent.store_transition(reward, *step_queue.get())

def clone(lst):
    copy = []
    copy.extend(lst)
    return copy

# init agent
agent = Agent(**agent_hyperparameters)

# init environment (game)
pygame.init()
env = SokobanGame()

# training parameters
max_episodes = 400
max_steps = 100

successes_before_train = 5
successful_episodes = 0
continuous_successes_goal = 8
continuous_successes = 0
steps_per_episode = []

step_queue_size = 5
beta = 0.9 # reward decay

for episode in range(1, max_episodes + 1):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break
        
    print(f"Episode: {episode}")
    env.reset_level()
    step_queue = queue.Queue(maxsize=step_queue_size)

    for step in range(1, max_steps + 1):
        state = proccess_state(env.map_info)
        action = agent.choose_action(state=state)
        done = env.step_action(action=action)
        next_state = proccess_state(env.map_info)
        stuck = check_all_boxes(env.map_info)

        step_queue.put((state, action, next_state, done, stuck))

        #if (reward < 0 and np.random.random() < 0.1) or reward > 0:
        if stuck or done:
            empty_queue(step_queue)

        if step_queue.full():
            reward = calculate_reward(step_queue)
            agent.store_transition(reward, *step_queue.get())

        if successful_episodes >= successes_before_train:
            if step % agent.replay_rate == 0:
                agent.learn()

        if done:
            successful_episodes += 1
            continuous_successes += 1
            print(
                f"SOLVED! Episode {episode} Steps: {step} Epsilon {agent.epsilon:.4f}")
            
            steps_per_episode.append(step)

            break
        else:
            continuous_successes = 0
            if stuck:
                break


plt.plot(range(1, len(steps_per_episode) + 1), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.show()

pygame.quit()
