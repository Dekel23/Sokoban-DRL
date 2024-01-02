from game import SokobanGame
from model import Agent
import matplotlib.pyplot as plt
import queue
import pygame
import numpy as np
from copy import deepcopy
import torch


def process_state(map_info):  # Take the game information and transform it into a stateS
    state = map_info[1:-1]  # Cut the frame
    state = [row[1:-1] for row in state]

    state = np.array(state, dtype=np.float32)  # transform to np.array in 1d
    state = np.reshape(state, (agent_hyperparameters['input_size'],))
    return state


def calculate_reward(state, action, next_state, done, stuck):  # Calculate the reward of a step basted on queue of the next steps
    if (state == next_state).all():  # If the agent chose wasteful action
        return reward_for_waste
    if stuck:  # If the agent stuck the boxes
        return reward_for_stuck * (stuck_reward_dacey**step_queue.qsize())
    if done:  # If the agent finished the game
        return reward_for_done * (done_reward_dacey**step_queue.qsize())

    return reward_for_move  # Reward for each step for inefficiency


def check_all_boxes(board):
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 4:  # For any box not on target
                if is_box_stuck(board, row, col):
                    return True
    return False


def is_box_stuck(board, row, col):  # if 2 adjacent directions blocked the box is stuck
    board[row][col] = 0

    if row > len(board) or col > len(board[0]):
        return False

    down = (board[row + 1][col] in [0, 1] or (board[row + 1][col] in [4, 5] and is_box_stuck(board, row + 1, col)))
    up = (board[row - 1][col] in [0, 1] or (board[row - 1][col] in [4, 5] and is_box_stuck(board, row - 1, col)))
    left = (board[row][col - 1] in [0, 1] or (board[row][col - 1] in [4, 5] and is_box_stuck(board, row, col - 1)))
    right = (board[row][col + 1] in [0, 1] or (board[row][col + 1] in [4, 5] and is_box_stuck(board, row, col + 1)))

    if (left and up) or (up and right) or (right and down) or (down and left):
        return True

    return False


def empty_queue(step_queue):  # If the agent done or stuck we reward and store the latest steps
    while not step_queue.empty():
        reward = calculate_reward(step_queue)
        agent.store_transition(reward, *step_queue.get())


# init agent
agent_hyperparameters = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'batch_size': 10,
    'action_size': 4,
    'epsilon_min': 0.15,
    'epsilon_dec': 0.99,
    'input_size': 9,
    'lr': 0.0001,
    'lr_dec': 0.99999
}
agent = Agent(**agent_hyperparameters)

# init environment (game)
pygame.init()
env = SokobanGame()

# training parameters
max_episodes = 2000
max_steps = 5

successes_before_train = 5
successful_episodes = 0
continuous_successes_goal = 10
continuous_successes = 0
steps_per_episode = []
ramdom_step_transition_rate = 0.01  # rate that non-special step is store in memory

# reward parameters
step_queue_size = 4
done_reward_dacey = 0.5
stuck_reward_dacey = 0.5

reward_for_stuck = 0
reward_for_waste = -5
reward_for_done = 5
reward_for_move = -1

for episode in range(1, max_episodes + 1):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break

    print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
    env.reset_level()
    step_queue = queue.Queue(maxsize=step_queue_size)

    for step in range(1, max_steps + 1):
        state = process_state(env.map_info)
        action = agent.choose_action(state=state)
        done = env.step_action(action=action)
        next_state = process_state(env.map_info)
        stuck = check_all_boxes(deepcopy(env.map_info))

        reward = calculate_reward(state, action, next_state, done, stuck)
        # if done or stuck:
        #     agent.store_transition(reward, state, action, next_state, done)
        # elif np.random.random() <= ramdom_step_transition_rate:
        agent.store_transition(reward, state, action, next_state, done)

        if successful_episodes >= successes_before_train:
            # if step % agent.replay_rate == 0:
            agent.learn()

        if done:
            successful_episodes += 1
            continuous_successes += 1
            print(f"SOLVED! Episode {episode} Steps: {step} Epsilon {agent.epsilon:.4f}")

            steps_per_episode.append(step)

            break
        else:
            continuous_successes = 0
            if stuck:
                steps_per_episode.append(max_steps)
                break

# Plot the step per episode graph
plt.plot(range(1, len(steps_per_episode) + 1), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.show()

pygame.quit()
