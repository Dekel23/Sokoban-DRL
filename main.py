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


def calculate_reward(state, action, next_state, done, stuck): # Cuclulate the reward of a step basted on queue of the next steps
    if (state == next_state).all(): # If the agent chose wasteful action
        return reward_for_waste
    if stuck:  # If the agent stuck the boxes
        return reward_for_stuck
    if done:  # If the agent finished the game
        return reward_for_done

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


# init agent
agent_hyperparameters = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'batch_size': 10,
    'action_size': 4,
    'epsilon_min': 0.1,
    'epsilon_dec': 0.99995,
    'input_size': 16,
    'lr': 0.0001,
    'lr_dec': 0.9999
}
agent = Agent(**agent_hyperparameters)

# init environment (game)
pygame.init()
env = SokobanGame()

# training parameters
max_episodes = 2000
max_random_steps = 20
max_learning_steps = 20

successes_before_train = 10
successful_episodes = 0
continuous_successes_goal = 10
continuous_successes = 0
steps_per_episode = []
ramdom_step_transition_rate = 0.05  # rate that non-special step is store in memory

# reward parameters
reward_for_stuck = 0
reward_for_waste = -3
reward_for_done = 10
reward_for_move = -0.5

for episode in range(1, max_episodes + 1):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break

    print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
    env.reset_level()

    for step in range(1, max_random_steps + 1):
        state = process_state(env.map_info)
        action = agent.choose_action(state=state)
        done = env.step_action(action=action)
        next_state = process_state(env.map_info)
        stuck = check_all_boxes(deepcopy(env.map_info))

        reward = calculate_reward(state, action, next_state, done, stuck)
        agent.store_transition(reward, state, action, next_state, done)

        if successful_episodes >= successes_before_train:
            # if step % agent.replay_rate == 0:
            agent.learn()
            max_random_steps = max_learning_steps

        if done:
            successful_episodes += 1
            continuous_successes += 1
            print(f"SOLVED! Episode {episode} Steps: {step} Epsilon {agent.epsilon:.4f}")
            print(continuous_successes)
            steps_per_episode.append(step)

            break

        elif stuck:
            continuous_successes = 0
            steps_per_episode.append(max_random_steps)

            break

    if not (done or stuck):
        continuous_successes = 0
        steps_per_episode.append(max_random_steps)


# Plot the step per episode graph
plt.plot(range(1, len(steps_per_episode) + 1), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.show()

pygame.quit()
