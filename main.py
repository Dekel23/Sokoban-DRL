from game import SokobanGame
from model import Agent
import matplotlib.pyplot as plt
import pygame
import numpy as np

agent_hyperparameters = {
    'gamma': 0.99,
    'epsilon': 1.0,
    'batch_size': 10,
    'action_size': 4,
    'epsilon_min': 0.1,
    'epsilon_dec': 0.99,
    'input_size': 25,
    'lr': 0.01
}

# init agent
agent = Agent(**agent_hyperparameters)

# init environment (game)
pygame.init()
env = SokobanGame()

# training parameters
max_episodes = 100
max_steps = 100

successes_before_train = 5
successful_episodes = 0
continuous_successes_goal = 10
continuous_successes = 0
steps_per_episode = []

for episode in range(1, max_episodes + 1):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break

    if episode == 31:
        pass

    print(f"Episode: {episode}")
    env.reset_level()

    for step in range(1, max_steps + 1):
        state = np.array(env.map_info, dtype=np.float32)
        state = np.reshape(state, (agent_hyperparameters['input_size'],))
        action = agent.choose_action(observation=state)
        next_state, reward, done = env.step_action(action=action)
        next_state = np.array(next_state, dtype=np.float32)
        next_state = np.reshape(next_state, (agent_hyperparameters['input_size'],))
        agent.store_transition(state, action, reward, next_state, done)

        if successful_episodes >= successes_before_train:
            if step % agent.replay_rate == 0:
                agent.learn()

        if done:
            successful_episodes += 1
            continuous_successes += 1
            print(
                f"SOLVED! Episode {episode} Steps: {step} Epsilon {agent.epsilon:.4f}")
            break
        else:
            continuous_successes = 0

    steps_per_episode.append(step)

pygame.quit()


plt.plot(range(1, max_episodes + 1), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.show()