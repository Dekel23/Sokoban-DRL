from game import SokobanGame
from model import Agent
import matplotlib.pyplot as plt
import pygame

pygame.init()

max_episodes = 1000
max_steps = 100
agent = Agent(gamma=0.99, epsilon=1.0, batch_size= 10, action_size= 4, epsilon_min= 0.1, epsilon_dec= 0.99, input_size=[5,5], lr=0.01)
env = SokobanGame()
successes_before_train = 5
successful_episodes = 0
continuous_successes_goal = 10
continuous_successes = 0
steps_per_episode = []

for e in range(max_episodes):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break

    print("Episode: %d" % (e+1))
    env.reset_level()

    for step in range(max_steps):
        state = env.map_info
        action=agent.choose_action(observation=state)
        next_state, reward, done = env.step_action(action=action)
        agent.store_transition(state, action, reward, next_state, done)

        if successful_episodes >= successes_before_train:
            if (step+1) % agent.replay_rate == 0:
                agent.learn()
        
        if done:
            successful_episodes += 1
            continuous_successes += 1
            print("SOLVED! Episode %d Steps: %d Epsilon %.4f" % (e+1, step+1, agent.epsilon))
            break
        else:
            continuous_successes = 0

    steps_per_episode.append(step+1)
        
plt.plot(range(1, max_episodes + 1), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.show()

pygame.quit()