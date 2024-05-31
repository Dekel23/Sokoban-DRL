from game import SokobanGame
from model import Agent
import matplotlib.pyplot as plt

import numpy as np
from reward_gen import *

def process_state(map_info, reshape=True):  # Take the game information and transform it into a stateS
    state = map_info[1:-1]  # Cut the frame
    state = [row[1:-1] for row in state]

    state = np.array(state, dtype=np.float32)  # transform to np.array in 1d
    if reshape:
        state = np.reshape(state, (agent_hyperparameters['input_size'],))
    
    return state

# init environment (game)
env = SokobanGame(level=61, graphics_enable=True)

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

reward_gen = Move_Done_Loop()

# training parameters
max_episodes = 1000
max_steps = 30

successes_before_train = 10
successful_episodes = 0
continuous_successes_goal = 20
continuous_successes = 0
steps_per_episode = []
target_rate = 5

loops_per_episode = []

for episode in range(1, max_episodes + 1):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break
    
    reward_gen.reset()

    print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
    env.reset_level()

    for step in range(1, max_steps + 1):
        state = process_state(env.map_info)
        action = agent.choose_action(state=state)
        done = env.step_action(action=action)
        next_state = process_state(env.map_info)

        reward = reward_gen.calculate_reward(state, next_state, done, agent.replay_buffer)
        agent.store_replay(state, action, reward, next_state, done)

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
    
    #print(f'number of loops in episode {episode} is {loop_counter}')
    loops_per_episode.append(reward_gen.loop_counter)

    if not done:
        continuous_successes = 0
        steps_per_episode.append(max_steps)


# Plot the step per episode graph
plt.subplot(211)
plt.plot(range(1, len(steps_per_episode) + 1), steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')

# Plot loops per episode graph
plt.subplot(212)
plt.plot(range(1, len(loops_per_episode) + 1), loops_per_episode)

plt.xlabel('Episode')
plt.ylabel('Loops')
plt.title('Loops per Episode')

plt.tight_layout()
plt.show()