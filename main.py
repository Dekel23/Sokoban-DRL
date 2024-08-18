from game import SokobanGame
from agent import Agent
import matplotlib.pyplot as plt

import numpy as np
from reward_gen import *
from one_step_agent import KStepAgent

# init environment (game)
env = SokobanGame(level=61, graphics_enable=False)

row = len(env.map_info)
col = len(env.map_info[0])

# init agent
agent_hyperparameters = {
    'gamma': 0.995,
    'epsilon': 1.0,
    'epsilon_min': 0.1,
    'epsilon_decay': 0.995,
    'input_size': (row - 2) * (col - 2),
    'beta': 0.99
}

agent = Agent(**agent_hyperparameters)
two_step_agent = KStepAgent(agent_hyperparameters, row - 2, col - 2)

reward_gen = MoveDoneLoop()

# training parameters
max_episodes = 1000
max_steps = 30

successes_before_train = 10
successful_episodes = 0
continuous_successes_goal = 20
continuous_successes = 0
steps_per_episode = []

loops_per_episode = []
accumulated_reward_per_epsiode = []

save_rate = 50

for episode in range(1, max_episodes + 1):
    if continuous_successes >= continuous_successes_goal:
        print("Agent training finished!")
        break

    if episode % save_rate == 0:
        two_step_agent.model.load_state_dict(agent.model.state_dict())
        two_step_agent.save_onnx_model(episode)
    
    print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
    env.reset_level()
    reward_gen.reset()

    for step in range(1, max_steps + 1):
        state = env.process_state()
        action = agent.choose_action(state=state)
        done = env.step_action(action=action)
        next_state = env.process_state()

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
    accumulated_reward_per_epsiode.append(reward_gen.accumulated_reward)

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