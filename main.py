
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from model_factory import *
from reward_gen import *
from game import SokobanGame
from hyperopt import hp, fmin, tpe, Trials, space_eval

# init env
env = SokobanGame(level=61, graphics_enable=False)
row = len(env.map_info) - 2
col = len(env.map_info[0]) - 2

# init model
model_type = "NN1"
model_parameters = {

}
model, optimizer = build_model(name=model_type, row=row, col=col, input_size=row*col, output_size=4, **model_parameters)

# init agent
agent_hyperparameters = {
    'gamma': 0.7,
    'epsilon': 1.0,
    'epsilon_min': 0.1,
    'epsilon_decay': 0.9995,
    'beta': 0.9,
    'batch_size': 20,
    'prioritized_batch_size': 8
}
agent = Agent(model=model, optimizer=optimizer, row=row, col=col, **agent_hyperparameters)

# init reward generator
reward_hyperparameters = {
    # 'r_waste': -5,
    # 'r_done': 50,
    # 'r_move': -0.5,
    'r_loop': -10, 
    # 'loop_decay': 0.75, 
    'loop_size': 5,
    'r_done': 100,
    'r_hot': 10,
    'r_cold': -5
}
reward_gen = HotCold(**reward_hyperparameters)

train_hyperparameters = {
    'max_episodes': 1000,
    'max_steps': 30,
    'successes_before_train': 10,
    'continuous_successes_goal': 20
}

successful_episodes = 0
continuous_successes = 0
steps_per_episode = []
loops_per_episode = []
accumulated_reward_per_epsiode = []


for episode in range(1, train_hyperparameters['max_episodes'] + 1):

    if continuous_successes >= train_hyperparameters['continuous_successes_goal']:
        print("Agent training finished!")
        break
    
    print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
    env.reset_level()
    reward_gen.reset()

    for step in range(1, train_hyperparameters['max_steps'] + 1):
        state = env.process_state()
        action = agent.choose_action(state=state)
        done = env.step_action(action=action)
        next_state = env.process_state()

        reward = reward_gen.calculate_reward(state, next_state, done, agent.replay_buffer)

        state = np.reshape(state, (row * col,))
        next_state = np.reshape(next_state, (row * col,))
        agent.store_replay(state, action, reward, next_state, done)

        if reward == reward_hyperparameters['r_hot']:
            agent.copy_to_prioritized_replay(1)

        if successful_episodes >= train_hyperparameters['successes_before_train']:
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
        steps_per_episode.append(train_hyperparameters['max_steps'])


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