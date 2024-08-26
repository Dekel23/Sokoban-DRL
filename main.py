
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from reward_gen import *
from model_factory import *
from game import SokobanGame
from hyperopt import hp, fmin, tpe, Trials, space_eval

# init env
env = SokobanGame(level=61, graphics_enable=False)
row = len(env.map_info) - 2
col = len(env.map_info[0]) - 2


model_hyperparameters = {
    'name': "NN1"
}
agent_hyperparameters = {
    'gamma': 0.5,
    'epsilon': 1.0,
    'epsilon_min': 0.15,
    'epsilon_decay': 0.9993,
    'beta': 0.9,
    'batch_size': 20,
    'prioritized_batch_size': 7
}
reward_hyperparameters = {
    'name': "Simple",
    'r_waste': -2,
    'r_done': 50,
    'r_move': -0.25,
    'r_loop': -1, 
    'loop_decay': 0.75, 
    'loop_size': 5,
    'r_hot': 5,
    'r_cold': -5
}

train_parameters = {
    'max_episodes': 1000,
    'max_steps': 30,
    'successes_before_train': 10,
    'continuous_successes_goal': 20
}
def run(agent, reward_gen, max_episodes, max_steps, successes_before_train, continuous_successes_goal):
    successful_episodes = 0
    continuous_successes = 0
    steps_per_episode = []
    loops_per_episode = []
    accumulated_reward_per_epsiode = []
    for episode in range(1, max_episodes + 1):

        if continuous_successes >= continuous_successes_goal:
            print("Agent training finished!")
            break
        
        print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
        env.reset_level()
        reward_gen.reset()

        for step in range(1, max_steps + 1):
            state = env.process_state()
            action = agent.choose_action(state=state)
            done = env.step_action(action=action)
            next_state = env.process_state()

            reward = reward_gen.calculate_reward(state, next_state, done, agent.replay_buffer)

            state = np.reshape(state, (row * col,))
            next_state = np.reshape(next_state, (row * col,))
            agent.store_replay(state, action, reward, next_state, done)

            if reward > 0:
                agent.copy_to_prioritized_replay(1)

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
    
    return steps_per_episode, loops_per_episode, accumulated_reward_per_epsiode

def plot_run(steps_per_episode, loops_per_episode, accumulated_reward_per_epsiode):
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

def test(model_hyperparameters, agent_hyperparameters, reward_hyperparameters, train_parameters):
    model, optimizer = build_model(row=row, col=col, input_size=row*col, output_size=4, **model_hyperparameters)
    agent = Agent(model=model, optimizer=optimizer, row=row, col=col, **agent_hyperparameters)
    reward_gen = build_gen(**reward_hyperparameters)
    length = 5

    min_episode = 1000
    min_steps_per_episode = []
    min_loops_per_episode = []
    max_accumulated_reward_per_epsiode = []
    sum_episode = 0

    for _ in range(length):
        steps_per_episode, loops_per_episode, accumulated_reward_per_epsiode = run(agent, reward_gen, **train_parameters)
        agent.epsilon = agent_hyperparameters['epsilon']
        episodes = len(steps_per_episode)
        if episodes < min_episode:
            min_steps_per_episode = steps_per_episode.copy()
            min_loops_per_episode = loops_per_episode.copy()
            max_accumulated_reward_per_epsiode = accumulated_reward_per_epsiode.copy()
        sum_episode += episodes
    return sum_episode/length, min_steps_per_episode, min_loops_per_episode, max_accumulated_reward_per_epsiode

def compare(model_hyperparameters, agent_hyperparameters, reward_hyperparameters, train_parameters):
    avg_episode_loop, steps_per_episode_loop, loops_per_episode_loop, accumulated_reward_per_epsiode_loop = test(model_hyperparameters, agent_hyperparameters, reward_hyperparameters, train_parameters)
    reward_hyperparameters['r_loop'] = 0
    avg_episode, steps_per_episode, loops_per_episode, accumulated_reward_per_epsiode = test(model_hyperparameters, agent_hyperparameters, reward_hyperparameters, train_parameters)

    print(f'with loops: {avg_episode_loop}, without loops: {avg_episode}')
    plot_run(steps_per_episode_loop, loops_per_episode_loop, accumulated_reward_per_epsiode_loop)
    plot_run(steps_per_episode, loops_per_episode, accumulated_reward_per_epsiode)

compare(model_hyperparameters, agent_hyperparameters, reward_hyperparameters, train_parameters)