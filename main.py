
import json
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


space = {
    # model parameters
    'model_name': "NN1",

    # agent parameters
    'gamma': 1 - hp.loguniform("gamma", -8, -1),
    'epsilon': 1.0,
    'epsilon_min': hp.normal("epsilon_min", 0.15, 0.05),
    'epsilon_decay': 1 - hp.loguniform("epsilon_decay", -8, -2),
    'beta': hp.normal("beta", 0.85, 0.05),
    'batch_size': hp.choice("batch_size", [16, 20, 24]),
    'prioritized_batch_size': hp.randint("prioritized_batch_size", 5, 10),

    # reward parameters
    'reward_name': "Simple",
    'r_waste': hp.uniform("r_waste", -5, -0.5),
    'r_done': hp.uniform("r_done", 10, 100),
    'r_move': hp.uniform("r_move", -3, 0),
    'r_loop': 0, 
    'loop_decay': 0.75, 
    'loop_size': 5,
    'r_hot': 5,
    'r_cold': -5
}

train_param = {
    'max_episodes': 800,
    'max_steps': 30,
    'successes_before_train': 10,
    'continuous_successes_goal': 20
}

def objective(param):
    model_hyperparameters = {
        'name': param['model_name']
    }
    agent_hyperparameters = {
        'gamma': param['gamma'],
        'epsilon': param['epsilon'],
        'epsilon_min': param['epsilon_min'],
        'epsilon_decay': param['epsilon_decay'],
        'beta': param['beta'],
        'batch_size': param['batch_size'],
        'prioritized_batch_size': param['prioritized_batch_size']
    }
    reward_hyperparameters = {
        'name': param['reward_name'],
        'r_waste': param['r_waste'],
        'r_done': param['r_done'],
        'r_move': param['r_move'],
        'r_loop': param['r_loop'], 
        'loop_decay': param['loop_decay'], 
        'loop_size': param['loop_size'],
        'r_hot': param['r_hot'],
        'r_cold': param['r_cold']
    }
    model, optimizer = build_model(row=row, col=col, input_size=row*col, output_size=4, **model_hyperparameters)
    agent = Agent(model=model, optimizer=optimizer, row=row, col=col, **agent_hyperparameters)
    reward_gen = build_gen(**reward_hyperparameters)
    episodes, moves, loops, tot_rewards = run(agent=agent, reward_gen=reward_gen, **train_param)
    return episodes/train_param['max_episodes']

def run(agent:Agent, reward_gen:RewardGenerator, max_episodes, max_steps, successes_before_train, continuous_successes_goal):
    successful_episodes = 0
    continuous_successes = 0
    steps_per_episode = []
    loops_per_episode = []
    accumulated_reward_per_epsiode = []
    total_episodes = 0
    
    for episode in range(1, max_episodes + 1):

        if continuous_successes >= continuous_successes_goal:
            print(f"Agent training finished! on episode: {episode-1}")
            break
        
        total_episodes += 1
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
        
        loops_per_episode.append(reward_gen.loop_counter)
        accumulated_reward_per_epsiode.append(reward_gen.accumulated_reward)

        if not done:
            continuous_successes = 0
            steps_per_episode.append(max_steps)
    
    if total_episodes == max_episodes:
        print(f"Agent training didn't finished!")
    
    return total_episodes, steps_per_episode, loops_per_episode, accumulated_reward_per_epsiode

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

trails = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=2, trials=trails)
bext_space = space_eval(space, best)

# Convert all numpy.int64 types to int
for key, value in bext_space.items():
    if isinstance(value, np.int64):
        bext_space[key] = int(value)

with open('best_hyperparameters.json', 'w') as f:
    json.dump(bext_space, f)

print("Best hyperparameters saved to best_hyperparameters.json")
print(bext_space)