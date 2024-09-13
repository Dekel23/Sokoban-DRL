import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from reward_gen import *
from model_factory import *
from game import SokobanGame
from hyperopt import hp, fmin, tpe, Trials, space_eval

# Simulate the DRL
def run(env: SokobanGame, row, col ,agent:Agent, reward_gen:RewardGenerator, max_episodes, max_steps, successes_before_train, continuous_successes_goal):
    successful_episodes = 0
    continuous_successes = 0
    steps_per_episode = [] # Steps buffer
    loops_per_episode = [] # Loops buffer
    accumulated_reward_per_epsiode = [] # Rewards buffer
    total_episodes = 0
    
    for episode in range(1, max_episodes + 1):

        if continuous_successes >= continuous_successes_goal: # If reached goal
            print(f"Agent training finished! on episode: {episode-1}")
            break
        
        total_episodes += 1
        print(f"Episode {episode} Epsilon {agent.epsilon:.4f}")
        env.reset_level() # Reset to start of level
        reward_gen.reset() # Reset reward counters

        for step in range(1, max_steps + 1):
            state = env.process_state() # Process current state
            action = agent.choose_action(state=state) # Agent choose action 
            done = env.step_action(action=action)  # Preform move
            next_state = env.process_state() # Process next state

            reward = reward_gen.calculate_reward(state, next_state, done, agent.replay_buffer) # Calculate step reward

            # Store step in replay buffer
            state = np.reshape(state, (row * col,))
            next_state = np.reshape(next_state, (row * col,))
            agent.store_replay(state, action, reward, next_state, done)

            # if reward > 0: # If good move store in prioritized replay buffer
            #     agent.copy_to_prioritized_replay(1)

            if successful_episodes >= successes_before_train: # If start learning
                agent.replay() # Update model parameters
                agent.update_target_model() # Update target model parameters

            if done:
                successful_episodes += 1
                continuous_successes += 1
                print(f"SOLVED! Episode {episode} Steps: {step} Epsilon {agent.epsilon:.4f}")
                print(continuous_successes)
                steps_per_episode.append(step)
                agent.copy_to_prioritized_replay(step) # Copy last moves to prioritiezed replay
                break
        
        # Update steps, loops and rewards buffers
        loops_per_episode.append(reward_gen.loop_counter)
        accumulated_reward_per_epsiode.append(reward_gen.accumulated_reward)
        if not done:
            continuous_successes = 0
            steps_per_episode.append(max_steps)
    
    if total_episodes == max_episodes:
        print(f"Agent training didn't finished!")
    
    return total_episodes, steps_per_episode, loops_per_episode, accumulated_reward_per_epsiode

# Plot simulation results
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

# Objective function to minimize
def objective(param, env, row, col, train_param):
    # Convert hyperparameters
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
    tot_episodes = 0
    siml = 4
    for _ in range(siml): # Simulate 5 times
        model, optimizer = build_model(row=row, col=col, input_size=row*col, output_size=4, **model_hyperparameters) # Create model
        agent = Agent(model=model, optimizer=optimizer, row=row, col=col, **agent_hyperparameters) # Create agent
        reward_gen = build_gen(**reward_hyperparameters) # Create reward system
        episodes, _, _, _ = run(env=env, row=row, col=col, agent=agent, reward_gen=reward_gen, **train_param)
        tot_episodes += episodes # Calculate total episodes
    return tot_episodes/(siml*train_param['max_episodes']) # Return loos value

# Search optimal hyper-parametes for specific level, model, and reward system 
def search_optim(file_name, env, col, row, train_param, args):
    # Define space for bayesian hyperparameter optimization
    space = {}

    # model parameters
    space['model_name'] = args.model

    # agent parameters
    space['epsilon'] = 1.0
    space['gamma'] =  1 - hp.loguniform("gamma", -3, -1)
    space['epsilon_min'] = hp.normal("epsilon_min", 0.13, 0.03)
    space['epsilon_decay'] = 1 - hp.loguniform("epsilon_decay", -4, -2)
    space['beta'] = hp.normal("beta", 0.9, 0.05)
    space['batch_size'] = hp.choice("batch_size", [20, 24])
    space['prioritized_batch_size'] = hp.randint("prioritized_batch_size", 5, 15)

    # reward parameters
    space['reward_name'] = args.reward_gen
    space['r_done'] = hp.uniform("r_done", 10, 50)
    space['loop_size'] = 5
    if args.reward_gen == "HotCold":
        space['r_waste'] = 0
        space['r_move'] = 0
        space['r_hot'] = hp.uniform("r_hot", 0.5, 4)
        space['r_cold'] = hp.uniform("r_cold", -4, -0.5)
    else:
        space['r_waste'] = hp.uniform("r_waste", -4, -0.5)
        space['r_move'] = hp.uniform("r_move", -3.5, -0.5)
        space['r_hot'] = 0
        space['r_cold'] = 0
    if args.loops:
        space['r_loop'] = hp.uniform("r_loop", -1, 0.1)
        space['loop_decay'] = hp.uniform("loop_decay", 0.5, 1)
    else:
        space['r_loop'] = 0
        space['loop_decay'] = 0

    trials = Trials()
    best = fmin(fn=lambda param: objective(param, env, row, col, train_param), space=space, algo=tpe.suggest, max_evals=args.iter, trials=trials)
    best_space = space_eval(space, best)

    # Convert all numpy.int64 types to int
    for key, value in best_space.items():
        if isinstance(value, np.int64):
            best_space[key] = int(value)

    # Save best hyperparameters dictionary in json file
    with open("best_hyperparameters/" + file_name + ".json", 'w') as f:
        json.dump(best_space, f)

def test_optim(file_name, env, col, row, train_param, args):
    # Load best hyperparameters
    with open("best_hyperparameters/" + file_name + ".json", 'r') as f:
        best_param = json.load(f)
    
    # Convert hyperparameters
    model_hyperparameters = {
        'name': best_param['model_name']
    }
    agent_hyperparameters = {
        'gamma': best_param['gamma'],
        'epsilon': best_param['epsilon'],
        'epsilon_min': best_param['epsilon_min'],
        'epsilon_decay': best_param['epsilon_decay'],
        'beta': best_param['beta'],
        'batch_size': best_param['batch_size'],
        'prioritized_batch_size': best_param['prioritized_batch_size']
    }
    reward_hyperparameters = {
        'name': best_param['reward_name'],
        'r_waste': best_param['r_waste'],
        'r_done': best_param['r_done'],
        'r_move': best_param['r_move'],
        'r_loop': best_param['r_loop'], 
        'loop_decay': best_param['loop_decay'], 
        'loop_size': best_param['loop_size'],
        'r_hot': best_param['r_hot'],
        'r_cold': best_param['r_cold']
    }
    
    # Simulate number of times
    for _ in range(args.iter):
        model, optimizer = build_model(row=row, col=col, input_size=row*col, output_size=4, **model_hyperparameters) # Create model
        agent = Agent(model=model, optimizer=optimizer, row=row, col=col, **agent_hyperparameters) # Create agent
        reward_gen = build_gen(**reward_hyperparameters) # Create reward system
        episodes, steps, loops, rewards = run(env=env, row=row, col=col, agent=agent, reward_gen=reward_gen, **train_param)
        if episodes < min_episodes: # Update best simulation
            min_episodes = episodes
            min_steps = steps.copy()
            min_loops = loops.copy()
            min_rewards = rewards.copy()

    # Update the file to contain the min episodes
    print(min_episodes)
    if "episode" in best_param:
        best_param["episode"] = min(best_param["episode"], min_episodes)
    else:
        best_param["episode"] = min_episodes
    with open("best_hyperparameters/" + file_name + ".json", 'w') as f:
        json.dump(best_param, f)

    # Plot best simulation data
    plot_run(min_steps, min_loops, min_rewards)

def main(args):
    # Init environment
    env = SokobanGame(level=args.level, graphics_enable=args.graphics, random=args.random)
    row = len(env.map_info) - 2
    col = len(env.map_info[0]) - 2

    file_name = "Level" + str(args.level) + "/" + args.model + "_" + args.reward_gen + "/no_loops"
    if args.loops:
        file_name = "Level" + str(args.level) + "/" + args.model + "_" + args.reward_gen + "/loops"

    train_param = {
        'max_episodes': args.max_episodes, # Max episodes per simulation
        'max_steps': args.max_steps, # Max steps per episode
        'successes_before_train': 10, # Start learning
        'continuous_successes_goal': 20 # End goal
    }
        
    if "search" in args.mode:
        search_optim(file_name, env, row, col, train_param, args)
    
    if "test" in args.mode:
        test_optim(file_name, env, row, col, train_param, args)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Script for formal verification on DRL model of the Japanese game Sokoban")
    parser.add_argument_group("Basic Options")
    parser.add_argument('--level', type=int, default=61, help="Set the level to simulate")
    parser.add_argument('--loops', type=bool, default=True, help="Set to check simulation with loops verification o.w no loops verification")
    parser.add_argument('--mode', type=str, choices=['search', 'test'], nargs='+', default=['search', 'test'], help="Select the simulation mode to either search for optimal hyperparameters, load existing ones and test their performance, or perform both operations")
    parser.add_argument('--iter', type=int, default=10, help="Set number of iterations for simulation")

    parser.add_argument_group("Sokoban Options")
    parser.add_argument('--random', type=bool, default=False, help="Set True for the kepper to start at random position each episode")
    parser.add_argument('--graphics', type=bool, default=False, help="Set True to see the UI of the Sokoban environment")
    
    parser.add_argument_group("Simulation Options")
    parser.add_argument('--model', type=str, choices=["NN1", "NN2", "CNN"], default="NN2", help="Set the NN model for the simulation")
    parser.add_argument('--reward_gen', type=str, choices=["Simple", "HotCold"], default="HotCold", help="Set the Reward Generator for the simulation")
    parser.add_argument('--max_episodes', type=int, default=800, help="Set the number of episodes per simulation")
    parser.add_argument('--max_steps', type=int, default=30, help="Set the number of steps per episode")
    
    args = parser.parse_args()
    main(args)