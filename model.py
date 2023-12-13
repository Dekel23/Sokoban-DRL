import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch as T

class DQNetWork(nn.Module):
    def __init__(self, input_size, fc1_size, fc2_size, action_size, lr, lr_dec):
        super(DQNetWork, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.action_size = action_size

        # 3 levels Neural Network
        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, self.action_size)

        # Adam optimizer fot the nn parameters
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_dec)
        self.loss = nn.MSELoss()  # Loss function is Mean Squared Error

        # Device to run the program
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Push all the data through the neural network
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, lr_dec, input_size, batch_size, action_size,
                 epsilon_min, epsilon_dec, max_mem_size=100000):
        self.gamma = gamma

        # Parameters that control the randomness of the agent
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

        # Parameters that control the memory of the agent
        self.action_space = [i for i in range(action_size)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0

        # Learning parameters
        self.lr = lr
        self.lr_dec = lr_dec
        self.iter_counter = 0
        self.replay_rate = 10
        self.target_rate = 10

        #self.q_target = T.zeros(batch_size)

        self.Q_eval = DQNetWork(input_size=input_size, fc1_size=50, fc2_size=50,
                                action_size=action_size, lr=lr, lr_dec=lr_dec)  # The model the agent uses
        
        self.state_memory = np.zeros(
            (self.mem_size, input_size), dtype=np.float32)  # State memory
        self.new_state_memory = np.zeros(
            (self.mem_size, input_size), dtype=np.float32)  # Next state memory
        self.action_memory = np.zeros(
            self.mem_size, dtype=np.int32)  # Action memory
        self.reward_memory = np.zeros(
            self.mem_size, dtype=np.float32)  # Reward memory
        self.done_memory = np.zeros(
            self.mem_size, dtype=np.bool_)  # Done memory

    # Store move in the game in the memory
    def store_transition(self, state, action, reward, next_state, done): 
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.done_memory[index] = done

        self.mem_counter += 1

    # Choose what action to take using the model
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array(state)).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_counter < self.batch_size:  # If the memory is less than batch size
            return

        self.Q_eval.optimizer.zero_grad() # Reset gradiants to zero

        # Choose batch size of random memories
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device) # Take batch from each memory
        new_state_batch = T.tensor(
            self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
            self.reward_memory[batch]).to(self.Q_eval.device)
        done_batch = T.tensor(self.done_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] # Calculate q_eval and q_next
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[done_batch] = 0.0
        
        if self.iter_counter % self.target_rate == 0: # in each target_rate the q_target is updated
            q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]


        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_counter += 1
        
        self.epsilon_decay()
        self.lr_decay()

    def epsilon_decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
    
    def lr_decay(self):
        self.Q_eval.scheduler.step()
