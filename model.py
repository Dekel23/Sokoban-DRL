import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch as T

class DQNetWork(nn.Module):
    def __init__(self,input_size, fc1_size, fc2_size, action_size,lr):
        super(DQNetWork, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.action_size = action_size
        self.fc1 = nn.Linear(*self.input_size, self.fc1_size) # 3 levels Neural Network
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, self.action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) # Adam optimizer fot the nn parameters
        self.loss = nn.MSELoss() # Loss function is Mean Squared Error

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # Divice to run the program
        self.to(self.device)

    # Push the info threw the nn and optimize it
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
    
class Agent:
    def __init__(self, gamma, epsilon, lr, input_size, batch_size, action_size,
                 epsilon_min, epsilon_dec, max_mem_size=100000):
        self.gamma = gamma
        self.lr = lr

        self.epsilon = epsilon # Parameters that control the randomness of the agent
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec

        self.action_space = [i for i in range(action_size)] # Parameters that control the memory of the agent
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.iter_counter = 0
        self.replay_rate = 10

        self.replace_target = 100

        self.Q_eval = DQNetWork(input_size=input_size, fc1_size=256, fc2_size=256, action_size=action_size, lr=lr) # The model the agent use
        self.state_memory = np.zeros((self.mem_size, *input_size), dtype=np.float32) # State memory
        self.new_state_memory = np.zeros((self.mem_size, *input_size), dtype=np.float32) # Next state memory
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32) # Action memory
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) # Reward memory
        self.done_memory = np.zeros(self.mem_size, dtype=np.bool_) # Done memory

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
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    # 
    def learn(self):
        if self.mem_counter < self.batch_size: # if the memory is less than batch size
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size) # choose batch size of random memories 
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_counter += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_dec
