from collections import deque
import random
import torch.nn as nn
import torch
import onnx
import numpy as np
from model_factory import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, model, optimizer, row, col, gamma, epsilon, epsilon_decay, epsilon_min, beta, batch_size, prioritized_batch_size):
        super(Agent, self).__init__()

        self.row = row
        self.col = col
        self.input_size = row * col
        self.action_size = 4
        self.action_space = [i for i in range(self.action_size)]
        
        # Create model and target model
        self.model, self.model_optimizer = model, optimizer
        self.target_model, self.target_model_optimizer = model, optimizer
        self.target_model.load_state_dict(self.model.state_dict())

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.beta = beta

        self.batch_size = batch_size # Replay size
        self.prioritized_batch_size = prioritized_batch_size # Size to copy to prioritized buffer
        self.replay_buffer = deque(maxlen=15000)
        self.prioritized_replay_buffer = deque(maxlen=5000)

    def forward(self, x):
        return self.model(x)

    # Add step to replay buffer
    def store_replay(self, state, action, reward, next_state, done):
        self.replay_buffer.appendleft([state, action, reward, next_state, done])

    # Copy last steps to prioritized replay buffer
    def copy_to_prioritized_replay(self, steps):
        for i in range(min(self.prioritized_batch_size, steps)):
            self.prioritized_replay_buffer.appendleft(self.replay_buffer[i])

    # Agent choose action based on current state
    def choose_action(self, state):
        if random.random() > self.epsilon: # Exploitation
            state = torch.tensor(state, dtype=torch.float32).to(device)
            if isinstance(self.model, nn.Sequential):
                state = state.view(1, -1)
            else:  # CNN model
                state = state.view(-1, 1, self.row, self.col)
            with torch.no_grad():
                actions = self(state)
            action = torch.argmax(actions).item()
        else: # Exploration 
            action = random.choice(self.action_space)

        return action
    
    # Agent learning function
    def replay(self):
        # Create batch for training
        minibatch = random.sample(self.replay_buffer, 1*self.batch_size // 4)
        minibatch.extend(random.sample(self.prioritized_replay_buffer, 3*self.batch_size // 4))

        # Create input output tensors batch
        states = torch.zeros((self.batch_size, self.input_size), dtype=torch.float32).to(device)
        targets = torch.zeros((self.batch_size, self.action_size), dtype=torch.float32).to(device)

        # Enumerate batch
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device) # Copy state to tensor
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device) # Copy next_state to tensor

            target = self.model(state_tensor).detach().squeeze(0) # Calulate model output (Q value)

            # Change expected output based on Bellman’s Equation and target model (Q function)
            if done:
                target[action] = reward
            else:
                max_action = self.model(next_state_tensor).argmax().item()
                target[action] = reward + self.gamma * self.target_model(next_state_tensor)[max_action].item()

            # Copy to tensor batch
            states[i] = state_tensor.view(-1)
            targets[i] = target
        
        # Update model parametes 
        self.model_optimizer.zero_grad()
        loss = nn.MSELoss()(self.model(states), targets)
        loss.backward()
        self.model_optimizer.step()
        self.update_epsilon()

    # Decrease exploration
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
    
    # Update target model closer to main model
    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.beta * target_param.data + (1 - self.beta) * param.data)

    # Save Agent model to onnx format file
    def save_onnx_model(self, episode):
        # Use a dummy input tensor that matches the expected input size
        dummy_input = torch.tensor([6,2,2,2,2,2,2,2,2,2,4,2,2,2,2,3], dtype=torch.float32).to(device)
        
        # Export the model to ONNX
        onnx_path = f"onnxs/sokoban_model_{episode}.onnx"
        torch.onnx.export(self, dummy_input, onnx_path)

        # Load and save the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.save(onnx_model, onnx_path)
        print(f"Model saved to {onnx_path}")