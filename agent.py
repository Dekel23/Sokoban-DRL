from collections import deque
import random
import torch.optim as optim
import torch.nn as nn
import torch
from copy import deepcopy
import onnx

class Agent(nn.Module):
    def __init__(self, gamma, epsilon, epsilon_decay, epsilon_min, input_size, beta):
        super(Agent, self).__init__()

        self.input_size = input_size
        self.action_size = 4
        self.action_space = [i for i in range(self.action_size)]
        
        self.model, self.model_optimizer = self._build_model()
        self.target_model, self.target_model_optimizer = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.beta = beta

        self.batch_size = 10
        self.prioritized_batch_size = 10
        self.replay_buffer = deque(maxlen=15000)
        self.prioritized_replay_buffer = deque(maxlen=5000)
    
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.action_size)
        )
        optimizer = optim.Adam(model.parameters())
        return model, optimizer

    def forward(self, x):
        return self.model(x)

    def store_replay(self, state, action, reward, next_state, done):
        self.replay_buffer.appendleft([state, action, reward, next_state, done])

    def copy_to_prioritized_replay(self, steps):
        for i in range(min(self.prioritized_batch_size, steps)):
            self.prioritized_replay_buffer.appendleft(self.replay_buffer[i])

    def choose_action(self, state):
        if random.random() > self.epsilon:
            state = torch.tensor(state, dtype=torch.float32)
            actions = self(state)
            action = torch.argmax(actions).item()
        else:
            action = random.choice(self.action_space)

        return action
    
    def replay(self):
        minibatch = random.sample(self.replay_buffer, self.batch_size // 2)
        minibatch.extend(random.sample(self.prioritized_replay_buffer, self.batch_size // 2))

        states = torch.zeros((self.batch_size, self.input_size), dtype=torch.float32)
        targets = torch.zeros((self.batch_size, self.action_size), dtype=torch.float32)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            target = self.model(state_tensor).detach()

            if done:
                target[action] = reward
            else:
                max_action = self.model(next_state_tensor).argmax().item()
                target[action] = reward + self.gamma * self.target_model(next_state_tensor)[max_action].item()

            states[i] = state_tensor
            targets[i] = target
        
        self.model_optimizer.zero_grad()
        loss = nn.MSELoss()(self.model(states), targets)
        loss.backward()
        self.model_optimizer.step()
        self.update_epsilon()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
    
    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.beta * target_param.data + (1 - self.beta) * param.data)



    def save_onnx_model(self, episode):
        torch_input = torch.randint(1, (1, self.input_size), dtype=torch.float32)
        
        # Export the model to ONNX
        onnx_path = f"onnxs/sokoban_model_{episode}.onnx"
        torch.onnx.export(self, torch_input, onnx_path, opset_version=18)
        
        # Load and simplify the ONNX model
        onnx_model = onnx.load(onnx_path)
        # try:
        #     onnx_model, check = simplify(onnx_model)
        # except:
        #     pass
        
        # Ensure the simplified model is valid
        # assert check, "Simplified ONNX model could not be validated"

        # Save the simplified ONNX model
        onnx.save(onnx_model, onnx_path)
        print(f"Model saved to {onnx_path}")