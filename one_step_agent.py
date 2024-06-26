import torch
import torch.nn as nn
import onnx
import numpy as np
from copy import deepcopy
from agent import Agent

class OneStepAgent(Agent):
    def __init__(self, agent_p, row, col):
        super(OneStepAgent, self).__init__(**agent_p)
        self.row = row
        self.col = col

    def forward(self, state):
        # Ensure state is a tensor
        state = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)

        actions = super(OneStepAgent, self).forward(state)
        action = actions.argmax().to(torch.int64)
        return self.move_one_step(state, action)
    
    def find_kepper(self, state):
        idxs = torch.nonzero((state == 6) | (state == 7), as_tuple= True)
        return torch.stack(idxs)[:,0]
                
    def move_one_step(self, state, action):
        state = state.reshape(self.row, self.col)
        directions = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])
        dist = directions[action]
        
        y_pos, x_pos = self.find_kepper(state)
        new_state = state.clone()

        # Create a mask for the keeper's position
        keeper_mask = (torch.arange(self.row)[:, None] == y_pos) & (torch.arange(self.col) == x_pos)
        
        # Update keeper's position
        new_state[keeper_mask & (state == 6)] = 2
        new_state[keeper_mask & (state == 7)] = 3

        # Create a mask for the next position
        next_pos_mask = (torch.arange(self.row)[:, None] == (y_pos + dist[0])) & (torch.arange(self.col) == (x_pos + dist[1]))
        
        # Update next position
        new_state[next_pos_mask & (state == 2)] = 6
        new_state[next_pos_mask & (state == 3)] = 7
        new_state[next_pos_mask & (state == 4)] = 6
        new_state[next_pos_mask & (state == 5)] = 7

        # Create a mask for the position two steps away
        next_next_pos_mask = (torch.arange(self.row)[:, None] == (y_pos + 2*dist[0])) & (torch.arange(self.col) == (x_pos + 2*dist[1]))
        
        # Update position two steps away
        box_push_mask = next_pos_mask & (state.unsqueeze(-1) == torch.tensor([4, 5])).any(-1)
        new_state[next_next_pos_mask & (state == 2) & box_push_mask] = 4
        new_state[next_next_pos_mask & (state == 3) & box_push_mask] = 5

        return new_state.reshape(self.row * self.col)
    
    def save_onnx_model(self, episode):
        # Use a dummy input tensor that matches the expected input size
        dummy_input = torch.tensor([6,2,2,2,2,2,2,2,2,2,4,2,2,2,2,3], dtype=torch.float32)
        
        # Export the model to ONNX
        onnx_path = f"onnxs/sokoban_model_{episode}.onnx"
        torch.onnx.export(self, dummy_input, onnx_path)
        
        # Load and save the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.save(onnx_model, onnx_path)
        print(f"Model saved to {onnx_path}")

class KStepAgent(OneStepAgent):
    def __init__(self, agent_p, row, col):
        super(KStepAgent, self).__init__(agent_p, row, col)

    def forward(self, state, k=1):
        # Ensure state is a tensor
        state = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        
        out = state
        for _ in range(k):
            out = super().forward(out)
        return out