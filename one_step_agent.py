import torch
import torch.nn as nn
import onnx
import numpy as np
from copy import deepcopy
from agent import Agent

class OneStepAgent(Agent):
    def __init__(self, env, agent_p):
        super(OneStepAgent, self).__init__(**agent_p)
        self.env = deepcopy(env)

    def forward(self, state):
        # Ensure state is a tensor
        state = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        
        # Convert tensor to numpy array for set_map_info
        np_state = state.detach().cpu().numpy()
        
        # Reshape the state to match the environment's expectations
        reshaped_state = np_state.reshape(len(self.env.map_info)-2, len(self.env.map_info[0])-2)
        
        self.env.set_map_info(reshaped_state)
        
        actions = super(OneStepAgent, self).forward(state)
        action = torch.argmax(actions).item()
        self.env.step_action(action)
        
        return self.process_state_tensor()

    def process_state_tensor(self):
        processed_state = self.env.process_state(reshape=True)
        # processed_state is already a 1D numpy array, so we can directly convert it to a tensor
        return torch.from_numpy(processed_state).float()

class KStepAgent(OneStepAgent):
    def __init__(self, env, agent_p):
        super(KStepAgent, self).__init__(env, agent_p)

    def forward(self, state, k=1):
        # Ensure state is a tensor
        state = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        
        out = state
        for _ in range(k):
            out = super().forward(out)
        return out

    def save_onnx_model(self, episode):
        # Use a dummy input tensor that matches the expected input size
        dummy_input = torch.tensor([6,2,2,2,2,2,2,2,2,2,4,2,2,2,2,3], dtype=torch.float32)
        
        # Export the model to ONNX
        onnx_path = f"onnxs/sokoban_model_{episode}.onnx"
        torch.onnx.export(self, (dummy_input, torch.tensor(1)), onnx_path, 
                          input_names=['input', 'k'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'input_size'},
                                        'output': {0: 'output_size'}})
        
        # Load and save the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.save(onnx_model, onnx_path)
        print(f"Model saved to {onnx_path}")