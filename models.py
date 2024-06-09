from onnxsim import simplify 
import onnx

import torch.nn as nn
import torch

class NLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(NLPModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        return self.model(x)

    def save_onnx_model(self, episode):
        torch_input = torch.randint(1, (1, self.input_size), dtype=torch.float32)
        
        # Export the model to ONNX
        onnx_path = f"onnxs/sokoban_model_{episode}.onnx"
        torch.onnx.export(self.model, torch_input, onnx_path, opset_version=18)
        
        # Load and simplify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx_model_simplified, check = simplify(onnx_model)
        
        # Ensure the simplified model is valid
        assert check, "Simplified ONNX model could not be validated"

        # Save the simplified ONNX model
        onnx.save(onnx_model_simplified, onnx_path)
        print(f"Model saved to {onnx_path}")