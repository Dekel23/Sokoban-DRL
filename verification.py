from maraboupy import Marabou, MarabouUtils, MarabouCore, MarabouNetworkONNX
import numpy as np

class Verificator:
    def __init__(self, model_path, save_ipqs=False):
        self.model = Marabou.read_onnx(model_path)
        self.save_ipqs = save_ipqs
        
    