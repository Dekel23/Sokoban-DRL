from maraboupy import Marabou, MarabouUtils, MarabouCore, MarabouNetworkONNX
import numpy as np

class Verificator:
    def __init__(self, model_path, n_steps, save_ipqs=False):
        self.model = Marabou.read_onnx(model_path)
        self.n_steps = n_steps
        self.save_ipqs = save_ipqs

    def encode_k_steps(self):
        if self.n_steps >= 2:
            self.update_inputs_and_outputs_for_k_steps()
            self.update_remaining_variables_for_k_steps()
            self.update_bounds_for_k_steps()
            self.update_equations_for_k_steps()
            self.finally_update_marabou_network_obj_metadata()

        # the updated k-step model
        if self.save_ipqs:
            self.model.saveQuery("small_model_with_k_"+str(self.n_steps)+"_after_fix")

    def stam(self):
        print(self.model.inputVars)
        original_amount_of_inputs = len(self.model.inputVars[0][0])
        array_of_new_inputs = np.arange(original_amount_of_inputs * self.n_steps)
        self.model.inputVars[0] = array_of_new_inputs
        print(self.model.inputVars)

        print(self.model.outputVars)
        first_original_output = self.model.outputVars[0][0][0]
        last_original_output = self.model.outputVars[0][0][-1] + 1
        array_of_new_outputs = [np.arange(first_original_output * self.n_steps, last_original_output * self.n_steps)]
        self.model.outputVars[0] = array_of_new_outputs
        print(self.model.outputVars)
        # # encode new INPUTS
        # original_amount_of_inputs = len(self.model.inputVars[0])
        # array_of_new_inputs = np.arange(original_amount_of_inputs*self.n_steps)
        # self.model.inputVars[0] = array_of_new_inputs

        # # encode new OUTPUTS
        # FIRST_NEW_OUTPUT_VARIABLE_INDEX = original_amount_of_inputs*self.n_steps
        # original_amount_of_outputs = len(self.model.outputVars[0])
        # array_of_new_outputs = np.arange(FIRST_NEW_OUTPUT_VARIABLE_INDEX, FIRST_NEW_OUTPUT_VARIABLE_INDEX+(original_amount_of_outputs*self.n_steps))
        # self.model.outputShape[1] = original_amount_of_outputs*self.n_steps
        # self.model.outputVars = np.reshape(array_of_new_outputs, newshape=(1, original_amount_of_outputs*self.n_steps))

def main():
    ver = Verificator('./onnxs/sokoban_model_200.nnet', 2)
    ver.stam()

if __name__ == '__main__':
    main()