from copy import deepcopy

class OneStepAgent():
    def __init__(self, agent, env):
        self.agent = agent
        self.env = deepcopy(env) # no changes by reference

    def get_next_state(self):
        current_state = self.env.process_state()
        action = self.agent.choose_action(current_state)
        self.env.step_action(action)

        return self.env.process_state()
    
    def get_current_state(self):
        return self.env.process_state()