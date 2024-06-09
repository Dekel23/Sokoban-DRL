from one_step_agent import OneStepAgent

class KStepAgent():
    def __init__(self, agent, env):
        self.step_agent = OneStepAgent(agent, env)

    def get_k_step(self, k):
        k_step = self.step_agent.get_current_state()

        for _ in range(k):
            k_step = self.step_agent.get_next_state()

        return k_step