from pysc2.agents import base_agent
from pysc2.lib import actions

import random


class RandomAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs):
        output = []
        selected_action_id = random.choice(obs.observation["available_actions"])
        args = self.action_spec.functions[selected_action_id].args
        for arg in args:
            output.append([random.randint(0, size - 1) for size in arg.sizes])
        return actions.FunctionCall(selected_action_id, output)




