from pysc2.lib import actions
import random

from oscar.agent.custom_agent import CustomAgent


# TODO: As RandomAgent is no more callable directly by pysc2, action_spec is undefined.
class RandomAgent(CustomAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs, locked_choice=None):
        output = []
        selected_action_id = random.choice(obs.observation["available_actions"])
        args = self.action_spec.functions[selected_action_id].args
        for arg in args:
            output.append([random.randint(0, size - 1) for size in arg.sizes])
        return {'actions': [actions.FunctionCall(selected_action_id, output)]}




