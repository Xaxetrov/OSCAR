import time
from pysc2.agents import base_agent

from oscar.meta_action.meta_action import *


class Economic(base_agent.BaseAgent):
    def __init__(self):
        self.actions_list = []
        super().__init__()

    def step(self, obs):
        if obs.observation["player"][1] >= 100 and len(self.actions_list) == 0:
            self.actions_list = build(obs, 2, BUILD_SUPPLY_DEPOT)
            return self.actions_list.pop()
        if self.actions_list:
            return self.actions_list.pop()
        return actions.FunctionCall(NO_OP, [])

