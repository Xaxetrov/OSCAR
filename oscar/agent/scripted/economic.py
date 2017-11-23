import time
from pysc2.agents import base_agent

from oscar.meta_action.meta_action import *

_NO_OP = actions.FUNCTIONS.no_op.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id

_NOT_QUEUED = [0]


class Economic(base_agent.BaseAgent):
    def __init__(self):
        self.actions_list = []
        super().__init__()

    def step(self, obs):
        if obs.observation["player"][1] >= 100:
            self.actions_list = build(obs, 2, _BUILD_SUPPLY_DEPOT)
            return self.actions_list.pop()
        if self.actions_list:
            return self.actions_list.pop()
        return actions.FunctionCall(_NO_OP, [])

