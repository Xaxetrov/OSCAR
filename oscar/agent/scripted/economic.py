import time
from pysc2.agents import base_agent

from oscar.meta_action.meta_action import *

_NO_OP = actions.FUNCTIONS.no_op.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_NOT_QUEUED = [0]


class Economic(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs):
        time.sleep(0.02)
        if self.steps % 500 == 0:
            build(obs, 2)
            print("steps :", self.steps)
        self.steps += 1
        return actions.FunctionCall(_NO_OP, [])
        # try:
        #     return select_scv(obs)
        # except NoValidSCVError:
        #     return actions.FunctionCall(_NO_OP, [])




