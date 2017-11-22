import time
from pysc2.agents import base_agent

from oscar.meta_action.meta_action import *

_NO_OP = actions.FUNCTIONS.no_op.id


class Economic(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs):
        time.sleep(0.5)
        try:
            return select_scv(obs)
        except NoValidSCVError:
            return actions.FunctionCall(_NO_OP, [])




