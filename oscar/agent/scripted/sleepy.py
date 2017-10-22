from pysc2.agents import base_agent
from pysc2.lib import actions

_NO_OP = actions.FUNCTIONS.no_op.id


class NoOpAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs):
        return actions.FunctionCall(_NO_OP, [])




