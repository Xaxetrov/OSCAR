import time
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from meta_action import *
from oscar.meta_action.train_unit import *

# Features
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_IDLE_WORKER_COUNT = 7

# Actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id

# Action arguments
_NOT_QUEUED = [0]
# When selecting a worker
_SET = [0]
_ADD = [1]
_ALL = [2]
_ADD_ALL = [3]


class CollectMineralGas(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.state = 0

    def reset(self):
        super().reset()
        self.state = 0

    def step(self, obs):
        # Phase 0 : select idle guy
        if self.state == 0:
            # There are some idle guys
            if obs.observation["player"][_IDLE_WORKER_COUNT] != 0:
                self.state = 1
                return actions.FunctionCall(_SELECT_IDLE_WORKER, [_SET])
        # Idle selected, right click on mineral
        elif self.state == 1:
            self.state = 0
            return harvest_mineral(obs)

        return actions.FunctionCall(_NO_OP, [])
