from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from oscar.agent.commander.Commander import Commander

import random
import sys

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


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




