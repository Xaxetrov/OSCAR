import numpy
import sys
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from random import randint

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]
_SCREEN = [0]

class FindAndDefeatZerglings():

    targets = [None, None]
    positions = [None, None]
    selected = None
    neutral_x, neutral_y, player_x, player_y = None, None, None, None

    def __init__(self):
        pass

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass
        
    def step(self, obs):
        # Find our units and targets
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        self.neutral_y, self.neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        self.player_y, self.player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

        return actions.FunctionCall(_NO_OP, [])