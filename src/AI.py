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

    def __init__(self):
        pass

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass

    def step(self, obs):

        # if some entities are selected
        if _MOVE_SCREEN in obs.observation["available_actions"]:

            # Find our units and targets
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            ennemies_y, ennemies_x = (player_relative == _PLAYER_HOSTILE).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

            """print("ennemies_y: " + str(ennemies_y))
            print("ennemies_x: " + str(ennemies_x))
            print("player_y: " + str(player_y))
            print("player_x: " + str(player_x))"""

            # if no observations, do nothing
            if not ennemies_y.any() or not player_y.any():
                return actions.FunctionCall(_NO_OP, [])

            player = [int(player_x.mean()), int(player_y.mean())]
            
            # compute closest ennemy
            closest, min_dist = None, None
            for p in zip(ennemies_x, ennemies_y):
                dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist

            time.sleep(0.4)
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, closest])
            
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

        