import random
from oscar.agent.custom_agent import CustomAgent
from oscar.util.attack import *
from oscar.constants import *


class AttackManager(CustomAgent):

    def __init__(self, message="I hate you"):
        self._message = message
        super().__init__()

    def step(self, obs, locked_choice=None):
        play = {}

        enemy_point = get_random_enemy_location(obs)
        if enemy_point:
            play['actions'] = [actions.FunctionCall(SELECT_ARMY, [NEW_SELECTION]),
                actions.FunctionCall(ATTACK_MINIMAP, [NOT_QUEUED, enemy_point.get_flipped().to_array()])]

        if 'actions' not in play:
            play['actions'] = [actions.FunctionCall(NO_OP, [])]

        return play

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)