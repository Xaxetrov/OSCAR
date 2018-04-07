import numpy as np
from oscar.constants import *
import random


def attack_minimap(obs, queued=True):
    """
    Attacks a position on the minimap. Position corresponds to a visible enemy on the screen if any, 
    randomly on the minimap otherwise (sort of scouting...)
    :param obs: the observation given by pysc2 at current step
    :param queued: if the returned action must be queued or not
    :return: a list of one attack minimap action
    """
    minimap_player_relative = obs.observation['minimap'][MINI_PLAYER_RELATIVE]
    minimap_height = obs.observation['minimap'][MINI_HEIGHT_MAP]
    minimap_visibility = obs.observation['minimap'][MINI_VISIBILITY]
    enemy_pos_y, enemy_pos_x = (minimap_player_relative == PLAYER_HOSTILE).nonzero()
    if len(enemy_pos_x) > 0:
        pos = random.choice(list(zip(enemy_pos_x, enemy_pos_y)))
    else:
        scout_pos_y, scout_pos_x = ((minimap_height != 0) & (minimap_visibility == 0)).nonzero()
        if len(scout_pos_x) > 0:
            pos = random.choice(list(zip(scout_pos_x, scout_pos_y)))
        else:
            minimap_size = np.shape(minimap_player_relative)[0]
            pos_x = random.randint(0, minimap_size - 1)
            pos_y = random.randint(0, minimap_size - 1)
            pos = (pos_x, pos_y)
    return [actions.FunctionCall(ATTACK_MINIMAP, [QUEUED if queued else NOT_QUEUED, pos])]