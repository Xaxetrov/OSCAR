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
    minimap_player_relative = obs.observation[MINIMAP][MINI_PLAYER_RELATIVE]
    minimap_height = obs.observation[MINIMAP][MINI_HEIGHT_MAP]
    minimap_visibility = obs.observation[MINIMAP][MINI_VISIBILITY]
    enemy_pos_y, enemy_pos_x = (minimap_player_relative == PLAYER_HOSTILE).nonzero()
    if len(enemy_pos_x) > 0:
        # find the closest enemy and set him as target
        self_y, self_x = (minimap_player_relative == PLAYER_SELF).nonzero()
        delta_x = np.abs(self_x.reshape((1, -1)) - enemy_pos_x.reshape((-1, 1)))
        delta_y = np.abs(self_y.reshape((1, -1)) - enemy_pos_y.reshape((-1, 1)))
        closest = np.min(delta_x + delta_y, axis=1)
        closest_enemy_id = np.argmin(closest)
        pos = (enemy_pos_x[closest_enemy_id], enemy_pos_y[closest_enemy_id])
    else:
        scout_pos_y, scout_pos_x = ((minimap_height != 0) & (minimap_visibility == 0)).nonzero()
        if len(scout_pos_x) > 0:
            pos = random.choice(list(zip(scout_pos_x, scout_pos_y)))
        else:
            # else explore around minerals
            scout_pos_y, scout_pos_x = (minimap_player_relative == PLAYER_NEUTRAL).nonzero()
            if len(scout_pos_x) > 0:
                pos = random.choice(list(zip(scout_pos_x, scout_pos_y)))
            else:
                minimap_size = np.shape(minimap_player_relative)[0]
                pos_x = random.randint(0, minimap_size - 1)
                pos_y = random.randint(0, minimap_size - 1)
                pos = (pos_x, pos_y)
    return [actions.FunctionCall(ATTACK_MINIMAP, [QUEUED if queued else NOT_QUEUED, pos])]