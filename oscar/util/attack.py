import random
from oscar.constants import *
from oscar.util.point import Point


def get_random_enemy_location(obs):
    player_relative = obs.observation['minimap'][MINI_PLAYER_RELATIVE]
    hostile_y, hostile_x = (player_relative == PLAYER_HOSTILE).nonzero()

    if len(hostile_x) == 0:
    	return None
    else:
	    hostile = list(zip(hostile_x, hostile_y))
	    selected = random.choice(hostile)
	    return Point(selected[0], selected[1])