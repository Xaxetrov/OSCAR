from oscar.constants import *


def compute_explored_ratio(obs, shared):
    visibility = obs.observation[MINIMAP][MINI_VISIBILITY]
    explored_y, explored_x = (visibility != UNEXPLORED_CELL).nonzero()
    return len(explored_x) / (shared['minimap'].width(obs) * shared['minimap'].height(obs))