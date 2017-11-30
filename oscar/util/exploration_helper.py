import numpy as np
from oscar.util.location import Location
from oscar.util.coordinates_helper import Coordinates_helper
from oscar.constants import *

# Returns a location in the map which is both close and not explored, if possible.
# Uses random.
def get_new_target(obs, cur_location, coordinates_helper = None, nb_try = 3):
    if not coordinates_helper:
        coordinates_helper = Coordinates_helper()

    best_target, best_target_score = None, None

    for i in range(0, nb_try):
        target_location = coordinates_helper.get_random_minimap_location()
        score = _score_position(obs, cur_location, target_location, coordinates_helper)

        if not best_target_score or score > best_target_score:
            best_target_score = score
            best_target = target_location

    return best_target
    
# Scores a position, based on its distance and its visibility.
# Could be negative.
def _score_position(obs, cur_location, target_location, coordinates_helper):
    _UNEXPLORED_SCORE = 2
    _EXPLORED_SCORE = 1
    _VISIBLE_SCORE = 0
    _DISTANCE_WEIGHT = 4.0

    distance = cur_location.distance(target_location)

    visibility = obs.observation['minimap'][MINI_VISIBILITY]
    limits = coordinates_helper.get_minimap_view_limits(target_location)

    visibility_score = 0
    for col in range(limits['x']['min'], limits['x']['max']+1):
        for row in range(limits['y']['min'], limits['y']['max']+1):
            if (visibility[row][col] == UNEXPLORED_CELL):
                visibility_score += _UNEXPLORED_SCORE
            elif (visibility[row][col] == EXPLORED_CELL):
                visibility_score += _EXPLORED_SCORE
            elif (visibility[row][col] == VISIBLE_CELL):
                visibility_score += _VISIBLE_SCORE
            else:
                raise Exception('Unknown value in visibility map: ' + location_visibility[row][col])

    return (visibility_score - _DISTANCE_WEIGHT * distance)