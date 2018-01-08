from oscar.meta_action.camera import *
from oscar.meta_action.select import *
from oscar.constants import *
from oscar.util.location import Location
from oscar.util.coordinates_helper import Coordinates_helper


"""
Returns a location on the minimap which worth being spyed.
If it doesn't find one, returns None.
Uses random.
"""
def get_spy_target(obs, minimap_location, units_tracker, coordinates_helper = None, nb_try = 5):
    if not coordinates_helper:
        coordinates_helper = Coordinates_helper()

    best_target, best_score = None, None

    for i in range(0, nb_try):
        target = coordinates_helper.get_random_minimap_location()
        score = _score_minimap_location(obs, target, units_tracker, coordinates_helper)

        if not best_score or score > best_score:
            best_score = score
            best_target = target

    if best_score == 0:
    	return None
    else:
    	return best_target


"""
Heuristic to estimate how much a location worth being spied.
Takes into account:
    - enemy units visible on the minimap
    - date of the last observation at this place
"""
def _score_minimap_location(obs, minimap_location, units_tracker, coordinates_helper):

    """ Scoring parameters """
    _ENEMY_WEIGHT = 0.5
    _DATE_WEIGHT = 0.5

    """ Computes a score based on the presence of ennemies """
    enemy_score = 0
    mini_player_relative = obs.observation["minimap"][MINI_PLAYER_RELATIVE]
    for x in range(minimap_location.x, minimap_location.x + coordinates_helper.field_of_view_minimap['x'] - 1):
        for y in range(minimap_location.y, minimap_location.y + coordinates_helper.field_of_view_minimap['y'] - 1):
            if mini_player_relative[y, x] == PLAYER_HOSTILE:
                enemy_score += 1
    if enemy_score == 0:
    	return 0

    # Normalization
    enemy_score /= coordinates_helper.field_of_view_minimap['x'] * coordinates_helper.field_of_view_minimap['y']

    """ Computes a score based on the date of the last observation of the location """
    last_scan_date = units_tracker.get_last_scan_time(
        minimap_location.x - coordinates_helper.field_of_view_minimap['x'] / 2, 
        minimap_location.x + coordinates_helper.field_of_view_minimap['x'] / 2, 
        minimap_location.y - coordinates_helper.field_of_view_minimap['y'] / 2, 
        minimap_location.y + coordinates_helper.field_of_view_minimap['y'] / 2, 
      )
    timestamp = obs.observation["score_cumulative"][1]
    date_score = None
    if last_scan_date:
        date_score = (timestamp - last_scan_date) / timestamp
    else:
        date_score = 1

    return (enemy_score * _ENEMY_WEIGHT + date_score * _DATE_WEIGHT)