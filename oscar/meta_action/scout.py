from oscar.meta_action import *
from oscar.constants import *
from oscar.util.location import Location
from oscar.util.coordinates_helper import Coordinates_helper

def scout(obs, coordinates_helper, minimap_location = None, propagate_error = True):
    result_action_list = None

    """try:
        result_action_list = select_idle_scv_screen_priority(obs)
    except NoValidSCVError:"""
    try:
        result_action_list = select_scv_on_screen(obs)
    except NoValidSCVError:
        if propagate_error:
            raise

    if not minimap_location:
        minimap_location = coordinates_helper.get_loc_in_minimap(obs)

    target = _get_scout_target(obs, minimap_location, coordinates_helper)
    move_to_target = actions.FunctionCall(
        MOVE_MINIMAP, [NOT_QUEUED, target.to_array()])
    result_action_list.append(move_to_target)

    return result_action_list


"""
Returns a location on the minimap which is as much close and unexplored as possible.
Uses random.
"""
def _get_scout_target(obs, minimap_location, coordinates_helper = None, nb_try = 5):
    if not coordinates_helper:
        coordinates_helper = Coordinates_helper()

    best_target, best_score = None, None

    for i in range(0, nb_try):

        target = coordinates_helper.get_random_minimap_location()
        score = _score_location(obs, minimap_location, target, coordinates_helper)

        if not best_score or score > best_score:
            best_score = score
            best_target = target

    minimap_view_center_offset = Location(
        0.5 * coordinates_helper.field_of_view_minimap['x'],
        0.5 * coordinates_helper.field_of_view_minimap['y']
    )
    return best_target.addition(minimap_view_center_offset)


"""    
Scores a minimap location, based on its distance and its visibility.
Could be negative.
"""
def _score_location(obs, cur_location, target_location, coordinates_helper):
    _UNEXPLORED_SCORE = 2
    _EXPLORED_SCORE = 1
    _VISIBLE_SCORE = 0
    _DISTANCE_WEIGHT = 0.2

    distance = cur_location.distance(target_location)

    visibility = obs.observation['minimap'][MINI_VISIBILITY]
    limits = coordinates_helper.get_minimap_view_limits(target_location)

    visibility_score = 0
    for x in range(limits['x']['min'], limits['x']['max']+1):
        for y in range(limits['y']['min'], limits['y']['max']+1):
            if (visibility[y][x] == VISIBLE_CELL):
                visibility_score += _VISIBLE_SCORE
            elif (visibility[y][x] == EXPLORED_CELL):
                visibility_score += _EXPLORED_SCORE
            elif (visibility[y][x] == UNEXPLORED_CELL):
                visibility_score += _UNEXPLORED_SCORE

    return (visibility_score - _DISTANCE_WEIGHT * distance)