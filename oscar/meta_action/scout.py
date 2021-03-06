from oscar.constants import *
from oscar.util.point import Point
from oscar.shared.minimap import Minimap
from oscar.shared.camera import Camera


def scout(obs, shared):
    """ Moves selected unit to scout the map """
    target = _get_scout_target(obs, shared)
    return [actions.FunctionCall(
        MOVE_MINIMAP, [NOT_QUEUED, target.to_array()])]


def _get_scout_target(obs, shared, samples = 5):
    """ Computes a scout target. """
    best_target, best_score = None, None
    camera_location = shared['camera'].location(obs, shared)

    for i in range(0, samples):
        target = shared['minimap'].random_point(obs)
        score = _score_target(obs, shared, camera_location, target)

        if not best_score or score > best_score:
            best_score = score
            best_target = target

    return best_target


def _score_target(obs, shared, camera_location, target):
    """ Scores a scout target using its distance and visibility. """
    _UNEXPLORED_SCORE = 2
    _EXPLORED_SCORE = 1
    _VISIBLE_SCORE = 0
    _DISTANCE_WEIGHT = 0.0

    visibility = obs.observation[MINIMAP][MINI_VISIBILITY]
    distance = camera_location.distance(target)

    visibility_score = 0
    for p in shared['camera'].iterate(obs):
        if visibility[p.y][p.x] == VISIBLE_CELL:
            visibility_score += _VISIBLE_SCORE
        elif visibility[p.y][p.x] == EXPLORED_CELL:
            visibility_score += _EXPLORED_SCORE
        elif visibility[p.y][p.x] == UNEXPLORED_CELL:
            visibility_score += _UNEXPLORED_SCORE

    return visibility_score - _DISTANCE_WEIGHT * distance
