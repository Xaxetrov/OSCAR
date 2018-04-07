from oscar.constants import *
from oscar.util.point import Point
from oscar.shared.minimap import Minimap
from oscar.shared.camera import Camera


def get_spy_target(obs, shared, samples = 5):
    """
    Returns a point on the minimap which worth being spyed.
    If it doesn't find one, returns None.
    Uses random.
    """

    best_target, best_score = None, None

    for i in range(0, samples):
        target = shared['camera'].random_target(obs)
        score = _score_spy_target(obs, shared, target)

        if not best_score or score > best_score:
            best_score = score
            best_target = target

    if best_score == 0:
        return None
    else:
        return best_target


def _score_spy_target(obs, shared, target):
    """
    Heuristic to estimate how much a point worth being spied.
    Takes into account:
        - enemy units visible on the minimap
        - date of the last observation at this place
    """

    """ Scoring parameters """
    _ENEMY_WEIGHT = 0.5
    _DATE_WEIGHT = 0.5

    """ Computes a score based on the presence of enemies """
    enemy_score = 0
    mini_player_relative = obs.observation["minimap"][MINI_PLAYER_RELATIVE]
    for p in shared['camera'].iterate(obs, target):
        if mini_player_relative[p.y, p.x] == PLAYER_HOSTILE:
            enemy_score += 1
    if enemy_score == 0:
        return 0
    enemy_score /= shared['camera'].width(obs) * shared['camera'].height(obs) # Normalization

    """ Computes a score based on the date of the last observation of the location """
    last_scan_date = shared['enemy_tracker'].get_last_scan_time(
        int(target.x - shared['camera'].width(obs) / 2), 
        int(target.x + shared['camera'].width(obs) / 2), 
        int(target.y - shared['camera'].height(obs) / 2), 
        int(target.y + shared['camera'].height(obs) / 2))
    date_score = None
    if last_scan_date:
        date_score = (shared['env'].timestamp - last_scan_date) / shared['env'].timestamp
    else:
        date_score = 1

    return enemy_score * _ENEMY_WEIGHT + date_score * _DATE_WEIGHT
