from oscar.constants import *
from oscar.util.point import Point
from oscar.util.minimap import Minimap
from oscar.util.camera import Camera


def get_spy_target(obs, enemy_tracker, timestamp, samples = 5):
    """
    Returns a point on the minimap which worth being spyed.
    If it doesn't find one, returns None.
    Uses random.
    """

    best_target, best_score = None, None

    for i in range(0, samples):
        target = Minimap.random_camera_target(obs)
        score = _score_spy_target(obs, target, enemy_tracker, timestamp)

        if not best_score or score > best_score:
            best_score = score
            best_target = target

    if best_score == 0:
    	return None
    else:
    	return best_target


def _score_spy_target(obs, target, enemy_tracker, timestamp):
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
    for p in Camera.iterate(obs, target):
        if mini_player_relative[p.y, p.x] == PLAYER_HOSTILE:
            enemy_score += 1
    if enemy_score == 0:
    	return 0
    enemy_score /= Camera.width(obs) * Camera.height(obs) # Normalization

    """ Computes a score based on the date of the last observation of the location """
    last_scan_date = enemy_tracker.get_last_scan_time(
        int(target.x - Camera.width(obs) / 2), 
        int(target.x + Camera.width(obs) / 2), 
        int(target.y - Camera.height(obs) / 2), 
        int(target.y + Camera.height(obs) / 2))
    date_score = None
    if last_scan_date:
        date_score = (timestamp - last_scan_date) / timestamp
    else:
        date_score = 1

    return (enemy_score * _ENEMY_WEIGHT + date_score * _DATE_WEIGHT)