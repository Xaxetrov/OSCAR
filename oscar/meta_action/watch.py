from oscar.util.screen_helper import *
from oscar.meta_action.camera import *
from oscar.constants import *
from oscar.util.coordinates_helper import Coordinates_helper

"""
Moves camera to a certain location with enemies to watch.
Non-deterministic.
"""
def watch_enemy(obs, coordinates_helper, observer):    
    NB_RANDOM_LOCATIONS = 5
    best_location, best_score = None, None
    for i in range(NB_RANDOM_LOCATIONS):
        new_loc = coordinates_helper.get_random_minimap_location()
        score = observer.score_minimap_location(obs, new_loc)

        if not score or score >= best_score:
            best_score = score
            best_location = new_loc

    return move_camera(best_location, coordinates_helper)