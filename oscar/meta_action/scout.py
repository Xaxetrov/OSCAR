from oscar.meta_action.camera import *
from oscar.meta_action.select import *
from oscar.constants import *
from oscar.util.location import Location
from oscar.util.coordinates_helper import Coordinates_helper
from oscar.util.exploration_helper import *

def scout(obs, coordinates_helper, cur_location = None, propagate_error = True):
    result_action_list = None

    try:
        result_action_list = select_idle_scv_screen_priority(obs)
    except NoValidSCVError:
        try:
            result_action_list = select_scv_on_screen(obs)
        except NoValidSCVError:
            if propagate_error:
                raise

    if not cur_location:
        cur_location = coordinates_helper.get_loc_in_minimap(obs)

    target = get_new_target(obs, cur_location, coordinates_helper, 5)
    move_to_target = actions.FunctionCall(ATTACK_MINIMAP, [NOT_QUEUED, target.to_array()])
    result_action_list.append(move_to_target)

    return result_action_list

