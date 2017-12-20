from oscar.meta_action.camera import *
from oscar.meta_action.select import *
from oscar.constants import *
from oscar.util.location import Location
from oscar.util.coordinates_helper import Coordinates_helper
from oscar.util.exploration_helper import *

def scout(obs, coordinates_helper, cur_location = None, propagate_error = True):
    result_action_list = None
    try:
        result_action_list = select_scv(obs)
    except NoValidSCVError:
        if propagate_error:
            raise

    if not cur_location:
        cur_location = coordinates_helper.get_loc_in_minimap(obs)

    target = get_new_target(obs, cur_location, coordinates_helper, 5)

    # move camera to target
    move_camera1 = move_camera(target, coordinates_helper)[0]
    result_action_list.append(move_camera1)
    
    # give move order to unit
    move_to_target = actions.FunctionCall(ATTACK_SCREEN, [NOT_QUEUED, coordinates_helper.get_screen_center().to_array()])
    result_action_list.append(move_to_target)
    
    # move camera back
    move_camera2 = move_camera(cur_location, coordinates_helper)[0]
    result_action_list.append(move_camera2)

    return result_action_list

