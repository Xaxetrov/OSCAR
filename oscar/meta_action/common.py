import random
import numpy as np

from oscar.constants import *
from oscar.meta_action.meta_action_error import *


def find_position(obs, unit_type_id, select_method="random_center", player_relative=PLAYER_SELF, exception=NoUnitError):
    # unit_type_id is either an iterable, or a single int. In case it is a single int, it is embedded in a list,
    # so it will be process as an iterable afterward.
    try:
        unit_type_id_list = list(unit_type_id)
    except TypeError:
        unit_type_id_list = [unit_type_id]

    unit_type_map = obs.observation["screen"][UNIT_TYPE]
    player_relative_map = obs.observation["screen"][PLAYER_RELATIVE]
    correct_unit_type_array = np.isin(unit_type_map, unit_type_id_list)
    correct_player_relative_array = (player_relative_map == player_relative)
    unit_y, unit_x = (correct_unit_type_array & correct_player_relative_array).nonzero()
    if len(unit_x) == 0:
        raise exception("Unit of id {0} for the player_relative {1} is not on the screen."
                        .format(unit_type_id_list, player_relative))

    if select_method == "random_center":
        return find_random_center(unit_type_map, unit_x, unit_y)
    elif select_method == "random":
        returned_index = random.randint(0, len(unit_x))
        return (unit_x[returned_index], unit_y[returned_index])
    elif select_method == "mean":
        return (int(unit_x.mean()), int(unit_y.mean()))
    elif select_method == "all":
        return (unit_x, unit_y)
    else:
        valid_select_method_name = ["random_center", "random", "mean", "all"]
        raise ValueError("Unexpected select_method name {0}. Allowed values are: {1}."
                         .format(select_method, ",".join(valid_select_method_name)))


def find_random_center(unit_type_map, unit_x, unit_y):
    while True:
        returned_index = random.randint(0, len(unit_x))
        selected_coordinate_x = unit_x[returned_index]
        selected_coordinate_y = unit_y[returned_index]

        lower_slice_x = max(selected_coordinate_x - 1, 0)
        lower_slice_y = max(selected_coordinate_y - 1, 0)
        proximity_unit_type_array = unit_type_map[lower_slice_x:selected_coordinate_x + 2,
                                                  lower_slice_y:selected_coordinate_y + 2]
        if np.min(proximity_unit_type_array) == np.max(proximity_unit_type_array):
            break

    return (selected_coordinate_x, selected_coordinate_y)
