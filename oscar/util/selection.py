import random
import numpy as np
from scipy.signal import convolve2d

from oscar.constants import *
from oscar.meta_action.meta_action_error import *
from oscar.shared.env import Env
from oscar.shared.screen import Screen

RANDOM_CENTER_JUMP_LIMIT = 3
RANDOM_CENTER_MOVE_FACTOR = 2
RANDOM_CENTER_ITERATION_LIMIT = 5


def find_position(obs, unit_type_id, select_method="random_center", player_relative=PLAYER_SELF, exception=NoUnitError):
    # unit_type_id is either an iterable, or a single int. In case it is a single int, it is embedded in a list,
    # so it will be process as an iterable afterward.
    try:
        unit_type_id_list = list(unit_type_id)
    except TypeError:
        unit_type_id_list = [unit_type_id]

    unit_type_map = obs.observation[SCREEN][SCREEN_UNIT_TYPE]
    player_relative_map = obs.observation[SCREEN][SCREEN_PLAYER_RELATIVE]
    correct_unit_type_array = np.isin(unit_type_map, unit_type_id_list)
    correct_player_relative_array = (player_relative_map == player_relative)
    unit_y, unit_x = (correct_unit_type_array & correct_player_relative_array).nonzero()
    if len(unit_x) == 0:
        raise exception("Unit of id {0} for the player_relative {1} is not on the screen."
                        .format(unit_type_id_list, player_relative))

    if select_method == "random_center":
        return find_random_center(unit_type_map, unit_x, unit_y, unit_type_id_list)
    elif select_method == "random":
        return random.choice(list(zip(unit_x, unit_y)))
    elif select_method == "mean":
        return int(unit_x.mean()), int(unit_y.mean())
    elif select_method == "all":
        return unit_x, unit_y
    elif select_method == "screen_scan":
        shared = {'env': Env}
        screen = Screen()
        scanned_units = screen.scan(obs=obs, shared=shared)
        for unit in scanned_units:
            if unit.unit_id in unit_type_id_list and unit.player_id == player_relative:
                break
        else:
            print("unit not found, trying random method")
            return random.choice(list(zip(unit_x, unit_y)))
        return unit.location.screen.x, unit.location.screen.y

    else:
        valid_select_method_name = ["random_center", "random", "mean", "all"]
        raise ValueError("Unexpected select_method name {0}. Allowed values are: {1}."
                         .format(select_method, ",".join(valid_select_method_name)))


def find_random_center(unit_type_map, unit_x, unit_y, unit_type_id_list):
    returned_index = random.randint(0, len(unit_x) - 1)
    selected_coordinate_x = unit_x[returned_index]
    selected_coordinate_y = unit_y[returned_index]

    random_jump = 0
    iteration_count = 0
    boolean_type_map = np.isin(unit_type_map, unit_type_id_list) * 1
    while True:
        iteration_count += 1
        lower_slice_x = max(selected_coordinate_x - 1, 0)
        lower_slice_y = max(selected_coordinate_y - 1, 0)
        proximity_unit_type_array = boolean_type_map[lower_slice_x:selected_coordinate_x + 2,
                                                     lower_slice_y:selected_coordinate_y + 2]
        if np.min(proximity_unit_type_array) == np.max(proximity_unit_type_array) \
                or random_jump > RANDOM_CENTER_JUMP_LIMIT \
                or iteration_count > RANDOM_CENTER_ITERATION_LIMIT:
            # if we are surrounded by the same id, we are probably at the right place
            # and if we needed to jump three times maybe no better position exist
            # the last condition is just to be sure that no endless loop pop
            break
        elif np.shape(proximity_unit_type_array) != (3, 3):
            # if we are on the edges of the screen/map just jump to an other place
            returned_index = random.randint(0, len(unit_x) - 1)
            selected_coordinate_x = unit_x[returned_index]
            selected_coordinate_y = unit_y[returned_index]
            random_jump += 1
        else:
            # find the line/column having the more True value and move in that direction
            line_avg = np.sum(proximity_unit_type_array, 1, dtype=float)
            column_avg = np.sum(proximity_unit_type_array, 0, dtype=float)
            # add a malus on the center: the best is to move out !
            line_avg[1] -= 0.2
            column_avg[1] -= 0.2
            # get best average pos
            move_x = np.argmax(line_avg) - 1
            move_y = np.argmax(column_avg) - 1
            if move_x == 0 and move_y == 0:
                break
            # apply the best move found with a factor to get closer to the center (at least hoping so)
            selected_coordinate_x += move_x * RANDOM_CENTER_MOVE_FACTOR
            selected_coordinate_y += move_y * RANDOM_CENTER_MOVE_FACTOR
            # check that the new coordinate are still on the screen
            selected_coordinate_x = max(0, selected_coordinate_x)
            selected_coordinate_x = min(selected_coordinate_x, SCREEN_RESOLUTION - 1)
            selected_coordinate_y = max(0, selected_coordinate_y)
            selected_coordinate_y = min(selected_coordinate_y, SCREEN_RESOLUTION - 1)
    return selected_coordinate_x, selected_coordinate_y


# TODO: debug, the result are coherent but it does not work on pysc2...
def convolution_selection(correct_unit_type_array: np.ndarray, unit_type_id_list: list):
    diameter = None  # size of the searched unit in screen
    for type in unit_type_id_list:
        if type in ALL_MINERAL_FIELD:
            diameter = MINERAL_FIELD_TILE_SIZE * TILES_SIZE_IN_CELL

    if diameter is None:
        raise ValueError("Convolution selection method is only defined for Minerals so far."
                         "you can define new diameter in convolution_selection if you want to"
                         "use it for other unit.")

    if diameter % 2 == 0:
        diameter += 1

    xx, yy = np.mgrid[:diameter, :diameter]
    radius = diameter//2
    circle = (xx - radius) ** 2 + (yy - radius) ** 2
    circle = np.logical_and(circle <= radius**2, circle > (radius-1)**2).astype(np.int)

    # np.savetxt("tmp.txt", correct_unit_type_array, delimiter="", fmt="%1d")

    score_map = convolve2d(in1=correct_unit_type_array,
                           in2=circle,
                           mode='same')

    # np.savetxt("tmp2.txt", score_map % 10, delimiter="", fmt="%1d")
    # np.savetxt("tmp3.txt", circle, delimiter="", fmt="%1d")

    x, y = np.unravel_index(np.argmax(score_map), score_map.shape)

    return x, y

