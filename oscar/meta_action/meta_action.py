import numpy as np
import random

from pysc2.lib import actions
from pysc2.lib import features

from oscar.constants import *

# Functions
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
_IDLE_WORKER_COUNT = 7

# Units ID
_TERRAN_COMMAND_CENTER = 18
_TERRAN_SCV = 45
_MINERAL_FIELD_LIST = (341, 483, 146, 147)
_VESPENE_GEYSER_LIST = (344, 342)

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [False]
_QUEUED = [True]
_NEW_SELECTION = [0]

# Others
_MAX_COLLECTING_DISTANCE = SCREEN_RESOLUTION / 3
_BUILDING_TILES_SIZE = SCREEN_RESOLUTION / 21


def build(obs, building_tiles_size):
    result_action_list = [select_scv(obs)]

    # Find a valid emplacement
    building_tiles_size += 2  # Handle the free space needed around the building.
    building_size = int(building_tiles_size * _BUILDING_TILES_SIZE)
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    height_map = obs.observation["screen"][_HEIGHT_MAP]
    valid_location_center_list = _find_valid_building_location(unit_type, height_map, building_size)
    if not valid_location_center_list:
        valid_location_center_list = _find_valid_building_location(unit_type, height_map, building_size,
                                                                  SCREEN_RESOLUTION / 20)
    if not valid_location_center_list:
        raise NoValidBuildingLocationError()
    building_location = random.choice(valid_location_center_list)
    print("building_location :", building_location)

    return result_action_list


def _find_valid_building_location(unit_type_screen, height_map, building_size, step=SCREEN_RESOLUTION/5):
    step = int(step)
    if building_size % 2 == 0:
        building_size += 1  # A building has a unique center, so it needs an odd size.
    half_building_size = building_size // 2  # Entire division, so it is round down.

    valid_center_location = []
    map_size = len(unit_type_screen)
    # Check if the cell i, j is a good location for the center of the building
    i = half_building_size
    while i < map_size - half_building_size:
        j = half_building_size
        while j < map_size - half_building_size:
            center_location_is_valid = True
            for k in range(-half_building_size, half_building_size):
                if not center_location_is_valid:
                    break
                for l in range(-half_building_size, half_building_size):
                    if unit_type_screen[i + k][j + l] != 0:
                        center_location_is_valid = False
                        break
            if center_location_is_valid and _check_map_height_for_building(height_map, building_size, (i, j)):
                valid_center_location.append((i, j))
            j += step
        i += step
    return valid_center_location


def _check_map_height_for_building(height_map, building_size, potential_center_location):
    if building_size % 2:
        building_size += 1
    half_size_building = building_size // 2

    center_row, center_col = potential_center_location
    for i in range(center_row - half_size_building, center_row + half_size_building):
        for j in range(center_col - half_size_building, center_col + half_size_building):
            if height_map[i][j] != height_map[center_row][center_col]:
                return False
    return True


def select_scv(obs):
    """
    Select a SCV, with priority order: first, try to select a SCV collecting resources (mineral or vespene gas).
    This will only try to select a SCV on screen. Then, if an idle SCV exist, select it (not necessarily on
    screen). If none of the previous actions works, raise a NoValidSCVError.
    :param obs: Observations of the current step.
    :param queued: Whether the action should be queued or not.
    :return: The action to execute to select a SCV.
    """
    unit_type = obs.observation["screen"][_UNIT_TYPE]
    scv_y, scv_x = (unit_type == _TERRAN_SCV).nonzero()
    command_center_y, command_center_x = (unit_type == _TERRAN_COMMAND_CENTER).nonzero()
    command_center = [int(command_center_x.mean()), int(command_center_y.mean())]

    # Select a SCV collecting mineral or vespene gas
    resources_id_list = _MINERAL_FIELD_LIST + _VESPENE_GEYSER_LIST
    resources_y, resources_x = np.isin(unit_type, resources_id_list).nonzero()
    for scv in zip(scv_x, scv_y):
        for resource in zip(resources_x, resources_y):
            dist = np.linalg.norm(np.array(scv) - np.array(resource))
            dist += np.linalg.norm(np.array(scv) - np.array(command_center))
            if dist < _MAX_COLLECTING_DISTANCE:
                print("target select :", scv)
                return actions.FunctionCall(_SELECT_POINT, [_NEW_SELECTION, scv])

    # Select an idle SCV
    if obs.observation["player"][_IDLE_WORKER_COUNT] != 0:
        return actions.FunctionCall(_SELECT_IDLE_WORKER, [0])  # TODO: wtf s the arg

    raise NoValidSCVError()


class NoValidSCVError(RuntimeError):
    """
    Raise when no valid scv can be selected according to defined rules.
    """


class NoValidBuildingLocationError(RuntimeError):
    """
    Raise when no valid location to build a building is found in the current screen
    """
