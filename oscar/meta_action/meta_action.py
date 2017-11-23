import numpy as np
import random

from oscar.constants import *


def build(obs, building_tiles_size, building_id):
    result_action_list = [select_scv(obs)]

    # Find a valid emplacement
    building_tiles_size += 2  # Handle the free space needed around the building.
    building_cell_size = int(building_tiles_size * TILES_SIZE_IN_CELL)
    unit_type = obs.observation["screen"][UNIT_TYPE]
    height_map = obs.observation["screen"][HEIGHT_MAP]
    valid_location_center_list = _find_valid_building_location(unit_type, height_map, building_cell_size)
    if not valid_location_center_list:
        print("precision")
        valid_location_center_list = _find_valid_building_location(unit_type, height_map, building_cell_size,
                                                                   SCREEN_RESOLUTION / 20)
    if not valid_location_center_list:
        raise NoValidBuildingLocationError()
    building_location = random.choice(valid_location_center_list)
    print("building_location :", building_location)
    build_action = actions.FunctionCall(building_id, [QUEUED, building_location])
    result_action_list.insert(0, build_action)

    # Send the scv to collect resources
    result_action_list.insert(0, harvest_mineral(obs))

    return result_action_list


def _find_valid_building_location(unit_type_screen, height_map, building_size, step=SCREEN_RESOLUTION/5):
    step = int(step)
    if building_size % 2 == 0:
        building_size += 1  # A building has a unique center, so it needs an odd size.
    half_building_size = building_size // 2  # Entire division, so it is round down.

    valid_center_location = []
    map_size = len(unit_type_screen)
    # Check if the cell i, j is a good location for the center of the building
    for i in range(half_building_size, map_size - half_building_size, step):
        for j in range(half_building_size, map_size - half_building_size, step):
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
    return valid_center_location


def harvest_mineral(obs):
    unit_type = obs.observation["screen"][UNIT_TYPE]
    mineral_y, mineral_x = np.isin(unit_type, ALL_MINERAL_FIELD).nonzero()
    random_index = np.random.randint(0, len(mineral_x))
    any_mineral = (mineral_x[random_index], mineral_y[random_index])
    return actions.FunctionCall(HARVEST_GATHER_SCREEN, [NOT_QUEUED, any_mineral])


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
    unit_type = obs.observation["screen"][UNIT_TYPE]
    scv_y, scv_x = (unit_type == TERRAN_SCV).nonzero()
    command_center_y, command_center_x = (unit_type == TERRAN_COMMAND_CENTER).nonzero()
    command_center = [int(command_center_x.mean()), int(command_center_y.mean())]

    # Select a SCV collecting mineral or vespene gas
    resources_id_list = MINERAL_FIELD_LIST + VESPENE_GEYSER_LIST
    resources_y, resources_x = np.isin(unit_type, resources_id_list).nonzero()
    for scv in zip(scv_x, scv_y):
        for resource in zip(resources_x, resources_y):
            dist = np.linalg.norm(np.array(scv) - np.array(resource))
            dist += np.linalg.norm(np.array(scv) - np.array(command_center))
            if dist < MAX_COLLECTING_DISTANCE:
                return actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, scv])

    # Select an idle SCV
    if obs.observation["player"][IDLE_WORKER_COUNT] != 0:
        return actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION])

    raise NoValidSCVError()


class NoValidSCVError(RuntimeError):
    """
    Raise when no valid scv can be selected according to defined rules.
    """


class NoValidBuildingLocationError(RuntimeError):
    """
    Raise when no valid location to build a building is found in the current screen
    """
