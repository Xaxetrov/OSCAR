from oscar.meta_action.move import *
from oscar.meta_action.select import *
from oscar.constants import *


def build(obs, building_tiles_size, building_id, propagate_error=False):
    result_action_list = select_scv(obs)

    # Find a valid emplacement
    building_tiles_size += 0  # Handle the free space needed around the building.
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
    building_location = (building_location[1], building_location[0])
    build_action = actions.FunctionCall(building_id, [NOT_QUEUED, building_location])
    result_action_list.append(build_action)

    # Send the scv back to collect resources
    try:
        result_action_list += harvest_mineral(obs)
    except NoUnitError:
        # If propagate_error is False, the SCV will be idle at the end of the build action.
        # Another agent will need to sent it back to work.
        if propagate_error:
            raise

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
