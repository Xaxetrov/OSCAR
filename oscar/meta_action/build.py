from oscar.meta_action.harvest import *
from oscar.meta_action.select import *
from oscar.constants import *
from oscar.shared.screen import Screen

from scipy.signal import convolve2d


def build(obs, building_tiles_size, building_id, propagate_error=True):
    result_action_list = None
    try:
        result_action_list = select_idle_scv_screen_priority(obs)
    except NoValidSCVError:
        try:
            result_action_list = select_scv_on_screen(obs)
        except NoValidSCVError:
            if propagate_error:
                raise

    # Find a valid emplacement
    building_tiles_size += 0  # Handle the free space needed around the building.
    building_cell_size = int(building_tiles_size * TILES_SIZE_IN_CELL)
    unit_type = obs.observation["screen"][SCREEN_UNIT_TYPE]
    height_map = obs.observation["screen"][HEIGHT_MAP]
    valid_point_center_list = find_valid_building_point(unit_type, height_map, building_cell_size)
    if not valid_point_center_list:
        valid_point_center_list = find_valid_building_point(unit_type, height_map, building_cell_size,
                                                            SCREEN_RESOLUTION / 20)
        if not valid_point_center_list:
            raise NoValidBuildingPointError()
    building_point = random.choice(valid_point_center_list)
    building_point = (building_point[1], building_point[0])
    build_action = actions.FunctionCall(building_id, [NOT_QUEUED, building_point])
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


def get_random_building_point(obs, shared, building_size, samples=20):
    half_building_size = int(building_size/2)

    for i in range(samples):
        center = shared['screen'].random_point(obs, half_building_size)
        is_valid = True

        screen_unit_type = obs.observation["screen"][SCREEN_UNIT_TYPE]
        height_map = obs.observation["screen"][HEIGHT_MAP]

        for y in range(center.y - half_building_size, center.y + half_building_size+1):
            for x in range(center.x - half_building_size, center.x + half_building_size+1):
                if screen_unit_type[x, y] != 0 or height_map[x, y] != height_map[center.x, center.y]:
                    is_valid = False
                    break
            if not is_valid:
                break

        if is_valid:
            return center


def find_valid_building_point(unit_type_screen, height_map, building_size, step=1):
    # step = int(step)
    if building_size % 2 == 0:
        building_size += 1  # A building has a unique center, so it needs an odd size.
    half_building_size = int(building_size // 2)  # Entire division, it rounds down.
    #
    # valid_center_point = []
    # screen_size = len(unit_type_screen)
    # Check if the cell i, j is a good point for the center of the building
    # for i in range(half_building_size, screen_size - half_building_size + 1, step):
    #     for j in range(half_building_size, screen_size - half_building_size + 1, step):
    #         building_centered_unit_type_screen = unit_type_screen[i - half_building_size:i + half_building_size + 1,
    #                                                               j - half_building_size:j + half_building_size + 1]
    #         if np.max(building_centered_unit_type_screen) == 0 and \
    #                 _check_map_height_for_building(height_map, building_size, (i, j)):
    #             valid_center_point.append((i, j))
    # return valid_center_point

    # convolution version: (about much more quicker)
    building_size += 4  # increase building size to prevent them to block units
    building_patch = np.ones(shape=(building_size, building_size))
    border_x = (height_map[:-1, :] - height_map[1:, :]) != 0
    border_y = (height_map[:, :-1] - height_map[:, 1:]) != 0
    # obstacles are unit, change of height or 0 height
    obstacle_map = unit_type_screen[:-1, :-1] + border_x[:, :-1].astype(np.int) + border_y[:-1, :].astype(np.int) \
                   + (height_map[:-1, :-1] == 0)
    score = convolve2d(obstacle_map, building_patch, mode='same')
    # select point where no obstacle are found
    yy, xx = np.where(score == 0)
    # better to only select edges of the score ? (and so build on the edges of posible position
    # and not anywhere in the middle ?)
    return list(zip(xx, yy))


def _check_map_height_for_building(height_map, building_size, potential_center_point):
    if building_size % 2 == 0:
        building_size += 1
    half_building_size = int(building_size // 2)

    (i, j) = potential_center_point

    building_centered_height_map = height_map[i - half_building_size:i + half_building_size + 1,
                                              j - half_building_size:j + half_building_size + 1]
    return np.max(building_centered_height_map) == np.min(building_centered_height_map)
