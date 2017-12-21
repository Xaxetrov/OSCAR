from random import randint
from oscar.util.selection import *
from oscar.util.screen_helper import *

""" 
Selects an idle scv, using built-in pySc2 function. 
Priority is given to scv visible on screen.
"""
def select_idle_scv_screen_priority(obs):
    """
    Select a SCV, with priority order: first, try to select a SCV collecting resources (mineral or vespene gas).
    This will only try to select a SCV on screen. Then, if an idle SCV exist, select it (not necessarily on
    screen). If none of the previous actions works, raise a NoValidSCVError.
    :param obs: Observations of the current step.
    :return: The action to execute to select a SCV.
    """
    scv_x, scv_y = find_position(obs, TERRAN_SCV, select_method="all", exception=NoValidSCVError)
    try:
        command_center = find_position(obs, TERRAN_COMMAND_CENTER, select_method="mean")
    except NoUnitError:
        return select_idle_scv(obs)

    resources_id = ALL_MINERAL_FIELD + ALL_VESPENE_GEYSER
    try:
        resource = find_position(obs, resources_id, select_method="mean", player_relative=PLAYER_NEUTRAL)
    except NoUnitError:
        return select_idle_scv(obs)

    best_scv = None
    best_distance = None
    for scv in zip(scv_x, scv_y):
        distance = np.linalg.norm(np.array(scv) - np.array(resource))
        distance += np.linalg.norm(np.array(scv) - np.array(command_center))
        if distance < MAX_COLLECTING_DISTANCE and (best_scv is None or distance < best_distance):
            best_scv = scv
            best_distance = distance
    if best_scv is not None:
        return [actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, best_scv])]

    return select_idle_scv(obs)

""" Selects an idle scv, using built-in pySc2 function. """
def select_idle_scv(obs):
    # Select an idle SCV
    if obs.observation["player"][IDLE_WORKER_COUNT] != 0:
        return [actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION])]
    raise NoValidSCVError()


""" Selects a random scv on screen, using screen layers. """
def select_scv_on_screen(obs):
    centers = get_center(obs, TERRAN_SCV, PLAYER_SELF)
    if len(centers) == 0:
        raise NoValidSCVError()

    # Picks random scv
    pos = list(centers[randint(0, len(centers)-1)])

    # Handles scv partially out of the screen
    if pos[0] < 0:
        pos[0] = 0
    elif pos[0] >= SCREEN_RESOLUTION:
        pos[0] = SCREEN_RESOLUTION-1
    if pos[1] < 0:
        pos[1] = 0
    elif pos[1] >= SCREEN_RESOLUTION:
        pos[1] = SCREEN_RESOLUTION-1

    return [actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, pos])]