from oscar.meta_action.common import *


def select_scv(obs):
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
        resources_x, resources_y = find_position(obs, resources_id, select_method="all", player_relative=PLAYER_NEUTRAL)
    except NoUnitError:
        return select_idle_scv(obs)

    best_scv = None
    best_distance = None
    for scv in zip(scv_x, scv_y):
        for resource in zip(resources_x, resources_y):
            distance = np.linalg.norm(np.array(scv) - np.array(resource))
            distance += np.linalg.norm(np.array(scv) - np.array(command_center))
            if distance < MAX_COLLECTING_DISTANCE and (best_scv is None or distance < best_distance):
                best_scv = scv
                best_distance = distance
    if best_scv is not None:
        return [actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, best_scv])]

    return select_idle_scv(obs)


def select_idle_scv(obs):
    # Select an idle SCV
    if obs.observation["player"][IDLE_WORKER_COUNT] != 0:
        return [actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION])]
    raise NoValidSCVError()
