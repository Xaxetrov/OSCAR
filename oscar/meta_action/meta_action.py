import numpy as np

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
_IDLE_WORKER_COUNT = 7

# Units ID
_TERRAN_COMMAND_CENTER = 18
_TERRAN_SCV = 45
_MINERAL_FIELD_LIST = (341, 483, 146, 147)
_VESPENE_GEYSER_LIST = (344, 342)

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]

#
_MAX_COLLECTING_DISTANCE = SCREEN_RESOLUTION / 3


def build_barracks():
    pass


def select_scv(obs):
    """
    Select a SCV, with priority order: first, try to select a SCV collecting resources (mineral or vespene gas).
    This will only try to select a SCV on screen. Then, if an idle SCV exist, select it (not necessarily on
    screen). If none of the previous actions works, raise a NoValidSCVError.
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
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, scv])

    # Select an idle SCV
    if obs.observation["player"][_IDLE_WORKER_COUNT] != 0:
        return actions.FunctionCall(_SELECT_IDLE_WORKER, [])

    raise NoValidSCVError()


class NoValidSCVError(RuntimeError):
    """
    Raise when no valid scv can be selected according to defined rules.
    """
