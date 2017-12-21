from oscar.util.selection import *


def train_unit(obs, building_id, action_train_id):
    command_center = find_position(obs, building_id)
    return [actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, command_center]),
            actions.FunctionCall(action_train_id, [NOT_QUEUED])]
