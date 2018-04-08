from oscar.constants import *
from oscar.util.selection import *


def harvest_mineral(obs, queued=True):
    any_mineral = find_position(obs, ALL_MINERAL_FIELD,
                                player_relative=PLAYER_NEUTRAL,
                                select_method='random')
    return [actions.FunctionCall(HARVEST_GATHER_SCREEN, [QUEUED if queued else NOT_QUEUED, any_mineral])]