import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_PLAYER_SELECTED = features.SCREEN_FEATURES.selected.index
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_NEW_SELECTION = [0]


class CollectMineralShards(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""

    def step(self, obs):
        super(CollectMineralShards, self).step(obs)
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            selected = obs.observation["screen"][_PLAYER_SELECTED]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (selected == 1).nonzero()
            if not neutral_y.any() or not player_y.any():
                return actions.FunctionCall(_NO_OP, [])
            player = [int(player_x.mean()), int(player_y.mean())]
            closest, min_dist = None, None
            for p in zip(neutral_x, neutral_y):
                dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
        else:
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            return actions.FunctionCall(_SELECT_POINT, [_NEW_SELECTION, [player_x[0], player_y[0]]])


MOVE_REPEAT = 1


class CollectMineralShardsP(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""

    def __init__(self):
        super().__init__()
        self.action = 'move'
        self.move_count = 0

    def step(self, obs):
        super(CollectMineralShardsP, self).step(obs)
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            selected = obs.observation["screen"][_PLAYER_SELECTED]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (selected == 1).nonzero()
            if self.action == 'move':
                # move the selected marines
                self.move_count += 1
                if self.move_count == MOVE_REPEAT:
                    self.action = 'select_other'  # set the next action to be a selection
                    self.move_count = 0
                if not neutral_y.any() or not player_y.any():
                    return actions.FunctionCall(_NO_OP, [])
                player = [int(player_x.mean()), int(player_y.mean())]
                closest, min_dist = None, None
                for p in zip(neutral_x, neutral_y):
                    dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                    if not min_dist or dist < min_dist:
                        closest, min_dist = p, dist
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
            else:
                # select the other marines
                self.action = 'move'  # set the next action to be a move
                not_select_py, not_select_px = ((player_relative - selected) == 1).nonzero()
                if len(not_select_py) > 0:
                    p = [int(not_select_px.mean()), int(not_select_py.mean())]
                    return actions.FunctionCall(_SELECT_POINT, [_NEW_SELECTION, p])
                # if both of them are selected, select one
                player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                return actions.FunctionCall(_SELECT_POINT, [_NEW_SELECTION, [player_x[0], player_y[0]]])
        else:
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            return actions.FunctionCall(_SELECT_POINT, [_NEW_SELECTION, [player_x[0], player_y[0]]])

    def reset(self):
        # print("reward: ", self.reward)
        # self.reward = 0
        super().reset()
