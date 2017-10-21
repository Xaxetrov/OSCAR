import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]


class CollectMyShards(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""

    def __init__(self):
        self.state = 0
        self.visited = []
        self.marines = []
        self.marine_selected = [False, False]

    def reset(self):
        super().reset()
        self.state = 0
        self.visited = []
        self.marines = []
        self.marine_selected = [False, False]

    def step(self, obs):
        super(CollectMyShards, self).step(obs)

        # First call
        if self.state == 0:
            # Find our units and targets
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

            # Divide them into 2 groups using y median
            sorted_neutrals = sorted(zip(neutral_x, neutral_y))
            neutrals_1, neutrals_2 = numpy.array_split(numpy.asarray(sorted_neutrals), 2)
            neutrals = [neutrals_1, neutrals_2]
            # print("target subgroups : ", neutrals_1, neutrals_2)

            # Closest neighbour queue for each marine
            self.marines = list(sorted(zip(player_x, player_y)))
            for marine_idx, marine in enumerate(self.marines):
                current_position = marine
                unvisited = neutrals[marine_idx].tolist()
                self.visited.append([current_position])
                while unvisited:
                    closest, min_dist = None, None
                    for p in unvisited:
                        dist = numpy.linalg.norm(numpy.array(current_position) - numpy.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    current_position = closest
                    unvisited.remove(current_position)
                    self.visited[marine_idx].append(current_position)

            # To do when mind job is done
            self.state = 1
            self.marine_selected[0] = True
            return actions.FunctionCall(_SELECT_POINT, [[0], self.marines[0]])

        # Queuing calls
        elif self.state == 1:
            for marine_idx, marine in enumerate(self.marines):
                if self.visited[marine_idx]:
                    to_queue = self.visited[marine_idx].pop(0)
                    return actions.FunctionCall(_MOVE_SCREEN, [_QUEUED, to_queue])
                elif self.marine_selected[marine_idx]:
                    self.marine_selected[marine_idx] = False
                    try:
                        self.marine_selected[marine_idx + 1] = True
                    except IndexError:
                        self.state = 2
                        return actions.FunctionCall(_NO_OP, [])
                    else:
                        return actions.FunctionCall(_SELECT_POINT, [[0], self.marines[marine_idx + 1]])


        # Done wait for new points
        else:  # state == 2
            # Find our units and targets
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if len(neutral_y) == 20:
                self.state = 0
                return self.step(obs)
            return actions.FunctionCall(_NO_OP, [])
