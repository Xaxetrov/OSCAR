# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy
import sys

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


class MoveToBeacon(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])
            target = [int(neutral_x.mean()), int(neutral_y.mean())]
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])


class CollectMineralShards(base_agent.BaseAgent):
    """An agent specifically for solving the CollectMineralShards map."""

    def step(self, obs):
        super(CollectMineralShards, self).step(obs)

        # If a unit is selected and can move
        if _MOVE_SCREEN in obs.observation["available_actions"]:

            # Find our units and targets
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

            # If there is no target or no unit do nothing
            if not neutral_y.any() or not player_y.any():
                return actions.FunctionCall(_NO_OP, [])

            #Our position is the mean of friendly units positions
            player = [int(player_x.mean()), int(player_y.mean())]

            # Find closest target
            closest, min_dist = None, None
            for p in zip(neutral_x, neutral_y):
                dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist

            # Order selected units to move to it
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])

        # If no unit is selected
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

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

        #First call
        if self.state == 0:
            # Find our units and targets
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

            #Divide them into 2 groups using y median
            sorted_neutrals = sorted(zip(neutral_x,neutral_y))
            neutrals_1, neutrals_2 = numpy.array_split(numpy.asarray(sorted_neutrals), 2)
            neutrals = [neutrals_1, neutrals_2]
            #print("target subgroups : ", neutrals_1, neutrals_2)

            #Closest neighbour queue for each marine
            self.marines = list(zip(player_x, player_y))
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

            #To do when mind job is done
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
                        self.marine_selected[marine_idx+1] = True
                    except IndexError:
                        return actions.FunctionCall(_NO_OP, [])
                    else:
                        return actions.FunctionCall(_SELECT_POINT, [[0], self.marines[marine_idx+1]])

            self.state = 2
            return actions.FunctionCall(_NO_OP, [])

        # Done wait for new points
        else: #state == 2
            return actions.FunctionCall(_NO_OP, [])
