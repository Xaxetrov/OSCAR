import numpy
import sys
import time
from random import randint

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ALL = [0]

_SCREEN = [0]
_NOT_QUEUED = [0]


class CollectMineralShardsBaseAgent(base_agent.BaseAgent):
    def reset(self):
        super().reset()
        print(self.reward)
        self.reward = 0


class CollectMineralShardsRandom(CollectMineralShardsBaseAgent):

    def step(self, obs):
        super().step(obs)
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            index_max_action = 2
        else:
            index_max_action = 1

        selected_action = randint(0, index_max_action)
        if selected_action == 0:
            return actions.FunctionCall(_NO_OP, [])
        elif selected_action == 1:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        else:
            move_x = randint(0, 83)
            move_y = randint(0, 83)
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, (move_x, move_y)])


class CollectMineralShardsCommonGlutton(CollectMineralShardsBaseAgent):

    def __init__(self):
        super().__init__()
        self.__last_position_marine_1 = None
        self.__last_position_marine_2 = None
        self.__marine_1_selected = False
        self.__marine_2_selected = False
        self.__current_target = []

    def reset(self):
        super().reset()
        self.__last_position_marine_1 = None
        self.__last_position_marine_2 = None
        self.__marine_1_selected = False
        self.__marine_2_selected = False
        self.__current_target = []

    def step(self, obs):
        time.sleep(0.5)
        super().step(obs)

        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

        (current_position_marine_1, current_position_marine_2) = self.__get_marines_position(player_x, player_y)
        if self.__last_position_marine_1 is None and self.__last_position_marine_2 is None:
            marine_1_is_moving = False
            marine_2_is_moving = False
        else:
            marine_1_is_moving = not numpy.array_equal(current_position_marine_1, self.__last_position_marine_1)
            marine_2_is_moving = not numpy.array_equal(current_position_marine_2, self.__last_position_marine_2)
        self.__last_position_marine_1 = current_position_marine_1
        self.__last_position_marine_2 = current_position_marine_2

        if not marine_1_is_moving:
            if not self.__marine_1_selected:
                self.__marine_1_selected = True
                self.__marine_2_selected = False
                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, current_position_marine_1])
            else:
                target = self.__find_target(current_position_marine_1, zip(neutral_x, neutral_y))
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        elif not marine_2_is_moving:
            if not self.__marine_2_selected:
                self.__marine_1_selected = False
                self.__marine_2_selected = True
                return actions.FunctionCall(_SELECT_POINT, [_SCREEN, current_position_marine_2])
            else:
                target = self.__find_target(current_position_marine_2, zip(neutral_x, neutral_y))
                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
        else:
            return actions.FunctionCall(_NO_OP, [])

        #
        #
        #
        # if _MOVE_SCREEN not in obs.observation["available_actions"]:
        #     print("ACTIONS : SELECT", flush=True)
        #     return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        #
        #
        #
        # player = [int(player_x.mean()), int(player_y.mean())]
        # closest = None
        # min_dist = 1e9
        # for shard in zip(neutral_x, neutral_y):
        #     dist = numpy.linalg.norm(numpy.array(player) - shard)
        #     if dist < min_dist:
        #         min_dist = dist
        #         closest = shard
        # print("ACTION : MOVE", flush=True)
        # return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])

    def __get_marines_position(self, player_x, player_y):
        if len(player_x) == 1:  # both marines are at the same position
            marine_position = numpy.array([player_x[0], player_y[0]])
            return (marine_position, marine_position)
        if self.__last_position_marine_1 is None:
            marine_1 = numpy.array([player_x[0], player_y[0]])
            marine_2 = numpy.array([player_x[1], player_y[1]])
            return (marine_1, marine_2)

        marine_index_0 = numpy.array([player_x[0], player_y[0]])
        marine_index_1 = numpy.array([player_x[1], player_y[1]])
        dist_marine_1_index_0 = numpy.linalg.norm(marine_index_0 - self.__last_position_marine_1)
        dist_marine_1_index_1 = numpy.linalg.norm(marine_index_1 - self.__last_position_marine_1)
        marine_1 = marine_index_0 if dist_marine_1_index_0 < dist_marine_1_index_1 else marine_index_1
        marine_2 = marine_index_0 if dist_marine_1_index_0 >= dist_marine_1_index_1 else marine_index_1
        return (marine_1, marine_2)

    def __find_target(self, marine, shards_list):
        closest = None
        min_dist = 1e9
        for shard in shards_list:
            if shard not in self.__current_target:
                dist = numpy.linalg.norm(numpy.array(marine) - shard)
                if dist < min_dist:
                    min_dist = dist
                    closest = shard
        self.__current_target.append(closest)
        return closest






















