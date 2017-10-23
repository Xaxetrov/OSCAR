from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions
from pysc2.lib import features
from gym import spaces
import gflags as flags
import sys
import numpy as np
import time

from oscar_env.envs.pysc2_env import Pysc2Env

"""API Constants"""
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class Pysc2MineralshardEnv(Pysc2Env):
    metadata = {'render.modes': ['human']}

    # action_space = spaces.Dict({"non-spacial": spaces.Discrete(3),
    #                             "spacial": spaces.Discrete(16 * 16)}
    #                            )
    action_space = spaces.Discrete(0 + 16 * 16)
    observation_space = spaces.Box(low=0, high=4, shape=(64, 64, 2))
    reward_range = [0.0, float('Inf')]

    def __init__(self):
        self.pysc2_env = SC2Env(map_name='CollectMineralShards',
                                agent_race='T',
                                screen_size_px=(64, 64),
                                minimap_size_px=(64, 64),
                                visualize=False,
                                step_mul=8,
                                game_steps_per_episode=None  # use map default
                                )
        super().__init__()

    def _step(self, action):
        if _MOVE_SCREEN in self.last_obs.observation["available_actions"]:
                # and action > 1:
                # and action["non-spacial"] == 2:
                sc2_action = self.get_move_action(action)
                # sc2_action = self.get_move_action(action["spacial"])
            # print("move pos", action["spacial"])
        # elif action < 1:
        #     # here two case: whether we have action id 256 or 257 or we have 0 or 1
        #     # in both case if _SELECT_ARMY is not available the following call handles it
        #     # sc2_action = self.get_non_spacial_action(action["non-spacial"])
        #     sc2_action = self.get_non_spacial_action(action)
        else:
            # else set NO OP...
            # help for NN -> automated selection
            sc2_action = self.get_non_spacial_action(1)
        # print(sc2_action)
        formatted_action = actions.FunctionCall(sc2_action[0], sc2_action[1])
        # call mother class to run action in SC2
        full_obs, reward, done, debug_dic = super()._step([formatted_action])
        # format observation to be the one corresponding to observation_space
        obs = self.format_observation(full_obs)
        return obs, reward, done, debug_dic

    def _reset(self):
        # call mother class to reset env
        full_obs = super()._reset()
        # format observation to be the one corresponding to observation_space
        obs = self.format_observation(full_obs)
        return obs

    def _render(self, mode='human', close=False):
        super()._render(mode, close)

    """
        Methods used for the translation of Gym's action to pysc2's action
    """
    @staticmethod
    def get_move_action(linear_position):
        """return a pysc2 action and argument to do a move action at the pos given
            -linear_position : position of the move on a 16x16 grid, integer equal to y*16+x
            """
        x_16 = (linear_position // 16)
        y_16 = (linear_position % 16)
        x_true = min(x_16 * 4, 63)
        y_true = min(y_16 * 4, 63)
        # x and y are not in the right order, else it doesn't work...
        action_args = [_NOT_QUEUED, [y_true, x_true]]
        return _MOVE_SCREEN, action_args

    @staticmethod
    def get_non_spacial_action(action_id):
        """return a pysc2 action coresponding to the given action id
            -action id: 0 -> NO_OP
                        1 -> Select all army
        """
        if action_id == 1:
            selected_action = _SELECT_ARMY
            action_args = [_SELECT_ALL]
        else:
            selected_action = _NO_OP
            action_args = []
        return selected_action, action_args

    @staticmethod
    def format_observation(obs):
        a = features.SCREEN_FEATURES
        player_relative = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
        selected = obs.observation["screen"][features.SCREEN_FEATURES.selected.index]
        formatted_obs = np.zeros(shape=(64, 64, 2))
        for formatted_row, pr_row, selected_row in zip(formatted_obs, player_relative, selected):
            for formatted_case, pr_case, selected_case in zip(formatted_row, pr_row, selected_row):
                formatted_case[0] = pr_case
                formatted_case[1] = selected_case
        return formatted_obs
