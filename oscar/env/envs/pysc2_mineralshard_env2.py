from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions
from pysc2.lib import features
from gym import spaces
import numpy as np
import copy

from oscar.env.envs.pysc2_env import Pysc2Env

"""API Constants"""
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_ADD_TO_SELECTION = [1]
_NEW_SELECTION = [0]

OBS_LENGTH = 4


class Pysc2MineralshardEnv2(Pysc2Env):
    metadata = {'render.modes': ['human']}

    # action_space = spaces.Dict({"non-spacial": spaces.Discrete(3),
    #                             "spacial": spaces.Discrete(16 * 16)}
    #                            )
    action_space = spaces.Discrete(0 + 16 * 16 * 2)  # two 16*16 output: first move second selection
    observation_space = spaces.Box(low=0, high=4, shape=(2 * OBS_LENGTH, 64, 64))
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
        self.obs_list = []
        super().__init__()

    def _step(self, action):
        if _MOVE_SCREEN in self.last_obs.observation["available_actions"] \
                and action < 256:
            sc2_action = self.get_move_action(action)
        elif action >= 256:
            # do a spacial selection
            sc2_action = self.get_select_action(action - 256)
        else:
            # else set NO OP...
            print("WARNING: action", action, "is a valid action")
            sc2_action = self.get_non_spacial_action(0)
        formatted_action = actions.FunctionCall(sc2_action[0], sc2_action[1])
        # call mother class to run action in SC2
        full_obs, reward, done, debug_dic = super()._step([formatted_action])
        # format observation to be the one corresponding to observation_space
        obs = self.format_observation(full_obs)
        # update obs list
        self.add_to_obs_list(obs)
        return np.array(self.obs_list, dtype=int, copy=True), reward, done, debug_dic

    def _reset(self):
        # call mother class to reset env
        full_obs = super()._reset()
        # format observation to be the one corresponding to observation_space
        obs = self.format_observation(full_obs)
        self.obs_list = []
        while len(self.obs_list) < OBS_LENGTH * len(obs):
            self.obs_list += obs
        # select all marines at first step
        # formatted_action = actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
        # full_obs, _, _, _ = super()._step([formatted_action])
        # self.add_to_obs_list(self.format_observation(full_obs))
        return np.array(self.obs_list, dtype=int, copy=True)

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
        x_16 = (linear_position % 16)
        y_16 = (linear_position // 16)
        x_64 = x_16 * 4 + 2  # + 2 to center on the 64x64 grid after the 16x16->64x64 conversion
        y_64 = y_16 * 4 + 2
        # print("Movement at x16:", x_16, "y16", y_16)
        # x and y are not in the right order, else it doesn't work...
        action_args = [_NOT_QUEUED, [x_64, y_64]]
        return _MOVE_SCREEN, action_args

    @staticmethod
    def get_select_action(linear_position):
        """return a pysc2 action and argument to do a move action at the pos given
            -linear_position : position of the move on a 16x16 grid, integer equal to y*16+x
            """
        x_16 = (linear_position // 16)
        y_16 = (linear_position % 16)
        x_64 = x_16 * 4
        y_64 = y_16 * 4
        # print("selection at x16:", x_16, "y16", y_16)
        # x and y are not in the right order, else it doesn't work...
        action_args = [_NEW_SELECTION, [y_64, x_64], [y_64 + 3, x_64 + 3]]
        return _SELECT_RECT, action_args

    @staticmethod
    def get_non_spacial_action(action_id):
        """return a pysc2 action corresponding to the given action id
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
        player_relative = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
        selected = obs.observation["screen"][features.SCREEN_FEATURES.selected.index]
        formatted_obs = [player_relative,
                         selected]
        return formatted_obs

    def add_to_obs_list(self, obs):
        # slide the fil with new observation
        for o in obs:
            self.obs_list.pop(0)
            self.obs_list.append(o)

    def get_action_mask(self):
        available_actions = self.last_obs.observation["available_actions"]

        # mask everything
        action_mask = np.zeros(shape=self.action_space.n, dtype=int)

        # unmask available action
        if _SELECT_RECT in available_actions:
            action_mask[256:] = 1
        if _MOVE_SCREEN in available_actions:
            action_mask[:256] = 1

        return action_mask

    @staticmethod
    def get_action_id_from_action(sc2_action, sc2_args):
        if sc2_action == _SELECT_POINT and len(sc2_args) == 2:
            y_64, x_64 = sc2_args[1]
            x_16 = x_64 // 4
            y_16 = y_64 // 4
            return int(16 * x_16 + y_16 + 256)
        elif sc2_action == _MOVE_SCREEN and len(sc2_args) == 2:
                x_64, y_64 = sc2_args[1]
                x_16 = x_64 // 4
                y_16 = y_64 // 4
                return int(16 * y_16 + x_16)
        else:
            print("unhandled action: ", sc2_action, sc2_args)
            return None
