from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions
from pysc2.lib import features
from gym import spaces
import numpy as np
from collections import deque

from oscar.env.envs.pysc2_env import Pysc2Env
from oscar.constants import *
from oscar import meta_action

OBS_LENGTH = 4


class Pysc2Simple64MetaEnv(Pysc2Env):
    metadata = {'render.modes': ['human']}

    action_space = spaces.Discrete(9)  # two 16*16 output: first move second selection
    observation_space = spaces.Box(low=0, high=float('Inf'), shape=(3 * OBS_LENGTH + 4, 64, 64))
    reward_range = [0.0, float('Inf')]

    def __init__(self):
        self.pysc2_env = SC2Env(  # map_name='CollectMineralsAndGas',
                                map_name='Simple64',
                                agent_race='T',
                                screen_size_px=(64, 64),
                                minimap_size_px=(64, 64),
                                visualize=True,
                                step_mul=16,
                                game_steps_per_episode=None  # use map default
                                )
        self.obs_list = deque()
        self.action_list = deque()
        self.total_reward = 0
        self.last_army_count = 0
        self.step_reward = 0
        super().__init__()

    def _step(self, action):
        self.action_list += self.get_meta_action(action)

        while True:
            formatted_action = self.action_list.popleft()
            if formatted_action.function not in self.last_obs.observation["available_actions"]:
                self.action_list.clear()
                formatted_action = actions.FunctionCall(NO_OP, [])
            # call mother class to run action in SC2
            full_obs, reward, done, debug_dic = super()._step([formatted_action])
            if len(self.action_list) == 0 or done:
                self.action_list.clear()
                break
        self.update_reward()
        return self.get_return_obs(full_obs), self.step_reward, done, debug_dic

    def _reset(self):
        self.total_reward = 0
        # call mother class to reset env
        full_obs = super()._reset()
        # format observation to be the one corresponding to observation_space
        obs = self.format_observation(full_obs)
        self.obs_list = deque()
        while len(self.obs_list) < OBS_LENGTH * len(obs):
            self.obs_list += obs
        return self.get_return_obs(full_obs)

    def _render(self, mode='human', close=False):
        super()._render(mode, close)

    """
        Methods used for the translation of Gym's action to pysc2's action
    """
    def get_meta_action(self, action_id):
        try:
            if action_id == 1:  # build supply
                return meta_action.build(self.last_obs,
                                         building_tiles_size=2,
                                         building_id=BUILD_SUPPLY_DEPOT)
            elif action_id == 2:  # build Barracks
                return meta_action.build(self.last_obs,
                                         building_tiles_size=3,
                                         building_id=BUILD_BARRACKS)
            elif action_id == 3:  # select IDLE SCV
                pass
                # return meta_action.select_idle_scv(self.last_obs)
            elif action_id == 4:  # harvest mineral
                return meta_action.harvest_mineral(self.last_obs, queued=True)
            elif action_id == 5:  # train Marines
                return meta_action.train_unit(self.last_obs,
                                              building_id=TERRAN_BARRACKS_ID,
                                              action_train_id=TRAIN_MARINE_QUICK)
            elif action_id == 6:  # train SCV
                return meta_action.train_unit(self.last_obs,
                                              building_id=TERRAN_COMMAND_CENTER,
                                              action_train_id=TRAIN_SCV_QUICK)
            elif action_id == 7:  # select army
                return [actions.FunctionCall(SELECT_ARMY, [SELECT_ALL])]
            elif action_id == 8:  # attack !
                return meta_action.attack_minimap(self.last_obs, queued=False)
        except meta_action.NoValidSCVError:
            pass
        except meta_action.NoValidBuildingLocationError:
            pass
        except meta_action.NoUnitError:
            pass

        return [actions.FunctionCall(NO_OP, [])]

    def get_return_obs(self, currant_obs):
        formatted_obs = self.format_observation(currant_obs)
        self.add_to_obs_list(formatted_obs)
        ret_obs_list = deque()
        ret_obs_list.append(np.full(shape=(64, 64), fill_value=currant_obs.observation['player'][MINERALS]))
        ret_obs_list.append(np.full(shape=(64, 64), fill_value=currant_obs.observation['player'][VESPENE]))
        ret_obs_list.append(np.full(shape=(64, 64), fill_value=currant_obs.observation['player'][FOOD_USED]))
        ret_obs_list.append(np.full(shape=(64, 64), fill_value=currant_obs.observation['player'][FOOD_CAP]))
        ret_obs_list += self.obs_list
        return np.array(ret_obs_list, copy=True)

    @staticmethod
    def format_observation(obs):
        unit_type = obs.observation["screen"][SCREEN_UNIT_TYPE]
        player_relative = obs.observation["screen"][SCREEN_PLAYER_RELATIVE]
        selected = obs.observation["screen"][SCREEN_SELECTED]
        formatted_obs = [player_relative,
                         selected, unit_type]
        return formatted_obs

    def add_to_obs_list(self, obs):
        # slide the fil with new observation
        for o in obs:
            self.obs_list.popleft()
            self.obs_list.append(o)

    def get_action_mask(self):
        return np.ones(shape=(self.action_space.n,))
        # raise NotImplementedError("action mask is not implemented for meta action")

    @staticmethod
    def get_action_id_from_action(sc2_action, sc2_args):
        raise NotImplementedError("conversion from sc2 action to env action is not implemented for meta action")

    def update_reward(self):
        new_total_reward = 0
        new_total_reward += self.last_obs.observation['score_cumulative'][5]
        new_total_reward += self.last_obs.observation['score_cumulative'][6]
        self.step_reward = new_total_reward - self.total_reward
        self.total_reward = new_total_reward
        army_count = self.last_obs.observation['player'][ARMY_COUNT]
        self.step_reward += max(0, army_count - self.last_army_count)
        self.last_army_count = army_count




