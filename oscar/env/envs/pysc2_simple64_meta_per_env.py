from oscar.env.envs.pysc2_simple64_meta_env import Pysc2Simple64MetaEnv
from oscar.constants import *

from gym import spaces
import numpy as np
from collections import deque


class Pysc2Simple64MetaPerEnv(Pysc2Simple64MetaEnv):

    observation_space = spaces.Box(low=0, high=float('Inf'), shape=(10,))

    def get_return_obs(self, currant_obs):
        unit_type = currant_obs.observation["screen"][SCREEN_UNIT_TYPE]
        ret_obs_list = deque()
        ret_obs_list.append(currant_obs.observation['player'][MINERALS])
        ret_obs_list.append(currant_obs.observation['player'][VESPENE])
        ret_obs_list.append(currant_obs.observation['player'][FOOD_USED])
        ret_obs_list.append(currant_obs.observation['player'][FOOD_USED_BY_ARMY])
        ret_obs_list.append(currant_obs.observation['player'][FOOD_USED_BY_WORKERS])
        ret_obs_list.append(currant_obs.observation['player'][FOOD_CAP])
        ret_obs_list.append(currant_obs.observation['player'][ARMY_COUNT])
        ret_obs_list.append(np.count_nonzero(unit_type == TERRAN_BARRACKS_ID) > 0)
        ret_obs_list.append(np.count_nonzero(unit_type == TERRAN_SUPPLYDEPOT) > 0)
        selected_unit_id = currant_obs.observation['single_select'][0][0]
        if selected_unit_id == 0:
            try:
                selected_unit_id = currant_obs.observation['multi_select'][0][0]
            except TypeError:
                pass
            except IndexError:
                pass
        ret_obs_list.append(selected_unit_id)
        return np.array(ret_obs_list, copy=True)

    @staticmethod
    def get_action_id_from_action(sc2_action, sc2_args):
        super().get_action_id_from_action(sc2_action, sc2_args)

