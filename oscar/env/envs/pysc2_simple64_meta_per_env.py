from oscar.env.envs.pysc2_simple64_meta_env import Pysc2Simple64MetaEnv
from oscar.constants import *

from gym import spaces
import numpy as np
from collections import deque


class Pysc2Simple64MetaPerEnv(Pysc2Simple64MetaEnv):

    observation_space = spaces.Box(low=0, high=float('Inf'), shape=(10,))

    def __init__(self):
        super().__init__()
        self.observation_space.n = self.observation_space.shape[0]

    def get_return_obs(self, currant_obs):
        unit_type = currant_obs.observation["screen"][SCREEN_UNIT_TYPE]
        ret_obs_list = deque()
        minerals = currant_obs.observation['player'][MINERALS] / 1000.0
        ret_obs_list.append(min(minerals, 1.0))
        vespene = currant_obs.observation['player'][VESPENE] / 1000.0
        ret_obs_list.append(min(vespene, 1.0))
        ret_obs_list.append(currant_obs.observation['player'][FOOD_USED] / 200.0)
        ret_obs_list.append(currant_obs.observation['player'][FOOD_USED_BY_ARMY] / 200.0)
        ret_obs_list.append(currant_obs.observation['player'][FOOD_USED_BY_WORKERS] / 200.0)
        ret_obs_list.append(currant_obs.observation['player'][FOOD_CAP] / 200.0)
        ret_obs_list.append(currant_obs.observation['player'][ARMY_COUNT] / 100.0)
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
        ret_obs_list.append(selected_unit_id / TERRAN_MARINE)
        return np.array(ret_obs_list, copy=True, dtype=float)

    @staticmethod
    def get_action_id_from_action(sc2_action, sc2_args):
        super().get_action_id_from_action(sc2_action, sc2_args)

