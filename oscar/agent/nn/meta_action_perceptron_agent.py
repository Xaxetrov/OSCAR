from collections import deque
import numpy as np
from gym import spaces

from oscar.agent.learning_agent import LearningAgent
from oscar import meta_action
from oscar.constants import *

ACTION_SPACE_SIZE = 9
OBSERVATION_SPACE_SHAPE = (12,)


class MetaActionPerceptronAgent(LearningAgent):

    def __init__(self, message="I'm learning", train_mode=False, shared_memory=None):
        self.last_obs = None
        self._message = message
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=0, high=float('Inf'), shape=OBSERVATION_SPACE_SHAPE)
        super().__init__(train_mode, shared_memory)

    def _step(self, obs):
        self.last_obs = obs
        # use a random meta action
        action_id = np.random.randint(0, ACTION_SPACE_SIZE)
        print("random action id:", action_id, flush=True)
        play = self._transform_action(action_id)
        print("play returned:", play, flush=True)
        return play
        # raise RuntimeError("Not implemented yet...")

    def _format_observation(self, full_obs):
        self.last_obs = full_obs
        unit_type = full_obs.observation["screen"][SCREEN_UNIT_TYPE]
        minimap_player_relative = full_obs.observation['minimap'][MINI_PLAYER_RELATIVE]
        ret_obs_list = deque()
        # current mineral reserves (normalized on 1000 max 1000)
        minerals = full_obs.observation['player'][MINERALS] / 1000.0
        ret_obs_list.append(min(minerals, 1.0))
        # current vespene reserves (normalized on 1000 max 1000)
        vespene = full_obs.observation['player'][VESPENE] / 1000.0
        ret_obs_list.append(min(vespene, 1.0))
        # food supply: total used, army, scv and max
        ret_obs_list.append(full_obs.observation['player'][FOOD_USED] / 200.0)
        ret_obs_list.append(full_obs.observation['player'][FOOD_USED_BY_ARMY] / 200.0)
        ret_obs_list.append(full_obs.observation['player'][FOOD_USED_BY_WORKERS] / 200.0)
        ret_obs_list.append(full_obs.observation['player'][FOOD_CAP] / 200.0)
        # army count (very similar to food used by army with only marines)
        ret_obs_list.append(full_obs.observation['player'][ARMY_COUNT] / 100.0)
        # information on which building are already build (don't check player id)
        ret_obs_list.append(np.count_nonzero(unit_type == TERRAN_BARRACKS_ID) > 0)
        ret_obs_list.append(np.count_nonzero(unit_type == TERRAN_SUPPLYDEPOT) > 0)
        # information on the currently selected unit
        selected_unit_id = full_obs.observation['single_select'][0][0]
        if selected_unit_id == 0:
            try:
                selected_unit_id = full_obs.observation['multi_select'][0][0]
            except (TypeError, IndexError):
                pass
        ret_obs_list.append(selected_unit_id / TERRAN_MARINE)
        # information on the remaining mineral on screen, normalized (205 is an experimental value...)
        number_of_mineral = np.count_nonzero(np.isin(unit_type, ALL_MINERAL_FIELD)) / 205
        ret_obs_list.append(number_of_mineral)
        # Enemy base found (bool)
        is_enemy_found = np.count_nonzero(minimap_player_relative == PLAYER_HOSTILE) > 0
        ret_obs_list.append(is_enemy_found)
        return np.array(ret_obs_list, copy=True, dtype=float)
    
    def _transform_action(self, action_id):
        action = self.get_meta_action(action_id)
        play = {'actions': action}
        return play

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
        except meta_action.ActionError:
            pass

        return [actions.FunctionCall(NO_OP, [])]

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
