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

    def step(self, obs, locked_choice=None):
        """
        Method called when in playing mod (cf LearningAgent)
        :param obs: current observation
        :param locked_choice: useless (compatibility)
        :return: a dict of the agent's choice for this step (action list, callbacks)
        """
        self.last_obs = obs
        # use a random meta action
        action_id = np.random.randint(0, ACTION_SPACE_SIZE)
        # print("random action id:", action_id, flush=True)
        play = self._transform_action(action_id)
        # print("play returned:", play, flush=True)
        return play
        # raise RuntimeError("Not implemented yet...")

    def _format_observation(self, full_obs):
        """
        transform the pysc2 observation into input for the learning agent (cf LearningAgent)
        :param full_obs: pysc2 observation
        :return: agent observation
        """
        self.last_obs = full_obs
        unit_type = full_obs.observation[SCREEN][SCREEN_UNIT_TYPE]
        minimap_player_relative = full_obs.observation[MINIMAP][MINI_PLAYER_RELATIVE]
        ret_obs_list = deque()
        # current mineral reserves (normalized on 1000 max 1000)
        minerals = full_obs.observation[PLAYER][MINERALS] / 1000.0
        ret_obs_list.append(min(minerals, 1.0))
        # current vespene reserves (normalized on 1000 max 1000)
        vespene = full_obs.observation[PLAYER][VESPENE] / 1000.0
        ret_obs_list.append(min(vespene, 1.0))
        # food supply: total used, army, scv and max
        ret_obs_list.append(full_obs.observation[PLAYER][FOOD_USED] / 200.0)
        ret_obs_list.append(full_obs.observation[PLAYER][FOOD_USED_BY_ARMY] / 200.0)
        ret_obs_list.append(full_obs.observation[PLAYER][FOOD_USED_BY_WORKERS] / 200.0)
        ret_obs_list.append(full_obs.observation[PLAYER][FOOD_CAP] / 200.0)
        # army count (very similar to food used by army with only marines)
        ret_obs_list.append(full_obs.observation[PLAYER][ARMY_COUNT] / 100.0)
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
        """
        transform an action id into a proper action (cf: LearningAgent)
        :param action_id: id of the action
        :return: action (in the hierarchy point of view)
        """
        action = self.get_meta_action(action_id)
        play = {'actions': action}
        return play

    def get_meta_action(self, action_id):
        """
        transform an action id into action by calling a meta action.
        :param action_id: id of the action
        :return: return of the meta action call
        """
        try:
            # action_id 0 is no_op
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

    def _available_action_mask(self):
        """
        Mask for unavailable actions (cf: LearningAgent)
        :return: a mask of the size of the possible action with 1 if the action
            is probably possible and 0 otherwise.
        """
        mask = np.ones(shape=self.action_space.n)
        # get useful information
        unit_type = self.last_obs.observation[SCREEN][SCREEN_UNIT_TYPE]
        minerals = self.last_obs.observation[PLAYER][MINERALS]
        vespene = self.last_obs.observation[PLAYER][VESPENE]
        food_used = self.last_obs.observation[PLAYER][FOOD_USED]
        food_used_worker = self.last_obs.observation[PLAYER][FOOD_USED_BY_WORKERS]
        food_cap = self.last_obs.observation[PLAYER][FOOD_CAP]
        has_supply_depot = np.count_nonzero(unit_type == TERRAN_SUPPLYDEPOT) > 0
        has_barrack = np.count_nonzero(unit_type == TERRAN_BARRACKS_ID) > 0
        # information on the currently selected unit
        selected_unit_id = self.last_obs.observation['single_select'][0][0]
        if selected_unit_id == 0:
            try:
                selected_unit_id = self.last_obs.observation['multi_select'][0][0]
            except (TypeError, IndexError):
                pass

        # perform basic check (mask unavailable actions)
        if minerals < 100 or food_used_worker == 0:
            mask[1] = 0
        if not has_supply_depot or minerals < 150 or food_used_worker == 0:
            mask[2] = 0
        if selected_unit_id != TERRAN_SCV:
            mask[4] = 0
        if not has_barrack or minerals < 50 or food_used + 1 > food_cap:
            mask[5] = 0
        if minerals < 50 or food_used + 1 > food_cap:
            mask[6] = 0
        if selected_unit_id not in [TERRAN_SCV, TERRAN_MARINE]:
            mask[8] = 0
        return mask

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
