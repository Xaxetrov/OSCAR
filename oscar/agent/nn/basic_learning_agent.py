from collections import deque
import numpy as np
from gym import spaces

from oscar.agent.learning_agent import LearningAgent
from oscar import meta_action
from oscar.constants import *

ACTION_SPACE_SIZE = 6
OBSERVATION_SPACE_SHAPE = (5,)


class BasicLearningAgent(LearningAgent):

    def __init__(self, message="I'm learning", train_mode=False, shared_memory=None):
        self.last_obs = None
        self._message = message
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=OBSERVATION_SPACE_SHAPE)
        super().__init__(train_mode, shared_memory)

    def _step(self, obs):
        """
        Method called when in playing mod (cf LearningAgent)
        :param obs: current observation
        :return: a dict of the agent's choice for this step (action list, callbacks)
        """
        self.last_obs = obs
        # use a random meta action
        action_id = np.random.randint(0, ACTION_SPACE_SIZE)
        print("random action id:", action_id, flush=True)
        play = self._transform_action(action_id)
        print("play returned:", play, flush=True)
        return play
        # raise RuntimeError("Not implemented yet...")

    def _format_observation(self, full_obs):
        """
        transform the pysc2 observation into input for the learning agent (cf LearningAgent)
        :param full_obs: pysc2 observation
        :return: agent observation
        """
        self.last_obs = full_obs
        unit_type = full_obs.observation["screen"][SCREEN_UNIT_TYPE]
        minimap_player_relative = full_obs.observation['minimap'][MINI_PLAYER_RELATIVE]
        ret_obs_list = deque()
        # food supply: are we on max
        food_available = full_obs.observation['player'][FOOD_USED] != full_obs.observation['player'][FOOD_CAP]
        ret_obs_list.append(food_available)
        # is army bigger than 10 ?
        ret_obs_list.append(full_obs.observation['player'][ARMY_COUNT] > 10)
        # information on which building are already build (don't check player id)
        ret_obs_list.append(np.count_nonzero(unit_type == TERRAN_BARRACKS_ID) > 0)
        ret_obs_list.append(np.count_nonzero(unit_type == TERRAN_SUPPLYDEPOT) > 0)
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
            elif action_id == 3:  # train Marines
                return meta_action.train_unit(self.last_obs,
                                              building_id=TERRAN_BARRACKS_ID,
                                              action_train_id=TRAIN_MARINE_QUICK)
            elif action_id == 4:  # train SCV
                return meta_action.train_unit(self.last_obs,
                                              building_id=TERRAN_COMMAND_CENTER,
                                              action_train_id=TRAIN_SCV_QUICK)
            elif action_id == 5:  # attack !
                action = [actions.FunctionCall(SELECT_ARMY, [SELECT_ALL])]
                action += meta_action.attack_minimap(self.last_obs, queued=False)
                return action
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
        unit_type = self.last_obs.observation["screen"][SCREEN_UNIT_TYPE]
        minerals = self.last_obs.observation['player'][MINERALS]
        food_used = self.last_obs.observation['player'][FOOD_USED]
        food_cap = self.last_obs.observation['player'][FOOD_CAP]
        has_supply_depot = np.count_nonzero(unit_type == TERRAN_SUPPLYDEPOT) > 0
        has_barrack = np.count_nonzero(unit_type == TERRAN_BARRACKS_ID) > 0

        # perform basic check (mask unavailable actions)
        if minerals < 100:
            mask[1] = 0
        if not has_supply_depot or minerals < 150:
            mask[2] = 0
        if not has_barrack or minerals < 50 or food_used + 1 > food_cap:
            mask[3] = 0
        if minerals < 50 or food_used + 1 > food_cap:
            mask[4] = 0
        return mask

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
