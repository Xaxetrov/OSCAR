from collections import deque
import numpy as np
from gym import spaces
# import cProfile, pstats, io
from keras.models import load_model

from oscar.agent.learning_agent import LearningAgent
from oscar import meta_action
from oscar.constants import *

ACTION_SPACE_SIZE = 6
OBSERVATION_SPACE_SHAPE = (6,)


class ComplexLearningAgent(LearningAgent):

    def __init__(self, message="I'm learning", train_mode=False, shared_memory=None):
        self.last_obs = None
        self._message = message
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(low=0.0, high=200.0, shape=OBSERVATION_SPACE_SHAPE)
        super().__init__(train_mode, shared_memory)
        if not train_mode:
            self.model = load_model("learning_tools/learning_nn/learning_complex.knn")
            self.model.summary()
        # self.pr = cProfile.Profile()

    def _step(self, obs):
        """
        Method called when in playing mod (cf LearningAgent)
        :param obs: current observation
        :return: a dict of the agent's choice for this step (action list, callbacks)
        """
        self.last_obs = obs
        nn_obs = self._format_observation(obs)
        action_probability, value = self.model.predict(nn_obs.reshape((1, ACTION_SPACE_SIZE)))
        action_probability *= self._available_action_mask()
        action_sum = np.sum(action_probability)
        if action_sum == 0.0:
            action_probability = self._available_action_mask()
            action_probability /= np.sum(action_probability)
        else:
            action_probability = action_probability / np.sum(action_probability)
        action_id = np.random.choice(range(ACTION_SPACE_SIZE), p=action_probability.reshape(ACTION_SPACE_SIZE))
        play = self._transform_action(action_id)
        return play

    def _format_observation(self, full_obs):
        """
        transform the pysc2 observation into input for the learning agent (cf LearningAgent)
        :param full_obs: pysc2 observation
        :return: agent observation
        """
        # self.pr.enable()
        self.last_obs = full_obs
        unit_type = full_obs.observation["screen"][SCREEN_UNIT_TYPE]
        ret_obs_list = deque()
        # minerals available
        minerals = full_obs.observation['player'][MINERALS]
        ret_obs_list.append(min(minerals / (10 * 40.0), 2.0))
        # food supply: are we on max
        food_available = full_obs.observation['player'][FOOD_CAP]
        ret_obs_list.append(food_available / 100.0)
        # is army count
        ret_obs_list.append(full_obs.observation['player'][ARMY_COUNT] / 100.0)
        # scv count
        ret_obs_list.append(min(full_obs.observation['player'][FOOD_USED_BY_WORKERS] / 24.0, 1.5))
        # information on which building are already build (don't check player id)
        # experimental measure: barrack surface is 118 with 84x84 screen
        ret_obs_list.append((np.count_nonzero(unit_type == TERRAN_BARRACKS_ID) // 110) / 4.0)
        # time step info
        ret_obs_list.append(min(self.episode_steps / (100.0 * 10.0), 2.0))
        # self.pr.disable()
        return np.array(ret_obs_list, copy=True, dtype=float)
    
    def _transform_action(self, action_id):
        """
        transform an action id into a proper action (cf: LearningAgent)
        :param action_id: id of the action
        :return: action (in the hierarchy point of view)
        """
        # self.pr.enable()
        action = self.get_meta_action(action_id)
        play = {'actions': action}
        # self.pr.disable()
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
                                         building_id=BUILD_SUPPLY_DEPOT,
                                         use_slow_scv_selection_method=False)
            elif action_id == 2:  # build Barracks
                return meta_action.build(self.last_obs,
                                         building_tiles_size=3,
                                         building_id=BUILD_BARRACKS,
                                         use_slow_scv_selection_method=False)
            elif action_id == 3:  # train Marines
                return meta_action.train_unit(self.last_obs,
                                              building_id=TERRAN_BARRACKS_ID,
                                              action_train_id=TRAIN_MARINE_QUICK)
            elif action_id == 4:  # attack !
                action = [actions.FunctionCall(SELECT_ARMY, [SELECT_ALL])]
                action += meta_action.attack_minimap(self.last_obs, queued=False)
                return action
            elif action_id == 5:  # train SCV
                return meta_action.train_unit(self.last_obs,
                                              building_id=TERRAN_COMMAND_CENTER,
                                              action_train_id=TRAIN_SCV_QUICK)
        except meta_action.ActionError:
            pass

        return [actions.FunctionCall(NO_OP, [])]

    def _available_action_mask(self):
        """
        Mask for unavailable actions (cf: LearningAgent)
        :return: a mask of the size of the possible action with 1 if the action
            is probably possible and 0 otherwise.
        """
        # self.pr.enable()
        mask = np.ones(shape=self.action_space.n)
        # get useful information
        unit_type = self.last_obs.observation["screen"][SCREEN_UNIT_TYPE]
        minerals = self.last_obs.observation['player'][MINERALS]
        food_used = self.last_obs.observation['player'][FOOD_USED]
        food_cap = self.last_obs.observation['player'][FOOD_CAP]
        food_used_worker = self.last_obs.observation['player'][FOOD_USED_BY_WORKERS]
        food_used_army = self.last_obs.observation['player'][ARMY_COUNT]
        has_supply_depot = np.count_nonzero(unit_type == TERRAN_SUPPLYDEPOT) > 0
        has_barrack = np.count_nonzero(unit_type == TERRAN_BARRACKS_ID) > 0

        # perform basic check (mask unavailable actions)
        if minerals < 100 or food_used_worker == 0:
            mask[1] = 0
        if not has_supply_depot or minerals < 150 or food_used_worker == 0:
            mask[2] = 0
        if not has_barrack or minerals < 50 or food_used + 1 > food_cap:
            mask[3] = 0
        if food_used_army == 0:
            mask[4] = 0
        # self.pr.disable()
        return mask

    def reset(self):
        # if self.failed_meta_action_counter != 0:
        #     s = io.StringIO()
        #     ps = pstats.Stats(self.pr, stream=s).sort_stats('cumulative')
        #     ps.print_stats()
        #     print(s.getvalue())
        # self.pr = cProfile.Profile()
        super().reset()

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
