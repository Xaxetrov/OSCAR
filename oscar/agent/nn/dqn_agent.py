from collections import namedtuple

import numpy
from keras.models import save_model
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from learning_tools.A3C_learner.neuralmodel import get_neural_network

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

"""Learning constants"""
_EPSILON_GREEDY = 0.95  # exploitation vs exploration criteria
_GAMMA = 0.5  # discount factor
_ALPHA = 0.2  # learning rate

InputStructure = namedtuple("InputStructure", "screen_size screen_number non_spatial_features")
OutputStructure = namedtuple("OutputStructure", "spatial_action_size non_spatial_action_size")


class DQNAgent(base_agent.BaseAgent):
    """
    A Deep Q learning agent
    See https://keon.io/deep-q-learning/ for a tutorial of a similar agent.
    """

    model = None

    def __init__(self, input, output):
        super().__init__()
        self.model = get_neural_network()
        self.action_old = None
        self.state_old = None
        self.predicted_reward_old = 0
        self.best_old_action_pos = [0, [0, 0]]
        self.best_action_pos = [0, [0, 0]]
        self.epsilon = _EPSILON_GREEDY
        self.spatial_action_size2x16 = 16 * 16  # TODO Change if more than 1 spatial output
        self.spatial_action_size = output.spatial_action_size
        self.non_spatial_action_num = output.non_spatial_action_size
        self.shrink = output.spatial_action_size / 16

    def get_random_action(self, obs):
        """
        Returns an available random action
        :param obs: the obs parameter given to the agent for the step call
        :return: guess what... A RANDOM ACTION
        """
        number_of_possible_action = 1  # _NO_OP
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            number_of_possible_action += self.spatial_action_size2x16
        if _SELECT_ARMY in obs.observation["available_actions"]:
            number_of_possible_action += 1  # TODO Find a way to use self.non_spatial_action_num
        # get a random number to select an action (including _NO_OP)
        selected_action_id = numpy.random.randint(0, number_of_possible_action)
        if _MOVE_SCREEN in obs.observation["available_actions"] and selected_action_id < self.spatial_action_size2x16:
            return self.get_move_action(selected_action_id)
        else:
            # here two case: whether we have action id 256 or 257 or we have 0 or 1
            # in both case if _SELECT_ARMY is not available the following call handles it
            return self.get_non_spacial_action(selected_action_id % self.spatial_action_size2x16)

    def get_move_action(self, linear_position):
        """
        Returns a pysc2 move action and argument to get to a given position
        :param linear_position: position of the move on a 16x16 grid, integer equal to y*16+x
        :return: The move action
        """
        x_16 = (linear_position % 16)
        y_16 = (linear_position // 16)
        x_true = x_16 * self.shrink
        y_true = y_16 * self.shrink
        action_args = [_NOT_QUEUED, [x_true, y_true]]
        self.best_action_pos = [1, [x_16, y_16]]
        return _MOVE_SCREEN, action_args

    def get_non_spacial_action(self, action_id):
        """
        Returns a pysc2 action corresponding to the given action id
        :param action_id: 0 -> NO_OP; 1 -> Select all army
        :return: an action id and its arguments
        """
        if action_id == 1:
            selected_action = _SELECT_ARMY
            action_args = [_SELECT_ALL]
            self.best_action_pos = [0, 1]
        else:
            selected_action = _NO_OP
            action_args = []
            self.best_action_pos = [0, 0]
        return selected_action, action_args

    def step(self, obs):
        """
        A deep Q learning iteration
        :param obs:
        :return:
        """
        super(DQNAgent, self).step(obs)

        state = [obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index],
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        formatted_state = numpy.zeros(shape=(1, self.spatial_action_size, self.spatial_action_size, 2), dtype=float)
        for formatted_row, state0_row, state1_row in zip(formatted_state[0], state[0], state[1]):
            for formatted_case, state0_case, state1_case in zip(formatted_row, state0_row, state1_row):
                formatted_case[0] = state0_case
                formatted_case[1] = state1_case

        # get reward prediction from neural network
        action = self.model.predict(formatted_state, batch_size=1)

        # if numpy.isnan(numpy.max(action[0])) or numpy.isnan(numpy.max(action[1])):
        #     print("action contain NaN !")
        #     if numpy.isnan(numpy.max(formatted_state)):
        #         print("formatted_state contain NaN too !!!")
        #     exit(1)

        action_vector = action[0][0][0:2]
        # mask _SELECT_ARMY action if not available
        if _SELECT_ARMY not in obs.observation["available_actions"]:
            action_vector[1] = 0.0
            # /!\ in this case the neural network will learn not to do this action -> side effect ?

        # compute best reward of the two main branch
        best_reward_spacial_action = numpy.max(action[1])
        best_reward_non_spacial_action = numpy.max(action_vector)

        # epsilon greedy exploration, epsilon probability to use neural network prediction
        if numpy.random.uniform() < self.epsilon:
            if best_reward_non_spacial_action < best_reward_spacial_action \
                    and _MOVE_SCREEN in obs.observation["available_actions"]:
                # get the best position according to reward
                position_vector = action[1][0]
                max_coordinate = numpy.argmax(position_vector)
                selected_action, action_args = self.get_move_action(max_coordinate)
                current_predicted_reward = best_predicted_reward = best_reward_spacial_action
            else:
                # select best action according to reward
                best_action_id = numpy.argmax(action_vector)
                current_predicted_reward = best_predicted_reward = best_reward_non_spacial_action
                selected_action, action_args = self.get_non_spacial_action(best_action_id)
        # 1 - epsilon probability to choose a random action
        else:
            # compute best reward according to neural network (used for learning)
            if best_reward_non_spacial_action < best_reward_spacial_action \
                    and _MOVE_SCREEN in obs.observation["available_actions"]:
                best_predicted_reward = best_reward_spacial_action
            else:
                best_predicted_reward = best_reward_non_spacial_action
            selected_action, action_args = self.get_random_action(obs)
            # get predicted reward of the random action
            if self.best_action_pos[0] == 0:
                current_predicted_reward = action[0][0][self.best_action_pos[1]]
            else:
                current_predicted_reward = action[1][0][self.best_action_pos[1][0]][self.best_action_pos[1][1]]

        # if we are not in the first step and so we have something to learn from our previous choice
        if self.action_old is not None and self.state_old is not None:
            # learn
            new_reward = self.predicted_reward_old + _ALPHA * \
                         (obs.reward + _GAMMA * best_predicted_reward - self.predicted_reward_old)
            # new_reward = min(max(new_reward, 0.0), 1.0)
            if self.best_old_action_pos[0] == 0:
                self.action_old[0][0][self.best_old_action_pos[1]] = new_reward
            else:
                self.action_old[1][0][self.best_old_action_pos[1][0]][self.best_old_action_pos[1][1]] = new_reward
            self.model.fit(x=self.state_old,
                           y=self.action_old,
                           batch_size=1,
                           epochs=1,
                           verbose=0,
                           validation_split=0)
        # remember the current values for next loop:
        self.action_old = action
        self.state_old = formatted_state
        self.best_old_action_pos = self.best_action_pos
        self.predicted_reward_old = current_predicted_reward

        return actions.FunctionCall(selected_action, action_args)

    def reset(self):
        super().reset()
        save_model(self.model, "constants.py/mineralshard.knn")
