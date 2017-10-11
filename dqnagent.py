"""A random agent for starcraft."""
import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization

from neuralmodel import get_neural_network

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

_EPSILON_GREEDY = 0.9 # exploration vs exploitation criteria
_GAMMA = 0.9 # discount factor
_ALPHA = 0.5 # learning rate



class DQNAgent(base_agent.BaseAgent):
    """A NN agent for starcraft."""

    model = None

    def __init__(self):
        super(DQNAgent, self).__init__()
        self.model = get_neural_network()
        self.action_old = None
        self.state_old = None
        self.predicted_reward_old = 0
        self.best_old_action_pos = [0, [0, 0]]
        self.best_action_pos = [0, [0, 0]]
        self.epsilon = _EPSILON_GREEDY

    def get_random_action(self, obs):
        """return a available random action
            -obs: the obs parameter given to the agent for the step call
        """
        number_of_possible_action = 1
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            number_of_possible_action += 256
        if _SELECT_ARMY in obs.observation["available_actions"]:
            number_of_possible_action += 1
        # get a random number to select an action (including _NO_OP)
        selected_action_id = numpy.random.randint(0, number_of_possible_action)
        if _MOVE_SCREEN in obs.observation["available_actions"] and selected_action_id < 256:
            return self.get_move_action(selected_action_id)
        else:
            # here two case, or we have action id 256 or 257 or we have 0 or 1
            # in both case if _SELECT_ARMY is not available the following call handle it
            return self.get_none_spacial_action(selected_action_id % 256)

    def get_move_action(self, linear_position):
        """return a pysc2 action and argument to do a move action at the pos given
            -linear_position : position of the move on a 16x16 grid, integer equal to y*16+x
            """
        x_16 = (linear_position % 16)
        y_16 = (linear_position // 16)
        x_64 = x_16 * 4
        y_64 = y_16 * 4
        action_args = [_NOT_QUEUED, [x_64, y_64]]
        self.best_action_pos = [1, [x_16, y_16]]
        return _MOVE_SCREEN, action_args

    def get_none_spacial_action(self, action_id):
        """return a pysc2 action coresponding to the given action id
            -action id: 0 -> NO_OP
                        1 -> Select all army
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
        super(DQNAgent, self).step(obs)

        state = [obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index],
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        formatted_state = numpy.zeros(shape=(1, 64, 64, 2), dtype=float)
        for formatted_row, state0_row, state1_row in zip(formatted_state[0], state[0], state[1]):
            for formatted_case, state0_case, state1_case in zip(formatted_row, state0_row, state1_row):
                formatted_case[0] = state0_case
                formatted_case[1] = state1_case

        # get reward prediction from neural network
        action = self.model.predict(formatted_state, batch_size=1)

        # compute best reward of the two main branch
        best_reward_spacial_action = numpy.max(action[1])
        best_reward_non_spacial_action = numpy.max(action[0])

        # epsilon greedy exploration, epsilon probability to use neural network prediction
        if numpy.random.uniform() < self.epsilon:
            action_vector = action[0][0]
            # mask _SELECT_ARMY action if not available
            if _SELECT_ARMY not in obs.observation["available_actions"]:
                action_vector[1] = 0.0
                # /!\ in this case the neural network will learn not to do this action -> side effect ?

            if best_reward_non_spacial_action < best_reward_spacial_action \
                    and _MOVE_SCREEN in obs.observation["available_actions"]:
                # get the best position according to reward
                position_vector = action[1][0]
                max_coordinate = numpy.argmax(position_vector)
                selected_action, action_args = self.get_move_action(max_coordinate)
                predicted_reward = best_reward_spacial_action
            else:
                # select best action according to reward
                best_action_id = numpy.argmax(action_vector)
                predicted_reward = best_reward_non_spacial_action
                selected_action, action_args = self.get_none_spacial_action(best_action_id)
        # 1 - epsilon probability to choose a random action
        else:
            # compute best reward according to neural network (used for learning)
            if best_reward_non_spacial_action < best_reward_spacial_action \
                    and _MOVE_SCREEN in obs.observation["available_actions"]:
                predicted_reward = best_reward_spacial_action
            else:
                predicted_reward = best_reward_non_spacial_action
            selected_action, action_args = self.get_random_action(obs)

        # if we are not in the first step and so we have something to learn from our previous choice
        if self.action_old is not None and self.state_old is not None:
            # learn
            new_reward = self.predicted_reward_old + _ALPHA * \
                         (obs.reward + _GAMMA * predicted_reward - self.predicted_reward_old)
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
        self.predicted_reward_old = predicted_reward

        return actions.FunctionCall(selected_action, action_args)

    def reset(self):
        super(DQNAgent, self).reset()
        save_model(self.model, "mineralshard.knn")