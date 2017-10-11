"""A random agent for starcraft."""
import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization

from neuralmodel import get_neural_network

_MOVE_ACTION = actions.FUNCTIONS.Move_screen.id
_NOP = actions.FUNCTIONS.no_op.id

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

_EPSILON_GREEDY = 0.9


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
        self.epsilon = _EPSILON_GREEDY

    def int_to_action(self, x):

        non_spatial_act =[numpy.zeros(2)]
        spatial_act = [numpy.zeros([16, 16])]

        if x < 2:
            non_spatial_act[0][x] = 1
        else:
            spatial_act[0][(x-2) // 16][(x-2) % 16] = 1
        action = numpy.array([non_spatial_act, spatial_act])
        return action

    def step(self, obs):
        super(DQNAgent, self).step(obs)

        state = [obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index],
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        formatted_state = numpy.zeros(shape=(1, 64, 64, 2), dtype=float)
        for formatted_row, state0_row, state1_row in zip(formatted_state[0], state[0], state[1]):
            for formatted_case, state0_case, state1_case in zip(formatted_row, state0_row, state1_row):
                formatted_case[0] = state0_case
                formatted_case[1] = state1_case

        if numpy.random.uniform() < self.epsilon:
            action = self.model.predict(formatted_state, batch_size=1)
        else:
            action = self.int_to_action(numpy.random.randint(0, 257))

        action_vector = action[0][0]
        if _SELECT_ARMY not in obs.observation["available_actions"]:
            action_vector[1] = 0.0

        if numpy.max(action[0]) < numpy.max(action[1]) and _MOVE_SCREEN in obs.observation["available_actions"]:
            selected_action = _MOVE_ACTION
            position_vector = action[1][0]
            # get the best position according to "score"
            max_coordinate = numpy.argmax(position_vector)
            x_16 = (max_coordinate % 16)
            y_16 = (max_coordinate // 16)
            x_64 = x_16 * 4
            y_64 = y_16 * 4
            action_args = [[0], [x_64, y_64]]
            predicted_reward = position_vector[x_16][y_16]
            best_action_pos = [1, [x_16, y_16]]
        else:
            # select best action according to reward
            best_action_id = numpy.argmax(action_vector)
            predicted_reward = numpy.max(action[0])
            if best_action_id == 1:
                selected_action = _SELECT_ARMY
                # select all
                action_args = [[0]]
                best_action_pos = [0, 1]
            else:
                selected_action = _NO_OP
                action_args = []
                best_action_pos = [0, 0]

        if self.action_old is not None and self.state_old is not None:
            # learn
            new_reward = self.predicted_reward_old + 0.5 * (obs.reward + 0.9 * predicted_reward - self.predicted_reward_old)
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
        self.best_old_action_pos = best_action_pos
        self.predicted_reward_old = predicted_reward

        return actions.FunctionCall(selected_action, action_args)

    def reset(self):
        super(DQNAgent, self).reset()
        save_model(self.model, "mineralshard.knn")