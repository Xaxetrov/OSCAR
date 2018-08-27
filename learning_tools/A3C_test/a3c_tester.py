"""A random agent for starcraft."""
import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from keras.models import Model, load_model, save_model

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

_EPSILON_GREEDY = 1.0  # exploration vs exploitation criteria


class A3CTester(base_agent.BaseAgent):
    """A NN agent for starcraft."""

    model = None

    def __init__(self):
        super().__init__()
        self.model = load_model("learning_tools/learning_nn/pysc2-mineralshard-v0.knn")
        # try:
        #     with open("config", mode='r') as config:
        #         self.number_of_run = int(config.readline())
        #         self.epsilon_step = int(config.readline()) / 100.0
        #         self.epsilon = 0.0
        #         self.step_by_epsilon = 240 * self.number_of_run
        # except OSError:
        #     self.number_of_run = 10
        #     self.epsilon = _EPSILON_GREEDY
        #     self.epsilon_step = 0.0
        #     self.step_by_epsilon = -1

    def get_random_action(self, obs):
        """return an available random action
            -obs: the obs parameter given to the agent for the step call
        """
        number_of_possible_action = 1  # _NO_OP
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            number_of_possible_action += 256
        if _SELECT_ARMY in obs.observation["available_actions"]:
            number_of_possible_action += 1
        # get a random number to select an action (including _NO_OP)
        selected_action_id = numpy.random.randint(0, number_of_possible_action)
        if _MOVE_SCREEN in obs.observation["available_actions"] and selected_action_id < 256:
            return self.get_move_action(selected_action_id)
        else:
            # here two case: whether we have action id 256 or 257 or we have 0 or 1
            # in both case if _SELECT_ARMY is not available the following call handles it
            return self.get_non_spacial_action(selected_action_id % 256)

    @staticmethod
    def get_move_action(linear_position):
        """return a pysc2 action and argument to do a move action at the pos given
            -linear_position : position of the move on a 16x16 grid, integer equal to y*16+x
            """
        x_16 = (linear_position // 16)
        y_16 = (linear_position % 16)
        x_true = min(x_16 * 4, 63)
        y_true = min(y_16 * 4, 63)
        # x and y are not in the right order, else it doesn't work...
        action_args = [_NOT_QUEUED, [y_true, x_true]]
        return _MOVE_SCREEN, action_args

    def step(self, obs):
        super().step(obs)

        if _MOVE_SCREEN not in obs.observation["available_actions"]:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

        if True:  # numpy.random.rand() < self.epsilon:
            state = [obs.observation[SCREEN][features.SCREEN_FEATURES.player_relative.index],
                     obs.observation[SCREEN][features.SCREEN_FEATURES.selected.index]]
            formatted_state = numpy.zeros(shape=(1, 64, 64, 2), dtype=float)
            for formatted_row, state0_row, state1_row in zip(formatted_state[0], state[0], state[1]):
                for formatted_case, state0_case, state1_case in zip(formatted_row, state0_row, state1_row):
                    formatted_case[0] = state0_case
                    formatted_case[1] = state1_case

            # get reward prediction from neural network
            action = self.model.predict(formatted_state, batch_size=1)

            # action_num = numpy.argmax(action[0][0])
            action_num = numpy.random.choice(256, p=action[0][0])

            selected_action, action_args = self.get_move_action(action_num)
        else:
            selected_action, action_args = self.get_random_action(obs)

        return actions.FunctionCall(selected_action, action_args)

    def reset(self):
        super().reset()
        print("reward for this game:", self.reward, "(", self.steps, "steps)")
        self.reward = 0

        # if self.steps == self.step_by_epsilon:
        #    with open("reward.csv", mode='a') as out_file:
        #        out_file.write(str(self.epsilon) + ", " + str(self.reward * 240.0 / self.steps) + "\n")
        #    self.epsilon += self.epsilon_step
        #    self.reward = 0
        #    self.steps = 0
