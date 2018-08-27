import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from learning_tools.A3C_learner.neuralmodel import get_neural_network

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

_EPSILON_GREEDY = 1.0 # exploration vs exploitation criteria


class PlayAgent(base_agent.BaseAgent):
    """
    Same as DQNAgent but does not learn.
    Simply applies a trained model to test it.
    """

    model = None

    def __init__(self):
        super(PlayAgent, self).__init__()
        self.model = get_neural_network()
        try:
            with open("constants.py", mode='r') as config:
                self.number_of_run = int(config.readline())
                self.epsilon_step = int(config.readline()) / 100.0
                self.epsilon = 0.0
                self.step_by_epsilon = 240 * self.number_of_run
        except OSError:
            self.number_of_run = 10
            self.epsilon = _EPSILON_GREEDY
            self.epsilon_step = 0.0
            self.step_by_epsilon = -1

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
        """
        Returns a pysc2 move action and argument to get to a given position
        :param linear_position: position of the move on a 16x16 grid, integer equal to y*16+x
        :return: The move action
        """
        x_16 = (linear_position % 16)
        y_16 = (linear_position // 16)
        x_64 = x_16 * 4
        y_64 = y_16 * 4
        action_args = [_NOT_QUEUED, [x_64, y_64]]
        return _MOVE_SCREEN, action_args

    @staticmethod
    def get_non_spacial_action(action_id):
        """
        Returns a pysc2 action corresponding to the given action id
        :param action_id: 0 -> NO_OP; 1 -> Select all army
        :return: an action id and its arguments
        """
        if action_id == 1:
            selected_action = _SELECT_ARMY
            action_args = [_SELECT_ALL]
        else:
            selected_action = _NO_OP
            action_args = []
        return selected_action, action_args

    def step(self, obs, locked_choice=None):
        super(PlayAgent, self).step(obs)

        if numpy.random.rand() < self.epsilon:
            state = [obs.observation[SCREEN][features.SCREEN_FEATURES.player_relative.index],
                     obs.observation[SCREEN][features.SCREEN_FEATURES.selected.index]]
            formatted_state = numpy.zeros(shape=(1, 64, 64, 2), dtype=float)
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

            # compute best reward of the two main branch
            best_reward_spacial_action = numpy.max(action[1])
            best_reward_non_spacial_action = numpy.max(action[0][0][0:2])

            action_vector = action[0][0]
            # mask _SELECT_ARMY action if not available
            if _SELECT_ARMY not in obs.observation["available_actions"]:
                action_vector[1] = 0.0
                # /!\ in this case the neural network will learn not to do this action -> side effect ?

            # if best_reward_non_spacial_action < action_vector[2] \
            if best_reward_non_spacial_action < best_reward_spacial_action \
                    and _MOVE_SCREEN in obs.observation["available_actions"]:
                # get the best position according to reward
                position_vector = action[1][0]
                max_coordinate = numpy.argmax(position_vector)
                selected_action, action_args = self.get_move_action(max_coordinate)
            else:
                # select best action according to reward
                print(action_vector)
                best_action_id = numpy.argmax(action_vector[0:2])
                selected_action, action_args = self.get_non_spacial_action(best_action_id)
        else:
            selected_action, action_args = self.get_random_action(obs)

        return actions.FunctionCall(selected_action, action_args)

    def reset(self):
        super(PlayAgent, self).reset()
        # print("reward for this game:", self.reward, "(", self.steps, "steps)")
        if self.steps == self.step_by_epsilon:
            with open("reward.csv", mode='a') as out_file:
                out_file.write(str(self.epsilon) + ", " + str(self.reward * 240.0 / self.steps) + "\n")
            self.epsilon += self.epsilon_step
            self.reward = 0
            self.steps = 0
