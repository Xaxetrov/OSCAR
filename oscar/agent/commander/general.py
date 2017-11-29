from pysc2.agents import base_agent
from pysc2.lib import actions

from oscar.hiearchy_factory import build_hierarchy
from oscar.constants import NO_OP

DEFAULT_CONFIGURATION = "config/economic.json"


class General(base_agent.BaseAgent):
    def __init__(self, configuration_filename=DEFAULT_CONFIGURATION):
        super().__init__()
        self._child = build_hierarchy(configuration_filename)
        print(self, flush=True)
        self._action_list = []
        self._callback = None
        self._callback_params = []

    def step(self, obs):
        if len(self._action_list) != 0:
            return self._check_and_return_action(obs)

        child_return = self._child.step(obs)
        self._action_list = child_return[0]
        try:
            self._callback = child_return[1]
        except IndexError:
            self._callback = None

        try:
            self._callback_params = child_return[2]
        except IndexError:
            self._callback_params = []

        return self._check_and_return_action(obs)

    def _check_and_return_action(self, obs):
        """
        Check that the first action in self._action_list is among available actions, and return it.
        If it is not, return call the callback argument provided by the last agent, empty the action list,
        and return NO_OP.
        Precondition: self._action_list is not empty.
        :param obs: The observation provided by pysc2.
        :return: The first action of the action list if it is valid, else NO_OP
        """
        if self._action_list[0].function in obs.observation["available_actions"]:
            return self._action_list.pop(0)
        else:
            self._action_list = []
            if callable(self._callback):
                self._callback(*self._callback_params)
            return actions.FunctionCall(NO_OP, [])

    def __str__(self):
        try:
            return "general:\n\t" + self._child.print_tree(1)
        except AttributeError:
            return "general:\n\t" + str(self._child)
