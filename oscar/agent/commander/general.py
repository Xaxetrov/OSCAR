from pysc2.agents import base_agent
from pysc2.lib import actions

from agent.custom_agent import CustomAgent
from oscar.hiearchy_factory import build_hierarchy
from oscar.constants import NO_OP

DEFAULT_CONFIGURATION = "config/economic.json"


class General(CustomAgent):
    def __init__(self, configuration_filename=DEFAULT_CONFIGURATION):
        super().__init__()
        self._child = build_hierarchy(configuration_filename)
        print(self, flush=True)
        self._action_list = []
        self.failure_callback = None
        self.success_callback = None
        self._failed = False

    def step(self, obs):
        if len(self._action_list) != 0:
            return self._check_and_return_action(obs)

        # Empty list
        # Callback depending on last success / failure
        if self._failed and callable(self.failure_callback):
            self.failure_callback()
        elif callable(self.success_callback):
            self.success_callback()
        # TODO: How to pass arguments then ?

        child_return = self._child.step(obs)
        self._action_list = child_return['actions']
        try:
            self.success_callback = child_return['success_callback']
        except IndexError:
            self.success_callback = None
        try:
            self.failure_callback = child_return['failure_callback']
        except IndexError:
            self.failure_callback = None

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
            self._failed = False
            return self._action_list.pop(0)
        else:
            self._action_list = []
            self._failed = True
            return actions.FunctionCall(NO_OP, [])

    def __str__(self):
        try:
            return "general:\n\t" + self._child.print_tree(1)
        except AttributeError:
            return "general:\n\t" + str(self._child)
