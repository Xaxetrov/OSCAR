from pysc2.lib import actions
from pysc2.agents.base_agent import BaseAgent

from oscar.hiearchy_factory import build_hierarchy
from oscar.constants import NO_OP

DEFAULT_CONFIGURATION = "config/explorator.json"


class General(BaseAgent):
    def __init__(self, configuration_filename=DEFAULT_CONFIGURATION):
        super().__init__()
        self._child = build_hierarchy(configuration_filename)
        print(self)
        self._action_list = []
        self._failure_callback = None
        self._success_callback = None
        self._success = False

    def step(self, obs):
        if len(self._action_list) != 0:
            return self._check_and_return_action(obs)

        # Empty list
        # Callback depending on last success / failure
        if self._success:
            if callable(self._success_callback):
                self._success_callback()
        else:
            if callable(self._failure_callback):
                self._failure_callback()
            try:
                self._child.unlock_choice()
            except AttributeError:
                pass
        # TODO: How to pass arguments then ?

        child_return = self._child.step(obs)
        self._action_list = child_return["actions"]
        try:
            self._success_callback = child_return["success_callback"]
        except KeyError:
            self._success_callback = None
        try:
            self._failure_callback = child_return["failure_callback"]
        except KeyError:
            self._failure_callback = None

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
            self._success = True
            return self._action_list.pop(0)
        else:
            self._action_list = []
            self._success = False
            return actions.FunctionCall(NO_OP, [])

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self._child.setup(obs_spec, action_spec)

    def __str__(self):
        try:
            return "general:\n\t" + self._child.print_tree(1)
        except AttributeError:
            return "general:\n\t" + str(self._child)
