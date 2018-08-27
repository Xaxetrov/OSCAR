from pysc2.lib import actions
from pysc2.agents.base_agent import BaseAgent

from oscar.hiearchy_factory import build_hierarchy
from oscar.constants import NO_OP
import time

# DEFAULT_CONFIGURATION = "config/full_hierarchy.json"
DEFAULT_CONFIGURATION = "config/idleSCVtest.json"


class General(BaseAgent):
    """
    The agent at the top of the command chain. It is usually him that will be the interface with PySC2
    """
    def __init__(self, configuration_filename=DEFAULT_CONFIGURATION):
        """
        Initializes members and call hierarchy factory
        :param configuration_filename: The configuration file to pass to the hierarchy factory on launch
        """
        super().__init__()
        self._child, self.training_memory = build_hierarchy(configuration_filename)
        print(self)
        self._action_list = []
        self._failure_callback = None
        self._success_callback = None
        self._success = False

    def step(self, obs):
        """
        Called by PySC2 directly.
        Either unstack the current action_list
        or callbacks depending on success or failure and ask for child for new actions
        :param obs: observation object passed by PySC2
        :return: some action
        """
        super().step(obs)
        #time.sleep(1.0)
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
        current_action = self._action_list.pop(0)
        if current_action.function in obs.observation["available_actions"] \
                and self._check_argument(current_action):
            self._success = True
            return current_action
        else:
            self._action_list = []
            self._success = False
            return actions.FunctionCall(NO_OP, [])

    def _check_argument(self, current_action):
        """
        Checks that passed action arguments are valid
        :param current_action: action with arguments to check
        :return: Valid or not
        """
        # TODO: check why 0: multi player ?
        asked_args = self.action_spec[0].functions[current_action.function].args
        args = current_action.arguments
        if len(args) != len(asked_args):
            print("---- Error args size not accurate ----")
            print("- action:", current_action.function, args)
            print("- wanted args:", asked_args)
            return False
        for arg, asked_arg in zip(args, asked_args):
            if len(arg) != len(asked_arg.sizes):
                print("---- Error arg size not accurate ----")
                print("- action:", current_action.function, args)
                print("- wanted args:", asked_args)
                return False
            for value, asked_size in zip(arg, asked_arg.sizes):
                if value < 0 or value >= asked_size:
                    print("---- Error arg value not accurate ----")
                    print("- action:", current_action.function, args)
                    print("- wanted args:", asked_args)
                    return False
            pass
        return True

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        self._child.setup(obs_spec, action_spec)

    def reset(self):
        super().reset()
        self._child.reset()

    def __str__(self):
        try:
            return "general:\n\t" + self._child.print_tree(1)
        except AttributeError:
            return "general:\n\t" + str(self._child)
