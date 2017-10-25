import random

from pysc2.agents import base_agent


class Commander(base_agent.BaseAgent):
    """
    A base class for commander agents.
    These are specialized agents that capable of delegating tasks to subordinates
    """

    def __init__(self, subordinates):
        """
        :param subordinates: a list of agents. These must be initialized already !
        """
        super().__init__()
        self._subordinates = subordinates

    def setup(self, obs_spec, action_spec):
        """
        Setups itself and all subordinates
        :return:
        """
        super().setup(obs_spec, action_spec)
        for subordinate in self._subordinates:
            subordinate.setup(obs_spec, action_spec)

    def step(self, obs):
        """
        Does some work and choses a subordinate that will play
        :param obs: observations from the game
        :return: an action chosen by a subordinate
        """
        super().step(obs)
        playing_subordinate = random.choice(self._subordinates)
        return playing_subordinate.step(obs)

    def reset(self):
        """
        Resets itself and all subordinates
        :return:
        """
        super().reset()
        for subordinate in self._subordinates:
            subordinate.reset()

    def add_subordinate(self, agent):
        """
        Hires a subordinate
        :param agent: to add
        :return:
        """
        self._subordinates.append(agent)

    def remove_subordinate(self, agent):
        """
        Fires a subordinate
        :param agent: to remove
        :return:
        """
        self._subordinates.remove(agent)

    def __str__(self):
        return self.print_tree(1)

    def print_tree(self, depth):
        ret = "I am a {} and I have {} subordinates :\n".format(type(self).__name__,
                                                                               len(self._subordinates))
        for subordinate in self._subordinates:
            ret += "\t" * depth
            if issubclass(type(subordinate), Commander):
                ret += subordinate.print_tree(depth + 1)
            else:
                ret += str(subordinate) + "\n"
        return ret
