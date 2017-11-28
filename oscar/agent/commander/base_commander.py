from abc import ABC, abstractmethod


class BaseCommander(ABC):
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

    def step(self, obs):
        """
        Does some work and choose a subordinate that will play
        :param obs: observations from the game
        :return: an action chosen by a subordinate
        """
        playing_subordinate = self.choose_subordinate()
        return playing_subordinate.step(obs)

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

    @abstractmethod
    def choose_subordinate(self):
        """
        Choose a subordinate among the list of subordinates, and make it play.
        :return: A subordinate among the list of subordinates.
        """

    def __str__(self):
        return self.print_tree(0)

    def print_tree(self, depth):
        depth += 1
        ret = "I am a {} and I have {} subordinates :\n".format(type(self).__name__, len(self._subordinates))
        for subordinate in self._subordinates:
            ret += "\t" * depth
            try:
                ret += subordinate.print_tree(depth + 1)
            except AttributeError:
                ret += str(subordinate) + "\n"
        return ret
