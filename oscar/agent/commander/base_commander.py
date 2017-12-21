from abc import ABC, abstractmethod

from oscar.agent.custom_agent import CustomAgent


class BaseCommander(ABC, CustomAgent):
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
        self._locked_choice = False
        self._playing_subordinate = None

    def step(self, obs):
        """
        Does some work and choose a subordinate that will play
        :param obs: observations from the game
        :return: an action chosen by a subordinate
        """

        if not self._locked_choice:
            self._playing_subordinate = self.choose_subordinate(obs)
        play = self._playing_subordinate.step(obs)
        try:
            self._locked_choice = play["locked_choice"]
        except KeyError:
            self._locked_choice = False
        return play

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
    def choose_subordinate(self, obs):
        """
        Choose a subordinate among the list of subordinates, and make it play.
        :return: A subordinate among the list of subordinates.
        """

    def unlock_choice(self):
        if self._locked_choice:
            self._locked_choice = False
            try:
                self._playing_subordinate.unlock_choice()
            # Method does not exist = not a commander
            except AttributeError:
                pass

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec())
        for subordinate in self._subordinates:
            subordinate.setup(obs_spec, action_spec)

    def reset(self):
        super().reset()
        for subordinate in self._subordinates:
            subordinate.reset()

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
