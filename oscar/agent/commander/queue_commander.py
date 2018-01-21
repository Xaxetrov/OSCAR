from oscar.agent.commander.base_commander import BaseCommander


class QueueCommander(BaseCommander):

    def __init__(self, subordinates):
        super().__init__(subordinates)
        self.__next_agent = 0

    def choose_subordinate(self, obs):
        """
        Round robin distribution
        :return: The chosen
        """
        playing_subordinate = self._subordinates[self.__next_agent]
        self.__next_agent = (self.__next_agent + 1) % len(self._subordinates)
        return playing_subordinate
