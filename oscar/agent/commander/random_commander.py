import random

from oscar.agent.commander.base_commander import BaseCommander


class RandomCommander(BaseCommander):
    """
    A special commander that chooses the playing subordinate by rolling a die
    """
    def __init__(self, subordinates):
        super().__init__(subordinates)

    def choose_subordinate(self, obs):
        return random.choice(self._subordinates)
