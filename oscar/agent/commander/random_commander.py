import random

from oscar.agent.commander.base_commander import BaseCommander


class RandomCommander(BaseCommander):
    def __init__(self, subordinates):
        super().__init__(subordinates)

    def choose_subordinate(self):
        return random.choice(self._subordinates)
