from oscar.agent.commander.base_commander import BaseCommander
from oscar.constants import *


class CombatManager(BaseCommander):

    _ARMY_SUPPLIER = 0
    _ATTACK_MANAGER = 1
    _MICRO_MANAGER = 2

    def __init__(self, subordinates):
        self.count = 0
        super().__init__(subordinates)

    def choose_subordinate(self, obs, locked_choice):
        if locked_choice:
            return self.play_locked_choice()

        if self.count % 80 == 0:
            print("-- attack manager --")
            playing_subordinate = self._subordinates[CombatManager._ATTACK_MANAGER]
        elif self.count % 4 == 0:
            print("-- army supplier --")
            playing_subordinate = self._subordinates[CombatManager._ARMY_SUPPLIER]
        else:
            print("-- micro manager --")
            playing_subordinate = self._subordinates[CombatManager._MICRO_MANAGER]

        self.count += 1
        return playing_subordinate