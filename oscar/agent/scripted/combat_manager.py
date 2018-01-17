from oscar.agent.commander.base_commander import BaseCommander
from oscar.constants import *


class CombatManager(BaseCommander):

    _ARMY_SUPPLIER = 0
    _ATTACK_MANAGER = 1

    def __init__(self, subordinates):
        self.count = 0
        super().__init__(subordinates)

    def choose_subordinate(self, obs, locked_choice):
        if locked_choice:
            return self.play_locked_choice()

        if self._shared['env'].timestamp % 5 == 0:
            playing_subordinate = self._subordinates[CombatManager._ATTACK_MANAGER]
        else:
            playing_subordinate = self._subordinates[CombatManager._ARMY_SUPPLIER]

        return playing_subordinate