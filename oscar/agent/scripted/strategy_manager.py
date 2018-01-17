from oscar.agent.commander.base_commander import BaseCommander
from oscar.constants import *
from oscar.util.location import Location


class StrategyManager(BaseCommander):

    _ECONOMY_MANAGER = 0
    _ARMY_MANAGER = 1
    _SCOUT = 2


    def __init__(self, subordinates):
        self.count = 0
        super().__init__(subordinates)

    def choose_subordinate(self, obs, locked_choice):

        """ Stores command center location """
        if self._shared['env'].timestamp == 0 \
            and len(self._shared['economy'].command_centers) == 0:
            for u in self._shared['screen'].scan_units(obs, self._shared, [TERRAN_COMMAND_CENTER], PLAYER_SELF):
                self._shared['economy'].add_command_center(obs, self._shared,
                    Location(screen_loc=u.location.screen, camera_loc=self._shared['camera'].location(obs, self._shared)))

        self._shared['env'].timestamp += 1

        if locked_choice:
            return self.play_locked_choice()

        if self.count % 50 == 0:
            playing_subordinate = self._subordinates[StrategyManager._SCOUT]
        elif self.count % 2 == 0:
            playing_subordinate = self._subordinates[StrategyManager._ECONOMY_MANAGER]
        else:
            playing_subordinate = self._subordinates[StrategyManager._ARMY_MANAGER]

        self.count += 1
        return playing_subordinate