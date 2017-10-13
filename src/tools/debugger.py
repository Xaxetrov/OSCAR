from pysc2.lib import features
from pysc2.lib import actions

import time


_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


class Debugger:

    def debug(self, obs, sleeping_time=21):
        print("Features - player relative index:", _PLAYER_RELATIVE)
        print("Actions - select army id:", actions.FUNCTIONS.select_army.id)
        print("Actions - move screen id:", actions.FUNCTIONS.Move_screen.id)
        # print("Obs - observation screen player relative:", (obs.observation["screen"][_PLAYER_RELATIVE] == 3).nonzero())
        time.sleep(sleeping_time)

