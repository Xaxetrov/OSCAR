import numpy as np
from oscar.constants import *
from oscar.util.unit import Unit
from oscar.util.location import Location
from oscar.util.point import Point


class Screen(object):

    def __init__(self):
        """ cache data """
        self._scanned_units = None
        self._scanned_units_timestamp = None

    @staticmethod
    def width(obs):
        return len(obs.observation['screen'][0])

    @staticmethod
    def height(obs):
        return len(obs.observation['screen'][0][0])

    @staticmethod
    def random_point(obs, margin=0):
        loc = Point()
        loc.x = np.random.randint(margin, Screen.width(obs)-margin-1)
        loc.y = np.random.randint(margin, Screen.height(obs)-margin-1)
        return loc

    def scan_units(self, obs, shared, unit_ids, player_id):
        scan = self.scan(obs, shared)
        return [u for u in scan if u.unit_id in unit_ids and u.player_id == player_id]

    def scan(self, obs, shared):
        """ Returns a list of the units on screen """

        if not self._scanned_units_timestamp or self._scanned_units_timestamp != shared['env'].timestamp:
            def _explore_contiguous(x, y, unit_id, player_relative, scanned):
                """
                Explores contiguous pixels corresponding to the same unit
                and returns approximate center of the unit.
                """

                def _try_explore(x, y, queue, scanned):
                    if x < screen_width and y < screen_height \
                        and scanned[x, y] == 0 \
                        and obs.observation['screen'][SCREEN_PLAYER_RELATIVE][x, y] == player_relative \
                        and obs.observation['screen'][SCREEN_UNIT_TYPE][x, y] == unit_id:
                        queue.append((x, y))
                        scanned[x, y] = 1
                        return True
                    else:
                        return False

                sum_x, sum_y, nb_scanned = x, y, 1
                queue = []
                queue.append((x, y))

                while len(queue) > 0:
                    cur = queue.pop()
                    if _try_explore(cur[0]+1, cur[1], queue, scanned):
                        sum_x += cur[0]+1
                        sum_y += cur[1]
                        nb_scanned += 1
                    if _try_explore(cur[0], cur[1]+1, queue, scanned):
                        sum_x += cur[0]
                        sum_y += cur[1]+1
                        nb_scanned += 1
                    if _try_explore(cur[0], cur[1]-1, queue, scanned):
                        sum_x += cur[0]
                        sum_y += cur[1]-1
                        nb_scanned += 1

                return round(sum_x/nb_scanned), round(sum_y/nb_scanned)

            screen_width = Screen.width(obs)
            screen_height = Screen.height(obs)
            scanned = np.zeros((screen_width, screen_height))
            self._scanned_units = []

            for x in range(screen_width):
                for y in range(screen_height):
                    if scanned[x, y] == 0 \
                        and obs.observation['screen'][SCREEN_UNIT_TYPE][x, y] != 0:
                        
                        center_x, center_y = _explore_contiguous(x, y,
                            obs.observation['screen'][SCREEN_UNIT_TYPE][x, y],
                            obs.observation['screen'][SCREEN_PLAYER_RELATIVE][x, y],
                            scanned)

                        self._scanned_units.append(Unit(
                            Location(screen_loc=Point(center_x, center_y)),
                            obs.observation['screen'][SCREEN_UNIT_TYPE][x, y],
                            obs.observation['screen'][SCREEN_PLAYER_RELATIVE][x, y]))

            self._scanned_units_timestamp = shared['env'].timestamp

        return self._scanned_units
