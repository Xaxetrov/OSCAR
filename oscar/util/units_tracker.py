import numpy as np
from oscar.constants import *

""" Stores a scanned unit data. """
class Unit_scan():

    def __init__(self, _screen_location, _unit_id):
        self.screen_location = _screen_location
        self.unit_id = _unit_id


""" Stores results of a screen scan. """
class Screen_scan():

    def __init__(self, _timestamp, _minimap_location):
        self.timestamp = _timestamp
        self.minimap_location = _minimap_location
        self.self = []
        self.hostile = []
        self.neutral = []

    def add_unit(self, screen_location, unit_id, player_relative):
        unit_scan = Unit_scan(screen_location, unit_id)

        if player_relative == PLAYER_SELF:
            self.self.append(unit_scan)
        elif player_relative == PLAYER_HOSTILE:
            self.hostile.append(unit_scan)
        else:
            self.neutral.append(unit_scan)


""" Keeps track of units location. """
class Units_tracker():

    def __init__(self):
        self.scans = []


    """
    Computes and stores approximate location of units on screen.
    :param minimap_location: the current location of the minimap view.
    :return: a Screen_scan object
    """
    def scan_screen(self, obs, minimap_location):
        timestamp = obs.observation["score_cumulative"][1]
        screen_scan = Screen_scan(timestamp, minimap_location)

        screen_size = {
            'x': len(obs.observation['screen'][SCREEN_PLAYER_RELATIVE]),
            'y': len(obs.observation['screen'][SCREEN_PLAYER_RELATIVE][0])
        }

        scanned = np.zeros((screen_size['x'], screen_size['y']))

        """
        Explores contiguous pixels corresponding to the same unit
        and returns approximate center of the unit.
        """
        def explore_contiguous(x, y, unit_id, player_relative, scanned):
            sum_x, sum_y, nb_scanned = x, y, 1
            queue = []
            queue.append((x, y))

            def try_explore(x, y, queue, scanned):
                if x < screen_size['x'] and y < screen_size['y'] \
                    and scanned[x, y] == 0 \
                    and obs.observation['screen'][SCREEN_PLAYER_RELATIVE][x, y] == player_relative \
                    and obs.observation['screen'][SCREEN_UNIT_TYPE][x, y] == unit_id:
                    queue.append((x, y))
                    scanned[x, y] = 1
                    return True
                else:
                    return False

            while len(queue) > 0:
                cur = queue.pop()
                if try_explore(cur[0]+1, cur[1], queue, scanned):
                    sum_x += cur[0]+1
                    sum_y += cur[1]
                    nb_scanned += 1
                if try_explore(cur[0], cur[1]+1, queue, scanned):
                    sum_x += cur[0]
                    sum_y += cur[1]+1
                    nb_scanned += 1

            return sum_x/nb_scanned, sum_y/nb_scanned

        for x in range(screen_size['x']):
            for y in range(screen_size['y']):
                if scanned[x, y] == 0 \
                    and obs.observation['screen'][SCREEN_UNIT_TYPE][x, y] != 0:
                    
                    screen_scan.add_unit(
                        explore_contiguous(x, y,
                            obs.observation['screen'][SCREEN_UNIT_TYPE][x, y],
                            obs.observation['screen'][SCREEN_PLAYER_RELATIVE][x, y],
                            scanned
                        ),
                        obs.observation['screen'][SCREEN_UNIT_TYPE][x, y],
                        obs.observation['screen'][SCREEN_PLAYER_RELATIVE][x, y]
                    )

        self.scans.insert(0, screen_scan)
        return screen_scan


    """
    Returns the timestamp of the last scan of a given area.
    If the area has not been scanned recently, returns None.
    :param x_min, x_max, y_min, y_max: area to retrieve scans from
    :param from_timestamp: time to start retrieving from
    :return: a timestamp
    """
    def get_last_scan_time(self, x_min, x_max, y_min, y_max, from_timestamp = 0):
        for s in self.scans:
            if s.timestamp < from_timestamp:
                return None

            if s.minimap_location.x >= x_min \
                and s.minimap_location.x <= x_max \
                and s.minimap_location.y >= y_min \
                and s.minimap_location.y <= y_max:
                return s.timestamp


    """
    Retrieves most recent scans, in a given area.
    :param x_min, x_max, y_min, y_max: area to retrieve scans from
    :param from_timestamp: time to start retrieving from
    :return: a list of scans
    """
    def get_scans(self, x_min, x_max, y_min, y_max, from_timestamp = 0):
        res = []

        for s in self.scans:
            if s.timestamp < from_timestamp:
                break

            if s.minimap_location.x >= x_min \
                and s.minimap_location.x <= x_max \
                and s.minimap_location.y >= y_min \
                and s.minimap_location.y <= y_max:
                res.append(s)

        return res

