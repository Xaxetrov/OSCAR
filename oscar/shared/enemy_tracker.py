from oscar.constants import *


class ScreenScan:
    """ Stores results of a screen scan. """

    def __init__(self, timestamp, camera_loc):
        self.timestamp = timestamp
        self.camera_loc = camera_loc
        self.self = []
        self.hostile = []
        self.neutral = []

    def add_unit(self, unit):
        if unit.player_id == PLAYER_SELF:
            self.self.append(unit)
        elif unit.player_id == PLAYER_HOSTILE:
            self.hostile.append(unit)
        else:
            self.neutral.append(unit)


class EnemyTracker:
    """
    Keeps track of the enemy by moving units in the opponent base
    and scanning the content of the screen.
    """

    def __init__(self):
        self.scans = []

    """
    Computes and stores the approximate location of units on screen.
    """
    def scan_screen(self, obs, shared):
        screen_scan = ScreenScan(shared['env'].timestamp, shared['camera'].location(obs, shared))

        units = shared['screen'].scan(obs, shared)
        for u in units:
            screen_scan.add_unit(u)

        self.scans.insert(0, screen_scan)

    def get_last_scan_time(self, x_min, x_max, y_min, y_max, from_timestamp=0):
        """
        Returns the timestamp of the last scan of a given area.
        If the area has not been scanned recently, returns None.
        :param x_min, x_max, y_min, y_max: area to retrieve scans from
        :param from_timestamp: time to start retrieving from
        :return: a timestamp
        """
        for s in self.scans:
            if s.timestamp < from_timestamp:
                return None

            if s.camera_loc.x >= x_min \
                and s.camera_loc.x <= x_max \
                and s.camera_loc.y >= y_min \
                and s.camera_loc.y <= y_max:
                return s.timestamp

    def get_scans(self, x_min, x_max, y_min, y_max, from_timestamp=0):
        """
        Retrieves most recent scans, in a given area.
        :param x_min, x_max, y_min, y_max: area to retrieve scans from
        :param from_timestamp: time to start retrieving from
        :return: a list of scans
        """
        for s in self.scans:
            if s.timestamp < from_timestamp:
                return

            if s.camera_loc.x >= x_min \
                and s.camera_loc.x <= x_max \
                and s.camera_loc.y >= y_min \
                and s.camera_loc.y <= y_max:
                yield s

