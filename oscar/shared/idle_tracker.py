import numpy as np
import random
from oscar.constants import *
from oscar.util.point import Point
from oscar.shared.minimap import Minimap
from oscar.shared.screen import Screen
from oscar.shared.camera import Camera
from oscar.util.unit import Unit
from oscar.util.location import Location


""" Keeps track of idle units. """
class IdleTracker():

    """
    Number of consecutive steps a unit should be seen at the same place
    to assess it's idle.
    """
    _MINIMAP_IDLE_STEPS = 5
    _SCREEN_IDLE_STEPS = 10

    # states
    _INITIAL_STATE = 0
    _SCAN_SCREEN_STATE = 1

    def __init__(self):
        self._last_obs = [None] * IdleTracker._MINIMAP_IDLE_STEPS # circular array

        self._idle_units = []
        self._idle_units_map = None
        self._blacklist_map = None

        """ state machine """
        self._state = None
        self._candidate = None
        self._scan_count = None
        self._candidate_list = None
        self.reset_search()

    def update(self, obs, shared):
        """
        Updates tracking data using minimap view.
        Should be called at the beginning of every step.
        """
        self._last_obs[shared['env'].timestamp % IdleTracker._MINIMAP_IDLE_STEPS] = obs

        if self._idle_units_map is None:
            self._idle_units_map = np.zeros((shared['minimap'].width(obs), shared['minimap'].height(obs)))
        if self._blacklist_map is None:
            self._blacklist_map = np.zeros((shared['minimap'].width(obs), shared['minimap'].height(obs)))

        self._update_idle_units_map(obs, shared)
        self._update_blacklist_map(obs, shared)

    def reset_search(self):
        """
        Cancels search in progress,
        but keeps in memory found idle units.
        """
        self._state = IdleTracker._INITIAL_STATE
        self._scan_count = 0
        self._candidate_list = None

    def search_idle_unit(self, obs, shared, unit_ids=None, target=None, max_dist=None):
        """
         Searches for idle units, close to a given location.
         To do so, checks the minimap, moves the camera and observes the screen during multiple steps.
         Needs to be called multiple times at successive steps with the same arguments.

         :param unit_ids: ids of units which should be found
         :param target: location close to which units should be found
         :param max_dist: maximum distance from the point a unit could be found

         :return: a dictionary with the following fields:
             'actions': actions to be performed to keep going the search (could be None)
             'unit': the point of a found unit if one has been found at this step (could be None)
         """

        response = {
            'actions': None,
            'unit': None
        }

        if self._state == IdleTracker._INITIAL_STATE:
            self._candidate = self._get_best_idle_candidate(obs, shared, unit_ids, target, max_dist)
            if self._candidate: # moves camera to observe candidate
                response['actions'] = [actions.FunctionCall(MOVE_CAMERA, 
                    [self._candidate.location.minimap.to_array()])]
                self._state = IdleTracker._SCAN_SCREEN_STATE
            else: # no candidate: search failed
                return response

        elif self._state == IdleTracker._SCAN_SCREEN_STATE:
            friendly_units = self._get_friendly_units_on_screen(obs, shared)

            if self._candidate_list is None:
                self._candidate_list = friendly_units
            else:
                # Takes intersection of previous and new scanned units
                new_candidate_list = []
                for u in friendly_units:
                    for v in self._candidate_list:
                        if u.equals(v):
                            new_candidate_list.append(u)
                self._candidate_list = new_candidate_list

            if len(self._candidate_list) == 0: # No idle unit on the screen
                if self._candidate in self._idle_units:
                    self._idle_units.remove(self._candidate)
                self._blacklist_screen(obs, shared)
                self.reset_search()
            else:
                self._scan_count += 1

                if self._scan_count >= IdleTracker._SCREEN_IDLE_STEPS: # idle units found on screen
                    for u in self._candidate_list:
                        if self._blacklist_map[u.location.minimap.x, u.location.minimap.y] == 0:
                            if u.unit_id in TERRAN_BUIDINGS:
                                self._blacklist_map[u.location.minimap.x, u.location.minimap.y] = 1
                            elif not response['unit'] \
                                and (unit_ids is None or u.unit_id in unit_ids):
                                response['unit'] = u
                                if u in self._idle_units:
                                    self._idle_units.remove(u)
                                self._idle_units_map[u.location.minimap.x, u.location.minimap.y] = 0
                                self._blacklist_map[u.location.minimap.x, u.location.minimap.y] = 1
                            else:
                                if self._idle_units_map[u.location.minimap.x, u.location.minimap.y] == 0:
                                    self._idle_units.append(u)
                                    self._idle_units_map[u.location.minimap.x, u.location.minimap.y] = 1

                    self.reset_search()
                else:
                    response['actions'] = [actions.FunctionCall(NO_OP, [])]

        if not response['actions'] and not response['unit']:
            return self.search_idle_unit(obs, shared, unit_ids, target, max_dist)
        else:
            return response

    @staticmethod
    def _get_friendly_units_on_screen(obs, shared):
        scanned = shared['screen'].scan(obs)

        friendly_units = []
        for s in scanned:
            if s.player_id == PLAYER_SELF and s.unit_id not in TERRAN_BUIDINGS:
                loc = Location(screen_loc=s.location.screen, camera_loc=shared['camera'].location(obs))
                loc.minimap = shared['camera'].location_from_screen(obs, shared, loc.camera, loc.screen)
                friendly_units.append(Unit(loc, s.unit_id))
        return friendly_units

    def _update_idle_units_map(self, obs, shared):
        cur_minimap = obs.observation['minimap'][MINI_PLAYER_RELATIVE]
        for x in range(shared['minimap'].width(obs)):
            for y in range(shared['minimap'].height(obs)):
                if cur_minimap[y, x] != PLAYER_SELF \
                    and self._idle_units_map[x, y] != 0:
                    self._idle_units_map[x, y] = 0

                    for u in self._idle_units:
                        if u.location.minimap.equals(Point(x, y)):
                            self._idle_units.remove(u)

    def _update_blacklist_map(self, obs, shared):
        cur_minimap = obs.observation['minimap'][MINI_PLAYER_RELATIVE]
        for x in range(shared['minimap'].width(obs)):
            for y in range(shared['minimap'].height(obs)):
                if cur_minimap[y, x] != PLAYER_SELF \
                    and self._blacklist_map[x, y] != 0:
                    self._blacklist_map[x, y] = 0

    def _blacklist_screen(self, obs, shared):
        camera = obs.observation['minimap'][MINI_CAMERA]
        player_relative = obs.observation['minimap'][MINI_PLAYER_RELATIVE]

        for x in range(shared['minimap'].width(obs)):
            for y in range(shared['minimap'].height(obs)):
                if camera[y, x] == 1 and player_relative[y, x] == PLAYER_SELF:
                    self._blacklist_map[x, y] = 1

    def _get_new_candidates_from_minimap(self, obs, shared):
        friendly_fixed_points = []

        for x in range(shared['minimap'].width(obs)):
            for y in range(shared['minimap'].height(obs)):

                if self._blacklist_map[x, y] != 0 or self._idle_units_map[x, y] != 0:
                    continue

                is_idle = True
                for i in range(IdleTracker._MINIMAP_IDLE_STEPS):
                    if not self._last_obs[i]:
                        return []
                    if self._last_obs[i].observation['minimap'][MINI_PLAYER_RELATIVE][y, x] != PLAYER_SELF:
                        is_idle = False
                        break

                if is_idle:
                    friendly_fixed_points.append(
                        Unit(Location(minimap_loc=Point(x, y))))

        return friendly_fixed_points

    def _get_best_idle_candidate(self, obs, shared, unit_ids=None, target=None, max_dist=None):
        """
        Returns minimap position of an idle unit candidate.
        """

        best_candidate, min_dist = None, None

        """ selects best identified idle unit if it exists """
        for u in self._idle_units:
            if self._blacklist_map[u.location.minimap.x, u.location.minimap.y] == 1:
                self._idle_units.remove(u)
                self._idle_units_map[u.location.minimap.x, u.location.minimap.y] = 0
            else:
                if unit_ids is None or u.unit_id in unit_ids:
                    if not target:
                        best_candidate = u
                        break
                    else:
                        dist = target.distance(u.location.minimap)
                        if (not max_dist or dist <= max_dist) and \
                            (not min_dist or dist < min_dist):
                            best_candidate = u
                            min_dist = dist

        """ if there is no identified idle unit, try to find a candidate with the minimap """
        if not best_candidate:
            minimap_candidates = self._get_new_candidates_from_minimap(obs, shared)
            random.shuffle(minimap_candidates)
            for u in minimap_candidates:
                if not target:
                    best_candidate = u
                    break
                else:
                    dist = target.distance(u.location.minimap)
                    if (not max_dist or dist <= max_dist) and \
                        (not min_dist or dist < min_dist):
                        best_candidate = u
                        min_dist = dist

        return best_candidate
