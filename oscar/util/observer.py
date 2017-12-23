import numpy as np
from oscar.constants import *
from oscar.util.screen_helper import *
from oscar.util.coordinates_helper import Coordinates_helper
from random import randint

""" 
Observes enemy units
and stores them for later use.
"""
class Observer():

    def __init__(self, coordinates_helper = None):
        self.coordinates_helper = coordinates_helper
        if not self.coordinates_helper:
            self.coordinates_helper = Coordinates_helper()

        self.last_observed_grid = np.zeros((self.coordinates_helper.minimap_size['x'], self.coordinates_helper.minimap_size['y']))
        self.observations = []

    """
    Heuristic to estimate how much a location worth being observed.
    Takes into account:
        - enemy units visible on the minimap
        - date of the last observation at this place

    Location are considered by their top-left corner.
    """
    def score_minimap_location(self, obs, minimap_location):
        score = 0.
        mini_player_relative = obs.observation["minimap"][MINI_PLAYER_RELATIVE]
        for x in range(minimap_location.x, minimap_location.x + self.coordinates_helper.field_of_view_minimap['x'] - 1):
            for y in range(minimap_location.y, minimap_location.y + self.coordinates_helper.field_of_view_minimap['y'] - 1):
                if mini_player_relative[x, y] == PLAYER_HOSTILE:
                    score += 1. /  (self.last_observed_grid(x, y) + 1)
        return score

    """
    Scans current screen by computing the position of enemy units.
    These positions are then stored as observations.
    """
    def scan_screen(self, obs, minimap_location):
        timestamp = obs.observation["score_cumulative"][1]

        # For each unit id, finds units location
        units_id = get_units_id(obs, PLAYER_HOSTILE)
        for id in units_id:
            centers = get_center(obs, id, PLAYER_HOSTILE)
            for c in centers:
                self.observations.append((c, timestamp))

        # Updates last_observed_grid
        for x in range(minimap_location[0], minimap_location[0] + self.coordinates_helper.minimap_size['x'] - 1):
            for y in range(minimap_location[1], minimap_location[1] + self.coordinates_helper.minimap_size['y'] - 1):   
                self.last_observed_grid[x, y] = timestamp




