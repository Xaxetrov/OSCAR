import random
import numpy as np
from oscar.constants import *
from oscar.util.point import Point


def get_micro_management_location(obs, shared):
    _MICRO_DISTANCE = 6

    player_relative = obs.observation['minimap'][MINI_PLAYER_RELATIVE]
    locations = []

    for y in range(shared['minimap'].height(obs)):
        for x in range(shared['minimap'].width(obs)):
            if player_relative[y, x] == PLAYER_SELF:
                p = Point(x, y)
                if _is_close_to_enemy(obs, shared, p, _MICRO_DISTANCE):
                    locations.append(p)

    return locations


def get_safe_screen_location(obs, shared, unit_point, influence_map):
    safe_x, safe_y = [], []

    safe_coeff = 0
    while len(safe_x) == 0:
        safe_x, safe_y = (influence_map == safe_coeff).nonzero()
        safe_coeff += 1

    best_loc, min_dist = None, None
    for loc in zip(safe_x, safe_y):
        dist = unit_point.distance(Point(loc[0], loc[1]))
        if not best_loc or dist < min_dist:
            best_loc = Point(loc[0], loc[1])
            min_dist = dist

    return best_loc


def get_enemy_influence_map(obs, shared):
    _ENEMY_DISTANCE_DANGER = 25

    influence_map = np.zeros((shared['screen'].width(obs), shared['screen'].height(obs)))
    enemies = shared['screen'].scan_units(obs, shared, list(TERRAN_UNITS), PLAYER_HOSTILE)

    for y in range(shared['screen'].height(obs)):
        for x in range(shared['screen'].width(obs)):
            for e in enemies:
                if Point(x, y).distance(e.location.screen) <= _ENEMY_DISTANCE_DANGER:
                    influence_map[x, y] += 1

    return influence_map


def get_closest_enemy(obs, shared, loc):
    player_relative = obs.observation['minimap'][MINI_PLAYER_RELATIVE]
    hostile_y, hostile_x = (player_relative == PLAYER_HOSTILE).nonzero()

    closest, min_dist = None, None
    for h in zip(hostile_x, hostile_y):
        dist = loc.distance(Point(h[0], h[1]))
        if not closest or dist < min_dist:
            closest = Point(h[0], h[1])
            min_dist = dist

    return closest


def _is_close_to_enemy(obs, shared, point, max_dist):
    player_relative = obs.observation['minimap'][MINI_PLAYER_RELATIVE]

    for y in range(point.y-max_dist, point.y+max_dist+1):
        if y < 0 or y >= shared['minimap'].height(obs):
            continue

        for x in range(point.x-max_dist, point.x+max_dist+1):
            if x < 0 or x >= shared['minimap'].width(obs):
                continue

            if player_relative[y, x] == PLAYER_HOSTILE:
                return True

    return False
