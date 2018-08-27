import numpy as np
from oscar.util.point import Point
from oscar.constants import MINIMAP


class Minimap(object):

    @staticmethod
    def width(obs):
        return len(obs.observation[MINIMAP][0])

    @staticmethod
    def height(obs):
        return len(obs.observation[MINIMAP][0][0])

    @staticmethod
    def random_point(obs):
        loc = Point()
        loc.x = np.random.randint(0, Minimap.width(obs))
        loc.y = np.random.randint(0, Minimap.height(obs))
        return loc

    @staticmethod
    def bound(obs, point):
        if point.x < 0:
            point.x = 0
        elif point.x >= Minimap.width(obs):
            point.x = Minimap.width(obs)-1
        if point.y < 0:
            point.y = 0
        elif point.y >= Minimap.height(obs):
            point.y = Minimap.height(obs)-1
        return point