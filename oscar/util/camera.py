import numpy as np
from oscar.util.point import Point
from oscar.constants import *


class Camera(object):

    @staticmethod
    def location(obs):
        mini_camera = obs.observation['minimap'][MINI_CAMERA]
        camera_y, camera_x = (mini_camera == 1).nonzero()
        return Point(round(sum(camera_x)/len(camera_x)), round(sum(camera_y)/len(camera_y)))

    @staticmethod
    def width(obs):
        mini_camera = obs.observation['minimap'][MINI_CAMERA]
        _, camera_x = (mini_camera == 1).nonzero()
        return max(camera_x) - min(camera_x) + 1

    @staticmethod
    def height(obs):
        mini_camera = obs.observation['minimap'][MINI_CAMERA]
        camera_y, _ = (mini_camera == 1).nonzero()
        return max(camera_y) - min(camera_y) + 1

    @staticmethod
    def iterate(obs, camera_loc=None):
        """ Generates camera pixels location """
        mini_camera = obs.observation['minimap'][MINI_CAMERA]

        if not camera_loc:
            camera_y, camera_x = (mini_camera == 1).nonzero()
            min_x, max_x = min(camera_x), max(camera_x)
            min_y, max_y = min(camera_y), max(camera_y)

            for y in range(min_y, max_y+1):
                for x in range(min_x, max_x+1):
                    yield(Point(x, y))

        else:
            width = Camera.width(obs)
            height = Camera.height(obs)
            for y in range(int(camera_loc.y - height/2), int(camera_loc.y + height/2)+1):
                for x in range(int(camera_loc.x - width/2), int(camera_loc.x + width/2)+1):
                    yield(Point(x, y))
