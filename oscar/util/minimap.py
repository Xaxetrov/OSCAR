import numpy as np
from oscar.util.point import Point
from oscar.util.camera import Camera
from oscar.util.screen import Screen
from oscar.util.location import Location


class Minimap(object):

    @staticmethod
    def width(obs):
        return len(obs.observation['minimap'][0])

    @staticmethod
    def height(obs):
        return len(obs.observation['minimap'][0][0])

    @staticmethod
    def random_point(obs):
        loc = Point()
        loc.x = np.random.randint(0, Minimap.width(obs))
        loc.y = np.random.randint(0, Minimap.height(obs))
        return loc

    @staticmethod
    def random_camera_target(obs):
        loc = Point()
        loc.x = np.random.randint(int(Camera.width(obs)/2), Minimap.width(obs) - int(Camera.width(obs)/2))
        loc.y = np.random.randint(int(Camera.height(obs)/2), Minimap.height(obs) - int(Camera.height(obs)/2))
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


    @staticmethod
    def compute_camera_location(obs, camera_loc, screen_loc):
        camera_width = Camera.width(obs)
        camera_height = Camera.height(obs)

        scale_x = camera_width / Screen.width(obs)
        scale_y = camera_height / Screen.height(obs)

        return Minimap.bound(obs, Point(
            int(round(camera_loc.x - 0.5*camera_width + screen_loc.x * scale_x)),
            int(round(camera_loc.y - 0.5*camera_height + screen_loc.y * scale_y))))