import numpy as np
from oscar.util.point import Point
from oscar.constants import *


class Camera(object):

    def __init__(self):
        """ cache data """
        self._location = None
        self._location_timestamp = None
        self._width = None
        self._height = None

    def location(self, obs, shared):
        if not self._location_timestamp or self._location_timestamp != shared['env'].timestamp:
            mini_camera = obs.observation['minimap'][MINI_CAMERA]
            camera_y, camera_x = (mini_camera == 1).nonzero()
            self._location = Point(round(sum(camera_x)/len(camera_x)), round(sum(camera_y)/len(camera_y)))
            self._location_timestamp = shared['env'].timestamp
        return self._location

    def width(self, obs):
        if not self._width:
            mini_camera = obs.observation['minimap'][MINI_CAMERA]
            _, camera_x = (mini_camera == 1).nonzero()
            self._width = max(camera_x) - min(camera_x) + 1
        return self._width

    def height(self, obs):
        if not self._height:
            mini_camera = obs.observation['minimap'][MINI_CAMERA]
            camera_y, _ = (mini_camera == 1).nonzero()
            self._height = max(camera_y) - min(camera_y) + 1
        return self._height

    def location_from_screen(self, obs, shared, camera_loc, screen_loc):
        scale_x = self.width(obs) / len(obs.observation['screen'][0])
        scale_y = self.height(obs) / len(obs.observation['screen'][0][0])

        return shared['minimap'].bound(obs, Point(
            int(round(camera_loc.x - 0.5*self.width(obs) + screen_loc.x * scale_x)),
            int(round(camera_loc.y - 0.5*self.height(obs) + screen_loc.y * scale_y))))

    def random_target(self, obs):
        minimap_width = len(obs.observation['minimap'][0])
        minimap_height = len(obs.observation['minimap'][0][0])

        loc = Point()
        loc.x = np.random.randint(int(self.width(obs)/2), minimap_width - int(self.width(obs)/2))
        loc.y = np.random.randint(int(self.height(obs)/2), minimap_height - int(self.height(obs)/2))
        return loc

    def iterate(self, obs, camera_loc=None):
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
            width = self.width(obs)
            height = self.height(obs)
            for y in range(int(camera_loc.y - height/2), int(camera_loc.y + height/2)+1):
                for x in range(int(camera_loc.x - width/2), int(camera_loc.x + width/2)+1):
                    yield(Point(x, y))
