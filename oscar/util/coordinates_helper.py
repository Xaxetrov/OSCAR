import numpy as np
import sys
import os
import json
from oscar.util.location import Location
from oscar.constants import *

#############################
# Helper for coordinates.
#############################
class Coordinates_helper():

    def __init__(self):
        self._calibration = Calibration()
        try:
            self._calibration.load()
        except Exception as error:
            # TO DO
            pass

    # Returns the location the camera should move to
    # move to the location provided as parameter.
    def get_camera_move(self, location):
        x_cam = location.x + self._calibration.camera_obs_offset['x']
        y_cam = location.y + self._calibration.camera_obs_offset['y']
        return Location(x_cam, y_cam)

    def get_screen_center(self):
        return Location(self.field_of_view_map['x']/2, self.field_of_view_map['y']/2)

    # Computes and returns current location in minimap
    def get_loc_in_minimap(self, obs):
        minimap = obs.observation['minimap'][3]

        for row in range(0, self.minimap_size['x']):
            for col in range(0, self.minimap_size['y']):
                if minimap[row][col] == 1:
                    return Location(col, row)

    def get_random_minimap_location(self):
        loc = Location()
        if self.obs_range['x']['min'] == self.obs_range['x']['max']:
            loc.x = self.obs_range['x']['min']
        else:
            loc.x = np.random.randint(self.obs_range['x']['min'], self.obs_range['x']['max'])

        if self.obs_range['y']['min'] == self.obs_range['y']['max']:
            loc.y = self.obs_range['y']['min']
        else:
            loc.y = np.random.randint(self.obs_range['y']['min'], self.obs_range['y']['max'])

        return loc

    # Limits are inclusive
    def get_minimap_view_limits(self, pos):
        limits = {'x': {'min': None, 'max': None}, 
            'y': {'min': None, 'max': None}}

        view_size = self.field_of_view_minimap
        limits['x']['min'] = pos.x
        limits['x']['max'] = pos.x + view_size['x'] - 1
        limits['y']['min'] = pos.y
        limits['y']['max'] = pos.y + view_size['y'] - 1

        return limits

    # Takes a location in the minimap and converts it to the global coordinates system.
    def minimap_to_global(self, minimap_loc):
        p = Location()
        p.x = round(minimap_loc.x * self.field_of_view_map['x'] / self.field_of_view_minimap['x'])
        p.y = round(minimap_loc.y * self.field_of_view_map['y'] / self.field_of_view_minimap['y'])
        return p

    # Takes a location in the global coordinates system and converts it to minimap coordinates.
    def global_to_minimap(self, global_loc):
        p = Location()
        p.x = round(global_loc.x * self.field_of_view_minimap['x'] / self.field_of_view_map['x'])
        p.y = round(global_loc.y * self.field_of_view_minimap['y'] / self.field_of_view_map['y'])
        return p

    # If the location is allowed (in the map), returns it as is.
    # If not, returns the closest allowed location.
    def bound(self, loc):
        if loc.x < self.obs_range['x']['min']:
            loc.x = self.obs_range['x']['min']
        elif loc.x > self.obs_range['x']['max']:
            loc.x = self.obs_range['x']['max']
        if loc.y < self.obs_range['y']['min']:
            loc.y = self.obs_range['y']['min']
        elif loc.y > self.obs_range['y']['max']:
            loc.y = self.obs_range['y']['max']
        return loc

    @property
    def minimap_size(self):
        return self._calibration.minimap_size

    @property
    def field_of_view_map(self):
        return self._calibration.field_of_view_map

    @property
    def field_of_view_minimap(self):
        return self._calibration.field_of_view_minimap

    @property
    def obs_range(self):
        return self._calibration.obs_range

      

#############################
# Manages maps calibration
#############################
class Calibration():

    PATH = "data/camera/"

    def __init__(self):
        self._minimap_size = {'x': None, 'y': None}
        self._field_of_view_map = {'x': None, 'y': None}
        self._field_of_view_minimap = {'x': None, 'y': None}
        self._move_range = {'x': {'min': None, 'max': None}, 'y': {'min': None, 'max': None}}
        self._obs_range = {'x': {'min': float('inf'), 'max': float('-inf')}, 'y': {'min': float('inf'), 'max': float('-inf')}}
        self._camera_obs_offset = {'x': None, 'y': None}
        
        self._camera_pos = {'x': 0, 'y': 0}
        self._last_obs = {'x': None, 'y': None}
        self._prev_offset = {'x': None, 'y': None}
        self._completed = False

        self._map_name = None
        with open("data/tmp/map_name.txt") as f:
            self._map_name = f.read().strip('\n') + ".json"

    def load(self):
        try:
            data = json.load(open(self.PATH + self._map_name))
        except Exception as error:
            raise Exception('Calibration file not found for this map.')
            return

        self._minimap_size = data['minimap_size']
        self._field_of_view_map = data['field_of_view_map']
        self._field_of_view_minimap = data['field_of_view_minimap']
        self._move_range = data['move_range']
        self._obs_range = data['obs_range']
        self._camera_obs_offset = data['camera_obs_offset']

    def export(self):
        data = {'minimap_size': self._minimap_size,
            'field_of_view_map': self._field_of_view_map,
            'field_of_view_minimap': self._field_of_view_minimap,
            'move_range': self._move_range,
            'obs_range': self._obs_range,
            'camera_obs_offset': self._camera_obs_offset}

        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)

        with open(self.PATH + self._map_name, 'w') as outfile:
            json.dump(data, outfile)

    @property
    def minimap_size(self):
        return self._minimap_size
    @property
    def field_of_view_map(self):
        return self._field_of_view_map
    @property
    def field_of_view_minimap(self):
        return self._field_of_view_minimap
    @property
    def move_range(self):
        return self._move_range
    @property
    def obs_range(self):
        return self._obs_range
    @property
    def camera_obs_offset(self):
        return self._camera_obs_offset

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass   

    # Step for camera calibration
    def step(self, obs):

        # Determines minimap size
        if not self._minimap_size['x'] or not self._minimap_size['y']:
            self._minimap_size['x'] = len(obs.observation["minimap"][MINI_VISIBILITY])
            self._minimap_size['y'] = len(obs.observation["minimap"][MINI_VISIBILITY][0])

        if not self._completed:
            minimap_y, minimap_x = obs.observation["minimap"][MINI_CAMERA].nonzero()

            # Determines map field of view
            if not self._field_of_view_map['x']:
                self._field_of_view_map['x'] = len(obs.observation["screen"][0][0])
                self._field_of_view_map['y'] = len(obs.observation["screen"][0])

            # Determines minimap field of view
            if not self._field_of_view_minimap['x']:
                for i in range(1, len(minimap_x)):
                    if (minimap_x[i] < minimap_x[i-1]):
                        self._field_of_view_minimap['x'] = i
                        break
                for i in range(1, len(minimap_y)):
                    if (minimap_y[i] > minimap_y[i-1]):
                        self._field_of_view_minimap['y'] = i
                        break

            # Computes move/obs offset
            offset = {'x':  self._camera_pos['x'] - minimap_x[0], 'y':  self._camera_pos['y'] - minimap_y[0]}
            if not self._camera_obs_offset['x'] and self._prev_offset['x'] == offset['x']:
                self._camera_obs_offset['x'] = offset['x'].item()
            self._prev_offset['x'] = offset['x']
            if not self._camera_obs_offset['y'] and self._camera_pos['y'] > 0 and self._prev_offset['y'] == offset['y']:
                self._camera_obs_offset['y'] = offset['y'].item()
            self._prev_offset['y'] = offset['y']

            self._last_obs['x'] = minimap_x[0].item()
            self._last_obs['y'] = minimap_y[0].item()

            self._obs_range['x']['min'] = min(self._obs_range['x']['min'], self._last_obs['x'])
            self._obs_range['x']['max'] = max(self._obs_range['x']['max'], self._last_obs['x'])
            self._obs_range['y']['min'] = min(self._obs_range['y']['min'], self._last_obs['y'])
            self._obs_range['y']['max'] = max(self._obs_range['y']['max'], self._last_obs['y'])

            if self._camera_pos['y'] > 0 and self._camera_pos['y'] + 1 < self._minimap_size['y']:
                self._camera_pos['y'] += 1
            elif self._camera_pos['x'] + 1 < self._minimap_size['x']:
                self._camera_pos['x'] += 1
            elif self._camera_pos['y'] == 0:
                self._camera_pos['y'] = 1
            else:
                self._completed = True
                self._move_range['x']['min'] = self._obs_range['x']['min'] + self._camera_obs_offset['x']
                self._move_range['x']['max'] = self._obs_range['x']['max'] + self._camera_obs_offset['x']

                if not self._camera_obs_offset['y']:
                    self._camera_obs_offset['y'] = offset['y'].item()

                self._move_range['y']['min'] = self._obs_range['y']['min'] + self._camera_obs_offset['y']
                self._move_range['y']['max'] = self._obs_range['y']['max'] + self._camera_obs_offset['y']
                self.export()
                print("\n--- Calibration completed and saved to " + self.PATH + self._map_name + " ---\n")

        return actions.FunctionCall(MOVE_CAMERA, [[self._camera_pos['x'], self._camera_pos['y']]])