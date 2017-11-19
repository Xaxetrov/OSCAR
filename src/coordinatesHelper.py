import numpy as np
import sys
import os
import json
import time
from random import randint
from location import Location

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_MINI_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_MINI_CAMERA = features.MINIMAP_FEATURES.camera.index

#############################
# Helper for coordinates.
# Input/output use obs coordinates system.
#############################
class CoordinatesHelper():

    def __init__(self):
        self.calibration = Calibration()
        try:
            self.calibration.load()
        except Exception as error:
            # TO DO
            pass

    def getMoveCameraAction(self, location):
        # translates input to camera coordinates sytem
        x_cam = location.x + self.calibration.cameraObsOffset['x']
        y_cam = location.y + self.calibration.cameraObsOffset['y']
        return actions.FunctionCall(_MOVE_CAMERA, [[x_cam, y_cam]])

    def getScreenCenter(self):
        return Location(self.fieldOfViewMap['x']/2, self.fieldOfViewMap['y']/2)

    # Computes and returns current location in minimap
    def getLocInMinimap(self, obs):
        minimap = obs.observation['minimap'][3]

        for row in range(0, self.minimapSize['x']):
            for col in range(0, self.minimapSize['y']):
                if minimap[row][col] == 1:
                    return Location(col, row)

    def getRandomMinimapLocation(self):
        loc = Location()
        loc.x = randint(self.obsRange['x']['min'], self.obsRange['x']['max'])
        loc.y = randint(self.obsRange['y']['min'], self.obsRange['y']['max'])
        return loc

    # Limits are inclusive
    def getMinimapViewLimits(self, pos):
        limits = {'x': {'min': None, 'max': None}, 
            'y': {'min': None, 'max': None}}

        viewSize = self.fieldOfViewMinimap
        limits['x']['min'] = pos.x
        limits['x']['max'] = pos.x + viewSize['x'] - 1
        limits['y']['min'] = pos.y
        limits['y']['max'] = pos.y + viewSize['y'] - 1

        return limits

    # Takes a location in the minimap and converts it to the global coordinates system.
    def minimapToGlobal(self, minimapLoc):
        p = Location()
        p.x = round(minimapLoc.x * self.fieldOfViewMap['x'] / self.fieldOfViewMinimap['x'])
        p.y = round(minimapLoc.y * self.fieldOfViewMap['y'] / self.fieldOfViewMinimap['y'])
        return p

    def globalToMinimap(self, globalLoc):
        p = Location()
        p.x = round(globalLoc.x * self.fieldOfViewMinimap['x'] / self.fieldOfViewMap['x'])
        p.y = round(globalLoc.y * self.fieldOfViewMinimap['y'] / self.fieldOfViewMap['y'])
        return p

    def bound(self, loc):
        if loc.x < self.obsRange['x']['min']:
            loc.x = self.obsRange['x']['min']
        elif loc.x > self.obsRange['x']['max']:
            loc.x = self.obsRange['x']['max']
        if loc.y < self.obsRange['y']['min']:
            loc.y = self.obsRange['y']['min']
        elif loc.y > self.obsRange['y']['max']:
            loc.y = self.obsRange['y']['max']
        return loc

    @property
    def minimapSize(self):
        return self.calibration.minimapSize

    @property
    def fieldOfViewMap(self):
        return self.calibration.fieldOfViewMap

    @property
    def fieldOfViewMinimap(self):
        return self.calibration.fieldOfViewMinimap

    @property
    def obsRange(self):
        return self.calibration.obsRange

      

#############################
# Manages maps calibration
#############################
class Calibration():

    PATH = "./data/camera/"

    def __init__(self):
        np.set_printoptions(threshold=np.nan)

        self._minimapSize = {'x': None, 'y': None}
        self._fieldOfViewMap = {'x': None, 'y': None}
        self._fieldOfViewMinimap = {'x': None, 'y': None}
        self._moveRange = {'x': {'min': None, 'max': None}, 'y': {'min': None, 'max': None}}
        self._obsRange = {'x': {'min': float('inf'), 'max': float('-inf')}, 'y': {'min': float('inf'), 'max': float('-inf')}}
        self._cameraObsOffset = {'x': None, 'y': None}
        
        self._cameraPos = {'x': 0, 'y': 0}
        self._lastObs = {'x': None, 'y': None}
        self._prevOffset = {'x': None, 'y': None}
        self._completed = False

        self._mapName = None
        with open("../tmp/map_name.txt") as f:
            self._mapName = f.read().strip('\n') + ".json"

    def load(self):
        try:
            data = json.load(open(self.PATH + self._mapName))
        except Exception as error:
            raise Exception('Calibration file not found for this map.')
            return

        self._minimapSize = data['minimapSize']
        self._fieldOfViewMap = data['fieldOfViewMap']
        self._fieldOfViewMinimap = data['fieldOfViewMinimap']
        self._moveRange = data['moveRange']
        self._obsRange = data['obsRange']
        self._cameraObsOffset = data['cameraObsOffset']

    def export(self):
        data = {'minimapSize': self._minimapSize,
            'fieldOfViewMap': self._fieldOfViewMap,
            'fieldOfViewMinimap': self._fieldOfViewMinimap,
            'moveRange': self._moveRange,
            'obsRange': self._obsRange,
            'cameraObsOffset': self._cameraObsOffset}

        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)

        with open(self.PATH + self._mapName, 'w') as outfile:
            json.dump(data, outfile)

    @property
    def minimapSize(self):
        return self._minimapSize
    @property
    def fieldOfViewMap(self):
        return self._fieldOfViewMap
    @property
    def fieldOfViewMinimap(self):
        return self._fieldOfViewMinimap
    @property
    def moveRange(self):
        return self._moveRange
    @property
    def obsRange(self):
        return self._obsRange
    @property
    def cameraObsOffset(self):
        return self._cameraObsOffset

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass   

    # Step for camera calibration
    def step(self, obs):

        # Determines minimap size
        if not self._minimapSize['x'] or not self._minimapSize['y']:
            self._minimapSize['x'] = len(obs.observation["minimap"][_MINI_VISIBILITY])
            self._minimapSize['y'] = len(obs.observation["minimap"][_MINI_VISIBILITY][0])

        if not self._completed:
            minimap_y, minimap_x = obs.observation["minimap"][_MINI_CAMERA].nonzero()

            # Determines map field of view
            if not self._fieldOfViewMap['x']:
                self._fieldOfViewMap['x'] = len(obs.observation["screen"][0][0])
                self._fieldOfViewMap['y'] = len(obs.observation["screen"][0])

            # Determines minimap field of view
            if not self._fieldOfViewMinimap['x']:
                for i in range(1, len(minimap_x)):
                    if (minimap_x[i] < minimap_x[i-1]):
                        self._fieldOfViewMinimap['x'] = i
                        break
                for i in range(1, len(minimap_y)):
                    if (minimap_y[i] > minimap_y[i-1]):
                        self._fieldOfViewMinimap['y'] = i
                        break

            # Computes move/obs offset
            offset = {'x':  self._cameraPos['x'] - minimap_x[0], 'y':  self._cameraPos['y'] - minimap_y[0]}
            if not self._cameraObsOffset['x'] and self._prevOffset['x'] == offset['x']:
                self._cameraObsOffset['x'] = offset['x'].item()
            self._prevOffset['x'] = offset['x']
            if not self._cameraObsOffset['y'] and self._cameraPos['y'] > 0 and self._prevOffset['y'] == offset['y']:
                self._cameraObsOffset['y'] = offset['y'].item()
            self._prevOffset['y'] = offset['y']

            self._lastObs['x'] = minimap_x[0].item()
            self._lastObs['y'] = minimap_y[0].item()

            self._obsRange['x']['min'] = min(self._obsRange['x']['min'], self._lastObs['x'])
            self._obsRange['x']['max'] = max(self._obsRange['x']['max'], self._lastObs['x'])
            self._obsRange['y']['min'] = min(self._obsRange['y']['min'], self._lastObs['y'])
            self._obsRange['y']['max'] = max(self._obsRange['y']['max'], self._lastObs['y'])

            if self._cameraPos['y'] > 0 and self._cameraPos['y'] + 1 < self._minimapSize['y']:
                self._cameraPos['y'] += 1
            elif self._cameraPos['x'] + 1 < self._minimapSize['x']:
                self._cameraPos['x'] += 1
            elif self._cameraPos['y'] == 0:
                self._cameraPos['y'] = 1
            else:
                self._completed = True
                self._moveRange['x']['min'] = self._obsRange['x']['min'] + self._cameraObsOffset['x']
                self._moveRange['x']['max'] = self._obsRange['x']['max'] + self._cameraObsOffset['x']
                self._moveRange['y']['min'] = self._obsRange['y']['min'] + self._cameraObsOffset['y']
                self._moveRange['y']['max'] = self._obsRange['y']['max'] + self._cameraObsOffset['y']
                self.export()
                print("\n--- Calibration completed and saved to " + self.PATH + self._mapName + " ---\n")

        return actions.FunctionCall(_MOVE_CAMERA, [[self._cameraPos['x'], self._cameraPos['y']]])