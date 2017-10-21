import numpy as np
import sys
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MINI_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MINI_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_MINI_CAMERA = features.MINIMAP_FEATURES.camera.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]
_SCREEN = [0]

class TestCamera():

    def __init__(self):
        self.mapDim = {'x': None, 'y': None}
        self.fieldOfViewDim = {'x': None, 'y': None}
        self.rangeCamera = {'x': {'min': None, 'max': None}, 'y': {'min': None, 'max': None}}
        self.rangeObs = {'x': {'min': float('inf'), 'max': float('-inf')}, 'y': {'min': float('inf'), 'max': float('-inf')}}
        self.cameraPos = {'x': 0, 'y': 0}
        self.cameraObsOffset = {'x': None, 'y': None}
        self.lastObs = {'x': None, 'y': None}
        self.prevOffset = {'x': None, 'y': None}

        self.completed = False

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass

    def step(self, obs):

        if not self.mapDim['x']:
            self.mapDim['x'] = len(obs.observation["minimap"][_MINI_VISIBILITY])
            self.mapDim['y'] = len(obs.observation["minimap"][_MINI_VISIBILITY][0])

        if not self.completed:
            minimap_y, minimap_x = obs.observation["minimap"][_MINI_CAMERA].nonzero()

            if not self.fieldOfViewDim['x']:
                for i in range(1, len(minimap_x)):
                    if (minimap_x[i] < minimap_x[i-1]):
                        self.fieldOfViewDim['x'] = i
                        break
                for i in range(1, len(minimap_y)):
                    if (minimap_y[i] > minimap_y[i-1]):
                        self.fieldOfViewDim['y'] = i
                        break

            offset = {'x':  self.cameraPos['x'] - minimap_x[0], 'y':  self.cameraPos['y'] - minimap_y[0]}
            if not self.cameraObsOffset['x'] and self.prevOffset['x'] == offset['x']:
                self.cameraObsOffset['x'] = offset['x']
            self.prevOffset['x'] = offset['x']
            if not self.cameraObsOffset['y'] and self.cameraPos['y'] > 0 and self.prevOffset['y'] == offset['y']:
                self.cameraObsOffset['y'] = offset['y']
            self.prevOffset['y'] = offset['y']

            self.lastObs['x'] = minimap_x[0]
            self.lastObs['y'] = minimap_y[0]

            self.rangeObs['x']['min'] = min(self.rangeObs['x']['min'], self.lastObs['x'])
            self.rangeObs['x']['max'] = max(self.rangeObs['x']['max'], self.lastObs['x'])
            self.rangeObs['y']['min'] = min(self.rangeObs['y']['min'], self.lastObs['y'])
            self.rangeObs['y']['max'] = max(self.rangeObs['y']['max'], self.lastObs['y'])

            if self.cameraPos['y'] > 0 and self.cameraPos['y'] + 1 < self.mapDim['y']:
                self.cameraPos['y'] += 1
            elif self.cameraPos['x'] + 1 < self.mapDim['x']:
                self.cameraPos['x'] += 1
            elif self.cameraPos['y'] == 0:
                self.cameraPos['y'] = 1
            else:
                self.completed = True
                self.displayResults()

        return actions.FunctionCall(_MOVE_CAMERA, [[self.cameraPos['x'], self.cameraPos['y']]])

    def displayResults(self):
        self.rangeCamera['x']['min'] = self.rangeObs['x']['min'] + self.cameraObsOffset['x']
        self.rangeCamera['x']['max'] = self.rangeObs['x']['max'] + self.cameraObsOffset['x']
        self.rangeCamera['y']['min'] = self.rangeObs['y']['min'] + self.cameraObsOffset['y']
        self.rangeCamera['y']['max'] = self.rangeObs['y']['max'] + self.cameraObsOffset['y']

        print("========")
        print("Map dim: " + str(self.mapDim))
        print("Field of view dim: " + str(self.fieldOfViewDim))
        print("Camera/top-left-obs offset: " + str(self.cameraObsOffset))
        print("Range camera: " + str(self.rangeCamera))
        print("========")