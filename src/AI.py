import numpy as np
import sys
import time

from pysc2.lib import actions
from pysc2.lib import features

from location import Location
from coordinatesHelper import CoordinatesHelper
from explorationHelper import ExplorationHelper

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
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

# States of the finite states automaton
class States:
    START = 0
    EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET = 1
    EXPLORATION__MOVE_UNITS_TO_TARGET = 2
    EXPLORATION__CENTER_CAMERA_ON_UNITS = 3
    ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY = 4
    ATTACK__CENTER_CAMERA_ON_UNITS = 5

class FindAndDefeatZerglings():

    _NEW_TARGET_NB_TRY = 5

    def __init__(self):
        self.coordinatesHelper = CoordinatesHelper()
        self.explorationHelper = ExplorationHelper(self.coordinatesHelper)
        self.state = States.START
        self.target = None

        self._locInMinimap = None
        self._ennemies_x = None
        self._ennemies_y = None
        self._units_x = None
        self._units_y = None
        self._unitsMeanLoc = None
        self._isUnitsMeanLocUpdated = False
        self._isEnnemiesVisible = None
        self._targetReached = None

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass

    def step(self, obs):
        
        self._ennemies_x = None
        self._ennemies_y = None
        self._units_x = None
        self._units_y = None
        self._isUnitsMeanLocUpdated = False
        self._isEnnemiesVisible = None
        self._targetReached = None

        # if some entities are selected
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            self.state = self.getNextState(obs, self.state)
            print("state: " + str(self.state))
            time.sleep(0.2)
            return self.getAction(obs, self.state)
            
        elif _SELECT_ARMY in obs.observation["available_actions"]:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

        else:
            return actions.FunctionCall(_NO_OP, [])
     
    def getNextState(self, obs, curState):
        if curState == States.START:
            if self.isEnnemyVisible(obs):
                return States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY
            else:
                return States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET

        elif curState == States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET:
            return States.EXPLORATION__MOVE_UNITS_TO_TARGET

        elif curState == States.EXPLORATION__MOVE_UNITS_TO_TARGET:
            return States.EXPLORATION__CENTER_CAMERA_ON_UNITS

        elif curState == States.EXPLORATION__CENTER_CAMERA_ON_UNITS:
            if self.isEnnemyVisible(obs):
                return States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY
            else:
                if self.isTargetReached(obs) or not self.isUnitsMoving(obs):
                    return States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET
                else:
                    return States.EXPLORATION__CENTER_CAMERA_ON_UNITS

        elif curState == States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY:
            return States.ATTACK__CENTER_CAMERA_ON_UNITS

        elif curState == States.ATTACK__CENTER_CAMERA_ON_UNITS:
            if self.isEnnemyVisible(obs):
                return States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY
            else:
                return States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET

    def getAction(self, obs, state):
        if state == States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET:
            self.target = self.explorationHelper.getNewTarget(obs, self.getLocInMinimap(obs), self._NEW_TARGET_NB_TRY)
            return self.coordinatesHelper.getMoveCameraAction(self.target)

        elif state == States.EXPLORATION__MOVE_UNITS_TO_TARGET:
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, self.coordinatesHelper.getScreenCenter().toArray()])

        elif state == States.EXPLORATION__CENTER_CAMERA_ON_UNITS:
            self._locInMinimap = self.getMinimapLocCenteredOnUnits(obs)
            return self.coordinatesHelper.getMoveCameraAction(self._locInMinimap)

        elif state == States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY:
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, self.getClosestEnnemy(obs).toArray()])

        elif state == States.ATTACK__CENTER_CAMERA_ON_UNITS:
            self._locInMinimap = self.getMinimapLocCenteredOnUnits(obs)
            return self.coordinatesHelper.getMoveCameraAction(self._locInMinimap)

    def getLocInMinimap(self, obs):
        if not self._locInMinimap:
            self._locInMinimap = self.coordinatesHelper.getLocInMinimap(obs)
        return self._locInMinimap

    def getEnnemiesLocations(self, obs):
        if self._ennemies_x is None or self._ennemies_y is None:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            self._ennemies_y, self._ennemies_x = (player_relative == _PLAYER_HOSTILE).nonzero()
        return self._ennemies_x, self._ennemies_y

    def getUnitsLocations(self, obs):
        if self._units_x is None or not self._units_y is None:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            self._units_y, self._units_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        return self._units_x, self._units_y

    def getUnitsMeanLocation(self, obs):
        if not self._isUnitsMeanLocUpdated:
            units_x, units_y = self.getUnitsLocations(obs) 
            if units_x.size > 0:
                newLoc = Location(int(units_x.mean()), int(units_y.mean()))
                if not self._unitsMeanLoc or not newLoc.equals(self._unitsMeanLoc):
                    self._unitsMeanLoc = newLoc
                    self._isUnitsMeanLocUpdated = True
        return self._unitsMeanLoc;

    def isUnitsMoving(self, obs):
        self.getUnitsMeanLocation(obs)
        return self._isUnitsMeanLocUpdated

    def isEnnemyVisible(self, obs):
        if self._isEnnemiesVisible is None:
            ennemies_x, ennemies_y = self.getEnnemiesLocations(obs)
            self._isEnnemiesVisible = ennemies_x.any()
        return self._isEnnemiesVisible

    def isTargetReached(self, obs):
        _TARGET_REACHED_SQUARED_DISTANCE = 25
        unitsMeanLoc = self.getUnitsMeanLocation(obs)
        return (not self.target or \
                    unitsMeanLoc.squarredDistance(self.target) < _TARGET_REACHED_SQUARED_DISTANCE)

    def getClosestEnnemy(self, obs):
        closest, min_dist = None, None
        ennemies_x, ennemies_y = self.getEnnemiesLocations(obs)
        unitsMeanLoc = self.getUnitsMeanLocation(obs)

        for p in zip(ennemies_x, ennemies_y):
            dist = np.linalg.norm(np.array(unitsMeanLoc.toArray()) - np.array(p))
            if not min_dist or dist < min_dist:
                closest, min_dist = p, dist
        return Location(closest[0], closest[1])

    def getMinimapLocCenteredOnUnits(self, obs):
        unitsMeanLoc = self.getUnitsMeanLocation(obs)
        move = unitsMeanLoc.difference(self.coordinatesHelper.getScreenCenter())
        globalLoc = self.coordinatesHelper.minimapToGlobal(self.getLocInMinimap(obs))
        centeredGlobalLoc = globalLoc.addition(move)
        centeredMinimapLoc = self.coordinatesHelper.globalToMinimap(centeredGlobalLoc)
        return self.coordinatesHelper.bound(centeredMinimapLoc)