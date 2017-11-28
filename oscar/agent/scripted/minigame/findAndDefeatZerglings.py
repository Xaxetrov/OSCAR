import numpy as np
import sys
import time

from oscar.constants import *
from oscar.util.location import Location
from oscar.util.coordinatesHelper import CoordinatesHelper
from oscar.util.explorationHelper import *

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
        if MOVE_SCREEN in obs.observation["available_actions"]:
            self.state = self.getNextState(obs, self.state)
            print("state: " + str(self.state))
            time.sleep(0.2)
            return self.getAction(obs, self.state)
            
        elif _SELECT_ARMY in obs.observation["available_actions"]:
            return actions.FunctionCall(SELECT_ARMY, [SELECT_ALL])

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
            self.target = getNewTarget(obs, self.getLocInMinimap(obs), self.coordinatesHelper, self._NEW_TARGET_NB_TRY)
            return actions.FunctionCall(MOVE_CAMERA, [self.coordinatesHelper.getCameraMove(self.target).toArray()])

        elif state == States.EXPLORATION__MOVE_UNITS_TO_TARGET:
            return actions.FunctionCall(ATTACK_SCREEN, [NOT_QUEUED, self.coordinatesHelper.getScreenCenter().toArray()])

        elif state == States.EXPLORATION__CENTER_CAMERA_ON_UNITS:
            self._locInMinimap = self.getMinimapLocCenteredOnUnits(obs)
            return actions.FunctionCall(MOVE_CAMERA, [self.coordinatesHelper.getCameraMove(self._locInMinimap).toArray()])

        elif state == States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY:
            return actions.FunctionCall(ATTACK_SCREEN, [NOT_QUEUED, self.getClosestEnnemy(obs).toArray()])

        elif state == States.ATTACK__CENTER_CAMERA_ON_UNITS:
            self._locInMinimap = self.getMinimapLocCenteredOnUnits(obs)
            return actions.FunctionCall(MOVE_CAMERA, [self.coordinatesHelper.getCameraMove(self._locInMinimap).toArray()])

    def getLocInMinimap(self, obs):
        if not self._locInMinimap:
            self._locInMinimap = self.coordinatesHelper.getLocInMinimap(obs)
        return self._locInMinimap

    def getEnnemiesLocations(self, obs):
        if self._ennemies_x is None or self._ennemies_y is None:
            player_relative = obs.observation["screen"][PLAYER_RELATIVE]
            self._ennemies_y, self._ennemies_x = (player_relative == PLAYER_HOSTILE).nonzero()
        return self._ennemies_x, self._ennemies_y

    def getUnitsLocations(self, obs):
        if self._units_x is None or not self._units_y is None:
            player_relative = obs.observation["screen"][PLAYER_RELATIVE]
            self._units_y, self._units_x = (player_relative == PLAYER_FRIENDLY).nonzero()
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