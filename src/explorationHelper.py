from coordinatesHelper import CoordinatesHelper
import pickle
import numpy as np
from pysc2.lib import features
from location import Location

_UNEXPLORED_CELL = 0 # cell never explored
_EXPLORED_CELL = 1 # cell explored, but not visible anymore
_VISIBLE_CELL = 2 # cell explored and still visible

_MINI_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index

#############################
# Helper for map exploration
#############################
class ExplorationHelper():

    def __init__(self, _coordinatesHelper = None):
    	np.set_printoptions(threshold=np.nan)
    	self.coordinatesHelper = None
    	if _coordinatesHelper:
    		self.coordinatesHelper = _coordinatesHelper
    	else:
    		self.coordinatesHelper = CoordinatesHelper()

    def getNewTarget(self, obs, curLocation, nbTry = 3):
    	bestTarget, bestTargetScore = None, None

    	for i in range(0, nbTry):
    		targetLocation = self.coordinatesHelper.getRandomMinimapLocation()
    		score = self.scorePosition(obs, curLocation, targetLocation)

    		if not bestTargetScore or score > bestTargetScore:
    			bestTargetScore = score
    			bestTarget = targetLocation

    	return bestTarget
    	
    # Scores a position, based on its distance and its visibility.
    # Could be negative.
    def scorePosition(self, obs, curLocation, targetLocation):
        _UNEXPLORED_SCORE = 2
        _EXPLORED_SCORE = 1
        _VISIBLE_SCORE = 0
        _DISTANCE_WEIGHT = 4.0

        distance = curLocation.distance(targetLocation)

        visibility = obs.observation['minimap'][_MINI_VISIBILITY]
        limits = self.coordinatesHelper.getMinimapViewLimits(targetLocation)

        visibilityScore = 0
        for col in range(limits['x']['min'], limits['x']['max']+1):
        	for row in range(limits['y']['min'], limits['y']['max']+1):
        		if (visibility[row][col] == _UNEXPLORED_CELL):
        			visibilityScore += _UNEXPLORED_SCORE
        		elif (visibility[row][col] == _EXPLORED_CELL):
        			visibilityScore += _EXPLORED_SCORE
        		elif (visibility[row][col] == _VISIBLE_CELL):
        			visibilityScore += _VISIBLE_SCORE
        		else:
        			raise Exception('Unknown value in visibility map: ' + locationVisibility[row][col])

        return (visibilityScore - _DISTANCE_WEIGHT * distance)

if __name__ == "__main__":
    explorationHelper = ExplorationHelper()
    obs = pickle.load( open( "obs.p", "rb" ) )
    target = explorationHelper.getNewTarget(obs, Location(13, 38), 30)