import numpy as np
from oscar.util.location import Location
from oscar.util.coordinatesHelper import CoordinatesHelper
from oscar.constants import *

# Returns a location in the map which is both close and not explored, if possible.
# Uses random.
def getNewTarget(obs, curLocation, coordinatesHelper = None, nbTry = 3):
    if not coordinatesHelper:
        coordinatesHelper = CoordinatesHelper()

    bestTarget, bestTargetScore = None, None

    for i in range(0, nbTry):
        targetLocation = coordinatesHelper.getRandomMinimapLocation()
        score = _scorePosition(obs, curLocation, targetLocation, coordinatesHelper)

        if not bestTargetScore or score > bestTargetScore:
            bestTargetScore = score
            bestTarget = targetLocation

    return bestTarget
    
# Scores a position, based on its distance and its visibility.
# Could be negative.
def _scorePosition(obs, curLocation, targetLocation, coordinatesHelper):
    _UNEXPLORED_SCORE = 2
    _EXPLORED_SCORE = 1
    _VISIBLE_SCORE = 0
    _DISTANCE_WEIGHT = 4.0

    distance = curLocation.distance(targetLocation)

    visibility = obs.observation['minimap'][MINI_VISIBILITY]
    limits = coordinatesHelper.getMinimapViewLimits(targetLocation)

    visibilityScore = 0
    for col in range(limits['x']['min'], limits['x']['max']+1):
        for row in range(limits['y']['min'], limits['y']['max']+1):
            if (visibility[row][col] == UNEXPLORED_CELL):
                visibilityScore += _UNEXPLORED_SCORE
            elif (visibility[row][col] == EXPLORED_CELL):
                visibilityScore += _EXPLORED_SCORE
            elif (visibility[row][col] == VISIBLE_CELL):
                visibilityScore += _VISIBLE_SCORE
            else:
                raise Exception('Unknown value in visibility map: ' + locationVisibility[row][col])

    return (visibilityScore - _DISTANCE_WEIGHT * distance)