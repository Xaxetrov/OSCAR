from oscar.meta_action.common import *
from util.location import Location
from util.coordinatesHelper import CoordinatesHelper

def moveCamera(self, location, coordinatesHelper = None):
	if not coordinatesHelper:
		coordinatesHelper = CoordinatesHelper()
    return actions.FunctionCall(MOVE_CAMERA, [coordinatesHelper.getCameraMove(location).toArray()])


