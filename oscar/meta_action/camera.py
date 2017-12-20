from oscar.util.selection import *
from oscar.util.location import Location
from oscar.util.coordinates_helper import Coordinates_helper

def move_camera(location, coordinates_helper = None):
	if not coordinates_helper:
		coordinates_helper = Coordinates_helper()
	return [actions.FunctionCall(MOVE_CAMERA, [coordinates_helper.get_camera_move(location).to_array()])]