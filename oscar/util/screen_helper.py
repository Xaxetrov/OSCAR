import numpy as np
from oscar.constants import *
from math import floor, ceil

""" 
Builds a disk mask using midpoint circle algorithm
Radius should be an integer
"""
def _get_disk_mask_integer_radius(radius):

	def draw_hor_line(mat, x0, x1, y):
		for x in range(x0, x1+1):
			if x >= 0 and x < len(mat) \
				and y >= 0 and y < len(mat[0]):
				mat[x][y] = 1

	def draw(mat, x0, y0, x, y):
		draw_hor_line(mat, x0 - x, x0 + x, y0 + y)
		draw_hor_line(mat, x0 - x, x0 + x, y0 - y)
		draw_hor_line(mat, x0 - y, x0 + y, y0 + x)
		draw_hor_line(mat, x0 - y, x0 + y, y0 - x)

	x0, y0 = radius, radius
	x, y = 0, radius
	d = 5/4.0 - radius
	mask = np.zeros((2*radius+1, 2*radius+1))	

	draw(mask, x0, y0, x, y)
	while x < y:
		if d < 0:
			x += 1
			d += 2*x + 1
		else:
			x += 1
			y -= 1
			d += 2*(x-y) + 1
		draw(mask, x0, y0, x, y)

	return mask

""" 
Builds a disk mask using midpoint circle algorithm
Radius is a float
"""
def _get_disk_mask_float_radius(radius):
	if radius == floor(radius):
		return _get_disk_mask_integer_radius(floor(radius))

	def get_resized_mask(mask, resize_factor):
		size = ceil(len(mask) * resize_factor)
		new_mask = np.zeros((size, size))

		padding = (ceil(len(mask)*resize_factor)/resize_factor - len(mask)) / 2
		
		for x in range(size):
			for y in range(size):
				num, sum = 0, 0

				for i in range(round(x/resize_factor), round(x/resize_factor+1/resize_factor)):
					if i < 0 or i >= len(mask):
						continue

					for j in range(round(y/resize_factor), round(y/resize_factor+1/resize_factor)):
						if j < 0 or j >= len(mask[0]):
							continue
						if mask[i, j] > 0:
							sum += 1
						num += 1

				if num > 0 and sum / num >= 0.5:
					new_mask[x, y] = 1

		return new_mask

	EXPAND_FACTOR = 4
	expanded = _get_disk_mask_integer_radius(round(radius * EXPAND_FACTOR))
	return get_resized_mask(expanded, 1. / EXPAND_FACTOR)

"""
Updates the "screen_unit_type" matrix by pushing units with "unit_id"
to the foreground, when there is some overlapping.

This operation could generate some errors in some ambiguous cases:
some pixels which don't contain the unit could be set to unit_id anyway.
"""
def _move_to_foreground(screen_layer, unit_density, target_id):
	# Stores whether a pixel has changed in screen_layer
	processed = np.zeros((len(screen_layer), len(screen_layer[0])))

	for x in range(len(screen_layer)):
		for y in range(len(screen_layer[0])):
			if processed[x][y] == 0 and screen_layer[x][y] == target_id:

				# Explores contiguous pixels with DFS
				# and moves them to foreground when necessary 
				stack = []
				stack.append((x, y))

				def to_foreground(x, y, screen_layer, unit_density, target_id):
					if unit_density[x, y] > 1:
						screen_layer[x, y] = target_id

				def process_pixel(x, y, screen_layer, unit_density, processed, target_id, stack):
					if x < 0 or x >= len(screen_layer) or y < 0 or y >= len(screen_layer[0]) \
						or processed[x, y] != 0:
						return

					processed[x, y] = 1
					to_foreground(x, y, screen_layer, unit_density, target_id)
					if screen_layer[x, y] == target_id:
						stack.append((x, y))

				while len(stack) > 0:
					cur = stack.pop()

					process_pixel(cur[0] - 1, cur[1], screen_layer, unit_density, processed, target_id, stack)
					process_pixel(cur[0] + 1, cur[1], screen_layer, unit_density, processed, target_id, stack)
					process_pixel(cur[0], cur[1] - 1, screen_layer, unit_density, processed, target_id, stack)
					process_pixel(cur[0], cur[1] + 1, screen_layer, unit_density, processed, target_id, stack)


"""
Returns whether there is a match of the mask on the screen_unit_type layer,
even if the disk is partially outside of the layer.
"""
def _is_match(screen_unit_type, unit_id, player_relative, player_id, unit_density, mask, radius, start_point):
	pixels_matched = 0

	for x in range(len(mask)):
		for y in range(len(mask[0])):
			abs_x = start_point[0] + x - 2*radius
			abs_y = start_point[1] + y - 2*radius

			if abs_x < 0 or abs_x >= len(screen_unit_type) \
				or abs_y < 0 or abs_y >= len(screen_unit_type[0]):
				continue

			if mask[x, y] != 0:
				if unit_density[abs_x, abs_y] <= 0 \
					or screen_unit_type[abs_x, abs_y] != unit_id \
					or player_relative[abs_x, abs_y] != player_id:
					return False
				else:
					pixels_matched += 1

	return (pixels_matched > 0)


""" Updates unit_density by decrementing cells matching with the mask. """
def _update_unit_density(unit_density, mask, radius, start_point):
	for x in range(len(mask)):
		for y in range(len(mask[0])):
			abs_x = start_point[0] + x - 2*radius
			abs_y = start_point[1] + y - 2*radius

			if abs_x < 0 or abs_x >= len(unit_density) \
				or abs_y < 0 or abs_y >= len(unit_density[0]):
				continue

			if mask[x, y] != 0:
				unit_density[abs_x, abs_y] -= 1

"""
Returns a list of unique ids corresponding
to the units visible on the screen, for a given player_id.

Keyword arguments:
player_id -- player id to match
"""
def get_units_id(obs, player_id):
	units_dic = {}
	units_list = []

	screen_unit_type = obs.observation["screen"][SCREEN_UNIT_TYPE]
	player_relative = obs.observation["screen"][SCREEN_PLAYER_RELATIVE]

	for x in range(len(screen_unit_type)):
		for y in range(len(screen_unit_type[0])):
			id = screen_unit_type[x, y]

			if id != 0 and player_relative[x, y] == player_id and not units_dic[id]:
				units_dic[id] = 1
				units_list.append(id)

	return units_list

"""
Computes and returns disks centers positions, in linear time of the size of the screen.
Handles overlapping and disks partially outside of the screen_unit_type.
Could match in an unexpected way in some rare ambiguous cases.

Requirement: disks_radius should be smaller than half of the width and the height of the screen.

Keyword arguments:
screen_unit_type -- screen_unit_type observation layer
unit_id -- unit id to match
player_relative -- player_relative observation layer
player_id -- player id to match
unit_density -- unit_density observation layer
disks_radius -- radius of disks to match
"""
def match_all(screen_unit_type, unit_id, player_relative, player_id, unit_density, disks_radius):
	assert disks_radius < len(screen_unit_type)/2 and disks_radius < len(screen_unit_type[0])/2, \
		"Error: disks_radius should be smaller than half of the width and the height of the screen_unit_type"

	"""
	Makes a copy of input layers
	to avoid modifying them outside of the function.
	"""
	_screen_unit_type = screen_unit_type.copy()
	_player_relative = player_relative.copy()
	_unit_density = unit_density.copy()

	# disk mask
	mask = _get_disk_mask_float_radius(disks_radius)
	
	# handles overlapping
	_move_to_foreground(_screen_unit_type, _unit_density, unit_id)
	_move_to_foreground(_player_relative, _unit_density, player_id)

	"""
	Scans screen_unit_type by moving the mask,
	following a rectangular path centered on screen center.
	"""
	scan_width, scan_height = len(_screen_unit_type) + 2*disks_radius, len(_screen_unit_type[0]) + 2*disks_radius
	rect_left, rect_right, rect_top, rect_bottom = None, None, None, None

	_LEFT = 0
	_RIGHT = 1
	_TOP = 2
	_BOTTOM = 3

	centroids = []

	# Keeps going until entire screen_unit_type is scanned.
	while rect_left is None or \
		rect_left > 0 or rect_right < scan_width-1 or rect_top > 0 or rect_bottom < scan_height-1:	
		
		# default direction
		direction = _LEFT

		if rect_left is None: # first iteration
			rect_left, rect_right, rect_top, rect_bottom = \
				floor(scan_width/2)+1, floor(scan_width/2), floor(scan_height/2), floor(scan_height/2)
		else:
			# Computes direction where there is the largest number of remaining pixels.
			min_dist = rect_left
			if scan_width-1 - rect_right > min_dist:
				direction = _RIGHT
				min_dist = scan_width-1 - rect_right
			if rect_top > min_dist:
				direction = _TOP
				min_dist = rect_top
			if scan_height-1 - rect_bottom > min_dist:
				direction = _BOTTOM
				min_dist = scan_height-1 - rect_bottom

		# Computes pixels line to scan
		start_x, start_y, end_x, end_y = None, None, None, None
		if direction == _LEFT:
			rect_left -= 1
			start_x, end_x = rect_left, rect_left
			start_y, end_y = rect_top, rect_bottom	
		elif direction == _RIGHT:
			rect_right += 1
			start_x, end_x = rect_right, rect_right
			start_y, end_y = rect_top, rect_bottom
		elif direction == _TOP:
			rect_top -= 1
			start_y, end_y = rect_top, rect_top
			start_x, end_x = rect_left, rect_right
		elif direction == _BOTTOM:
			rect_bottom += 1
			start_y, end_y = rect_bottom, rect_bottom
			start_x, end_x = rect_left, rect_right

		# Scan line
		for x in range(start_x, end_x+1):
			for y in range(start_y, end_y+1):

				"""
				If the mask matches,
				adds center position to centroids list
				and removes matched disk from unit_density.
				"""
				if _is_match(_screen_unit_type, unit_id, _player_relative, player_id, _unit_density, mask, disks_radius, (x, y)):
					centroids.append((y-disks_radius, x-disks_radius))
					_update_unit_density(_unit_density, mask, disks_radius, (x, y))

	return centroids

"""
Computes and returns units centers positions.
Handles units overlapping and partially outside of the screen.
Could match in an unexpected way in some rare ambiguous cases.

Keyword arguments:
unit_id -- unit id to match
player_id -- player id to match
"""
def get_center(obs, unit_id, player_id):
	unit_tile_size = None
	if unit_id == TERRAN_COMMAND_CENTER:
		unit_tile_size = TERRAN_COMMAND_CENTER_TILE_SIZE
	elif unit_id == TERRAN_SCV:
		unit_tile_size = TERRAN_SCV_TILE_SIZE
	elif unit_id == MINERAL_FIELD:
		unit_tile_size = MINERAL_FIELD_TILE_SIZE
	else:
		raise ValueError("Unit id ({0}) not handled.".format(unit_id))
		return

	return match_all(obs.observation["screen"][SCREEN_UNIT_TYPE], \
		unit_id, \
		obs.observation["screen"][SCREEN_PLAYER_RELATIVE], \
		player_id, \
		obs.observation["screen"][SCREEN_UNIT_DENSITY], \
		floor(unit_tile_size * TILES_SIZE_IN_CELL / 2.))