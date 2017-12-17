import numpy as np
from math import floor

""" Builds a disk mask using midpoint circle algorithm """
def _get_disk_mask(radius):

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
Updates the "screen" matrix by pushing units with "unit_id"
to the foreground, when there is some overlapping.

This operation could generate some errors in some ambiguous cases:
some pixels which don't contain the unit could be set to unit_id anyway.
"""
def _move_to_foreground(screen, unit_density, unit_id):
	# Stores whether a pixel has changed in screen
	processed = np.zeros((len(screen), len(screen[0])))

	for x in range(len(screen)):
		for y in range(len(screen[0])):
			if processed[x][y] == 0 and screen[x][y] == unit_id:

				# Explores contiguous pixels with DFS
				# and moves them to foreground when necessary 
				stack = []
				stack.append((x, y))

				def to_foreground(x, y, screen, unit_density, unit_id):
					if unit_density[x, y] > 1:
						screen[x, y] = unit_id

				def process_pixel(x, y, screen, unit_density, processed, unit_id, stack):
					if x < 0 or x >= len(screen) or y < 0 or y >= len(screen[0]) \
						or processed[x, y] != 0:
						return

					processed[x, y] = 1
					to_foreground(x, y, screen, unit_density, unit_id)
					if screen[x, y] == unit_id:
						stack.append((x, y))

				while len(stack) > 0:
					cur = stack.pop()

					process_pixel(cur[0] - 1, cur[1], screen, unit_density, processed, unit_id, stack)
					process_pixel(cur[0] + 1, cur[1], screen, unit_density, processed, unit_id, stack)
					process_pixel(cur[0], cur[1] - 1, screen, unit_density, processed, unit_id, stack)
					process_pixel(cur[0], cur[1] + 1, screen, unit_density, processed, unit_id, stack)


"""
Returns whether there is a match of the mask on the screen layer,
even if the disk is partially outside of the layer.
"""
def _is_match(screen, unit_density, mask, radius, unit_id, start_point):
	pixels_matched = 0

	for x in range(len(mask)):
		for y in range(len(mask[0])):
			abs_x = start_point[0] + x - 2*radius
			abs_y = start_point[1] + y - 2*radius

			if abs_x < 0 or abs_x >= len(screen) \
				or abs_y < 0 or abs_y >= len(screen[0]):
				continue

			if mask[x, y] != 0 and \
				(unit_density[abs_x, abs_y] <= 0 or screen[abs_x, abs_y] != unit_id):
				return False
			elif mask[x, y] != 0:
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
Computes and returns disks centers positions, in linear time of the size of the screen.
Handles overlapping and disks partially outside of the screen.
Could match in an unexpected way in some rare ambiguous cases.

Requirement: disks_radius should be smaller than half of the width and the height of the screen.

Keyword arguments:
screen -- screen observation layer
unit_density -- unit_density observation layer
disks_radius -- radius of disks to match
unit_id -- the unit id to be matched in screen
"""
def get_centroid_centers(screen, unit_density, disks_radius, unit_id):
	assert disks_radius < len(screen)/2 and disks_radius < len(screen[0])/2, \
		"Error: disks_radius should be smaller than half of the width and the height of the screen"

	"""
	Makes a copy of input layers
	to avoid modifying them outside of the function.
	"""
	_screen = screen.copy()
	_unit_density = unit_density.copy()

	# disk mask
	mask = _get_disk_mask(disks_radius)
	
	# handles overlapping
	_move_to_foreground(_screen, _unit_density, unit_id)

	"""
	Scans screen by moving the mask,
	following a rectangular path centered on screen center.
	"""
	scan_width, scan_height = len(_screen) + 2*disks_radius, len(_screen[0]) + 2*disks_radius
	rect_left, rect_right, rect_top, rect_bottom = None, None, None, None

	_LEFT = 0
	_RIGHT = 1
	_TOP = 2
	_BOTTOM = 3

	centroids = []

	# Keeps going until entire screen is scanned.
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
				if _is_match(_screen, _unit_density, mask, disks_radius, unit_id, (x, y)):
					centroids.append((x-disks_radius, y-disks_radius))
					_update_unit_density(_unit_density, mask, disks_radius, (x, y))

	return centroids