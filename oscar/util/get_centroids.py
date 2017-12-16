import numpy as np

# start points for matching
_TOP_LEFT = 0
_TOP_RIGHT = 1
_BOTTOM_LEFT = 2
_BOTTOM_RIGHT = 3


"""
Updates the "player_relative" matrix by pushing units with "unit_id"
to the foreground, when there is some overlapping.

This operation could generate some errors in some ambiguous cases:
some pixels which don't contain the unit could be set to unit_id anyway.

Keyword arguments:
player_relative, unit_density -- observation layers
unit_id -- id of the units to move to the foreground
"""
def _move_to_foreground(player_relative, unit_density, unit_id):
	# Stores whether a pixel has changed in player_relative
	processed = np.zeros((len(player_relative), len(player_relative[0])))

	for i in range(len(player_relative)):
		for j in range(len(player_relative[0])):
			if processed[i][j] == 0 and player_relative[i][j] == unit_id:

				# Explores contiguous pixels with DFS
				# and moves them to foreground when necessary 
				stack = []
				stack.append((i, j))

				def to_foreground(i, j, player_relative, unit_density, unit_id):
					if unit_density[i, j] > 1:
						player_relative[i, j] = unit_id

				def process_pixel(i, j, player_relative, unit_density, processed, unit_id, stack):
					if processed[i, j] != 0:
						return

					processed[i, j] = 1
					to_foreground(i, j, player_relative, unit_density, unit_id)
					if player_relative[i, j] == unit_id:
						stack.append((i, j))

				while len(stack) > 0:
					cur = stack.pop()

					process_pixel(cur[0] - 1, cur[1], player_relative, unit_density, processed, unit_id, stack)
					process_pixel(cur[0] + 1, cur[1], player_relative, unit_density, processed, unit_id, stack)
					process_pixel(cur[0], cur[1] - 1, player_relative, unit_density, processed, unit_id, stack)
					process_pixel(cur[0], cur[1] + 1, player_relative, unit_density, processed, unit_id, stack)


# Builds a disk mask using midpoint circle algorithm
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


def _get_match_start_end_step(direction, mask_size):
	# Gets step depending on direction
	step_x, step_y = None, None
	if direction == _TOP_LEFT:
		step_x, step_y = -1, -1
	elif direction == _TOP_RIGHT:
		step_x, step_y = 1, -1
	elif direction == _BOTTOM_LEFT:
		step_x, step_y = -1, 1
	else: # _BOTTOM_RIGHT
		step_x, step_y = 1, 1

	# Gets start and end boudaries using steps
	start_x, start_y, end_x, end_y = None, None, None, None
	if step_x == 1:
		start_x = 0
		end_x = mask_size[0]
	else: # step_x == -1
		start_x = mask_size[0] - 1
		end_x = -1
	if step_y == 1:
		start_y = 0
		end_y = mask_size[1]
	else: # step_y == -1
		start_y = mask_size[1] - 1
		end_y = -1

	return start_x, start_y, end_x, end_y, step_x, step_y


"""
Returns whether their is a match between the mask and the player_relative layer.
Returns true even if the match is partially outside of the layer.

Keyword arguments:
unit_id -- the unit id to be matched in player_relative
direction -- direction used for matching (_TOP_LEFT, _TOP_RIGHT, _BOTTOM_LEFT or _BOTTOM_RIGHT)
start_point -- start point of the matching
"""
def _is_match(player_relative, mask, unit_id, direction, start_point):
	start_x, start_y, end_x, end_y, step_x, step_y = \
		_get_match_start_end_step(direction, (len(mask), len(mask[0])))

	for i in range(start_x, end_x, step_x):
		for j in range(start_y, end_y, step_y):
			abs_x = start_point[0] + i*step_x
			abs_y = start_point[1] + j*step_y

			if abs_x < 0 or abs_x >= len(player_relative) \
				or abs_y < 0 or abs_y >= len(player_relative[0]):
				continue

			if mask[i, j] != 0 and \
				player_relative[abs_x, abs_y] != unit_id:
				return False

	return True


"""
Updates unit_density by decrementing cells matching with the mask.

Keyword arguments:
direction -- direction used for matching (_TOP_LEFT, _TOP_RIGHT, _BOTTOM_LEFT or _BOTTOM_RIGHT)
start_point -- start point of the matching
"""
def _update_unit_density(unit_density, mask, direction, start_point):
	start_x, start_y, end_x, end_y, step_x, step_y = \
		_get_match_start_end_step(direction, (len(mask), len(mask[0])))

	for i in range(start_x, end_x, step_x):
		for j in range(start_y, end_y, step_y):
			abs_x = start_point[0] + i*step_x
			abs_y = start_point[1] + j*step_y

			if abs_x < 0 or abs_x >= len(unit_density) \
				or abs_y < 0 or abs_y >= len(unit_density[0]):
				continue

			if mask[i, j] != 0:
				unit_density[abs_x, abs_y] -= 1


"""
Computes and returns disks centers positions by matching them with a mask.
"""
def get_centroid_centers(player_relative, unit_density, disks_radius, unit_id):
	return