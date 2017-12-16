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
	return

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


"""
Returns whether their is a match between the mask and the player_relative layer.
Returns true even if the match is partially outside of the layer.

Keyword arguments:
unit_id -- the unit id to be matched in player_relative
direction -- direction used for matching (_TOP_LEFT, _TOP_RIGHT, _BOTTOM_LEFT or _BOTTOM_RIGHT)
start_point -- start point of the matching
"""
def _is_match(player_relative, mask, unit_id, direction, start_point):
	return false


"""
Updates unit_density by decrementing cells matching with the mask.

Keyword arguments:
unit_id -- the unit id to be matched in player_relative
direction -- direction used for matching (_TOP_LEFT, _TOP_RIGHT, _BOTTOM_LEFT or _BOTTOM_RIGHT)
start_point -- start point of the matching
"""
def _update_unit_density(unit_density, mask, unit_id, direction, start_point):
	return


"""
Computes and returns disks centers positions by matching them with a mask.
"""
def get_centroid_centers(player_relative, unit_density, disks_radius, unit_id):
	return