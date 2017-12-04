import numpy as np

# State of the automaton
class _State():
	def __init__(self, _relative_pixel):
		self.relative_pixel = _relative_pixel
		self.true_nextstate = None
		self.false_nextstate = None
		self.relative_disk_center = None

def _run_automata(screen, matched_pixels, i, j, state, val):
	# Absolute coordinates
	x = i + state.relative_pixel[0]
	y = j + state.relative_pixel[1]

	# Final state in the automaton
	if state.relative_disk_center is not None:
		nb_matched = 0

		if x >= 0 and x < len(screen) \
			and y >= 0 and y < len(screen[0]):
			matched_pixels[x][y] = 1
			nb_matched += 1

		# If disk included in disk, tries to match largest disk
		if state.true_nextstate is not None:
			ret, n, matched_pixels = _run_automata(screen, matched_pixels, i, j, state.true_nextstate, val)
			nb_matched += n
			if ret is not None:
				return ret, nb_matched, matched_pixels

		return {'x': i + state.relative_disk_center['x'], \
				'y': j + state.relative_disk_center['y']}, nb_matched, matched_pixels

	if x < 0 or x >= len(screen) \
		or y < 0 or y >= len(screen[0]):

		ret_true, ret_false = None, None
		nb_matched_true, nb_matched_false = None, None
		matched_pixels_true, matched_pixels_false = None, None

		if state.true_nextstate is not None:
			matched_pixels_true = matched_pixels.copy()
			ret_true, nb_matched_true, matched_pixels_true = _run_automata(screen, matched_pixels_true, i, j, state.true_nextstate, val)
		if state.false_nextstate is not None:
			matched_pixels_false = matched_pixels.copy()
			ret_false, nb_matched_false, matched_pixels_false = _run_automata(screen, matched_pixels_false, i, j, state.false_nextstate, val)
		
		if ret_true and (not ret_false or nb_matched_true >= nb_matched_false):
			matched_pixels = matched_pixels_true
			return ret_true, nb_matched_true, matched_pixels
		elif ret_false:
			matched_pixels = matched_pixels_false
			return ret_false, nb_matched_false, matched_pixels
		else:
			return None, 0, matched_pixels

	elif screen[x][y] == val:
		if state.true_nextstate is None:
			return None, 0, matched_pixels
		else:
			matched_pixels[x][y] = 1
			ret, nb_matched, matched_pixels = _run_automata(screen, matched_pixels, i, j, state.true_nextstate, val)
			if ret is None: # backtracking
				matched_pixels[x][y] = 0
				nb_matched -= 1
			return ret, (nb_matched+1), matched_pixels
	else:
		if state.false_nextstate is None:
			return None, 0, matched_pixels
		else:
			return _run_automata(screen, matched_pixels, i, j, state.false_nextstate, val)

# Given a screen, uses the automaton to match disks and returns their centroids.
def _match(screen, automaton, val):
	centroids = []
	matched_pixels = np.zeros((len(screen), len(screen[0])))

	for i in range(len(screen)):
		for j in range(len(screen[0])):

			if screen[i][j] == val and matched_pixels[i][j] == 0:

				c, nb, matched_pixels = _run_automata(screen, matched_pixels, i, j, automaton, val)
				if c:
					centroids.append(c)
				else:
					raise Exception("Error: can't match disk")
	return centroids

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

# Build automaton to match the mask
def _build_automaton(mask):
	# Extracts mask coordinates
	mask_pixels = []
	for i in range(len(mask)):
		for j in range(len(mask[0])):
			if mask[i][j] == 1:
				mask_pixels.append({'x': i, 'y': j})

	# Creates a new branch of states
	def _addBranch(start_state, relative_pixels, relative_disk_center):
		cur_state = start_state
		while len(relative_pixels) > 0:
			cur_state.true_nextstate = _State(relative_pixels.popitem()[0])
			cur_state = cur_state.true_nextstate
		cur_state.relative_disk_center = relative_disk_center

	start_state = None
	for startPixel in mask_pixels:
		relative_disk_center = {'x': ((len(mask)-1) / 2 - startPixel['x']), \
						'y': ((len(mask[0])-1) / 2 - startPixel['y'])}

		relative_pixels = {}
		for curPixel in mask_pixels:
			dx = curPixel['x'] - startPixel['x']
			dy = curPixel['y'] - startPixel['y']
			if dx != 0 or dy != 0:
				relative_pixels[(dx, dy)] = (dx, dy)
		
		if not start_state:
			start_state = _State(relative_pixels.popitem()[0])
			_addBranch(start_state, relative_pixels, relative_disk_center)
		else:
			cur_state = start_state
			while len(relative_pixels) > 0:
				if cur_state.relative_pixel in relative_pixels:
					del relative_pixels[cur_state.relative_pixel]
					cur_state = cur_state.true_nextstate
				else:
					if cur_state.false_nextstate:
						cur_state = cur_state.false_nextstate
					else:
						cur_state.false_nextstate = _State(relative_pixels.popitem()[0])
						_addBranch(cur_state.false_nextstate, relative_pixels, relative_disk_center)

	return start_state

# Computes disks centers positions by matching them with a mask.
# We assume input contains disks with no error, potentially overlapping, such as resources.
# Linear time complexity using a generalized Rabin-Karp algorithm.
def match_disk_overlapping_no_error(screen, disks_radius, val):
	mask = _get_disk_mask(disks_radius)
	automaton = _build_automaton(mask)
	centroids = _match(screen, automaton, val)
	return centroids, mask