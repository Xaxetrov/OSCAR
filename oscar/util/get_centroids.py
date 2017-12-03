import numpy as np

class _State():
    def __init__(self, _relative_pixel):
        self.relative_pixel = _relative_pixel
        self.true_nextstate = None
        self.false_nextstate = None
        self.relative_disk_center = {'x': None, 'y': None}

def _match(screen, automaton):
    return

# builds mask using midpoint circle algorithm
def _get_disk_mask(radius):

    def draw_hor_line(mat, x0, x1, y):
        for x in range(x0, x1+1):
            if x >= 0 and x < len(mat) \
                and y >= 0 and y < len(mat[0]):
                mat[x][y] = 1

    def draw(mat, x0, y0, x, y):
        draw_hor_line(mat, x0 - x, x0 + x, y0 + y)
        draw_hor_line(mat, x0 - x, x0 + x, y0 - y)
        draw_hor_line(mat, y0 - y, y0 + y, x0 + x)
        draw_hor_line(mat, y0 - y, y0 + y, x0 - x)

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
def match_disk_overlapping_no_error(screen, disks_radius):
    mask = _get_disk_mask(disks_radius)
    automaton = _build_automaton(mask)