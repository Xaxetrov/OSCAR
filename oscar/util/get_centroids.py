import numpy as np

def _print_mat(mat):
    for i in range(len(mat)):
        line = ""
        for j in range(len(mat[0])):
            if mat[i][j] == 1:
                line += "x "
            else:
                line += "  "
        print(line)
    print("")


def _match(screen, automaton):
    return


# builds mask using midpoint circle algorithm
def get_disk_mask(radius):

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
    return


# Computes disks centers positions by matching them with a mask.
# We assume input contains disks with no error, potentially overlapping, such as resources.
# Linear time complexity using a generalized Rabin-Karp algorithm.
def match_disk_overlapping_no_error(screen, disks_radius):
    _get_disk_mask(disks_radius)
