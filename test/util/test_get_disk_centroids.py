import pickle 
import numpy as np
from oscar.util.misc import *
from oscar.meta_action.common import *
from random import randint
from oscar.constants import *

############
# Tests for the function "oscar.util.misc.get_disks_centroids()"
############

SCREEN_SIZE = 60

def draw(x, y):
	mat = np.zeros((SCREEN_SIZE, SCREEN_SIZE))
	for i in range(len(x)):
		mat[x[i]][y[i]] = 1

	for i in range(SCREEN_SIZE):
		line = ""
		for j in range(SCREEN_SIZE):
			if mat[i][j] == 0:
				line += "__"
			else:
				line += "x_"
		print(line)
	print("")

def draw_line(screen, x0, x1, y):
	for x in range(x0, x1+1):
		if x >= 0 and x < len(screen) \
			and y >= 0 and y < len(screen[0]):
			screen[x][y] = 1

def draw_circle(screen, x0, y0, radius):
    x = radius-1
    y = 0
    dx = 1
    dy = 1
    err = dx - 2*radius

    while x >= y:
        draw_line(screen, x0 - x, x0 + x, y0 + y)
        draw_line(screen, x0 - y, x0 + y, y0 + x)
        draw_line(screen, x0 - x, x0 + x, y0 - y)
        draw_line(screen, x0 - y, x0 + y, y0 - x)

        if err <= 0:
            y += 1
            err += dy
            dy += 2
        if err > 0:
            x -= 1
            dx += 2
            err += dx - 2*radius

screen = np.zeros((SCREEN_SIZE, SCREEN_SIZE))

nbDisks = randint(1, 12)
radius = randint(2, 10)

for j in range(nbDisks):
	x = randint(0, SCREEN_SIZE-1)
	y = randint(0, SCREEN_SIZE-1)
	draw_circle(screen, x, y, radius)

points_x = []
points_y = []
for x in range(SCREEN_SIZE):
	for y in range(SCREEN_SIZE):
		if screen[x][y] == 1:
			points_x.append(x);
			points_y.append(y);

draw(points_x, points_y)
centroids_x, centroids_y = get_disks_centroids(points_x, points_y)
draw(centroids_x, centroids_y)