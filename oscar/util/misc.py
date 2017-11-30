import numpy as np
from oscar.constants import *

def draw(mat):
	for i in range(60):
		line = ""
		for j in range(60):
			if mat[i][j] > 0:
				line += "x_"
			else:
				line += "__"
		print(line)
	print("")

# Given a list of pixels coordinates corresponding to disks,
# returns the coordinates of approximate centroids, using erosion.
# Runs in O(n) with n the number of input pixels.
def get_disks_centroids(pixels_x, pixels_y):
	# Populates a matrix with input data
	mat = np.zeros((SCREEN_RESOLUTION, SCREEN_RESOLUTION))
	for p_x, p_y in zip(pixels_x, pixels_y):
		mat[p_x][p_y] = 1

	# Erodes iteratively
	centers_x = []
	centers_y = []
	to_erode = zip(pixels_x, pixels_y)
	while True:
		to_erode_next = []
		eroded = []

		for x, y in to_erode:
			isCentroid, isEroded, neighbors = _erode(mat, x, y)
			if isEroded:
				eroded.append((x, y))
			if isCentroid:
				centers_x.append(x)
				centers_y.append(y)
			
			to_erode_next += neighbors

		if len(to_erode_next) == 0:
			break

		for x, y in eroded:
			mat[x][y] = 0

		to_erode = to_erode_next

	draw(mat)
	return centers_x, centers_y

def _erode(mat, x, y):
	unseen_neighbors = []
	neighbors = _get_neighbors(mat, x, y)
	isEroded = False

	isCentroid = (len(neighbors) <= 1)

	if len(neighbors) > 1 and len(neighbors) < 4:
		isEroded = True
		for n_x, n_y in neighbors:
			if mat[n_x][n_y] == 1:
				mat[n_x][n_y] = 0.5
				unseen_neighbors.append((n_x, n_y))
	return isCentroid, isEroded, unseen_neighbors

def _get_neighbors(mat, x, y):
	neighbors = []
	if x-1 > 0 and mat[x-1][y] > 0:
		neighbors.append((x-1, y))
	if x+1 < SCREEN_RESOLUTION and mat[x+1][y] > 0:
		neighbors.append((x+1, y))
	if y-1 > 0 and mat[x][y-1] > 0:
		neighbors.append((x, y-1))
	if y+1 < SCREEN_RESOLUTION and mat[x][y+1] > 0:
		neighbors.append((x, y+1))
	return neighbors



