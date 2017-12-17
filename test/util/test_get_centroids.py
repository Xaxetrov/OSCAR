import pickle 
import numpy as np
from oscar.util.get_centroids import *
from oscar.meta_action.common import *
from random import randint
from oscar.constants import *

""" Tests for the function "oscar.util.get_centroids """

def print_screen(screen):
    alias = {}
    nextAlias = 1

    for j in range(len(screen[0])):
        line = ""
        for i in range(len(screen)):
            if screen[i][j] == 0:
                line += "0"
            else:
                if screen[i][j] not in alias:
                    alias[screen[i][j]] = nextAlias
                    nextAlias += 1

                line += str(alias[screen[i][j]])
        print(line)

    print("alias: " + str(alias))
    print("")


def get_disk_mask(radius):

    def draw_hor_line(mask, x0, x1, y):
        for x in range(x0, x1+1):
            if x >= 0 and x < len(mask) \
                and y >= 0 and y < len(mask[0]):
                mask[x][y] = 1

    def draw(mask, x0, y0, x, y):
        draw_hor_line(mask, x0 - x, x0 + x, y0 + y)
        draw_hor_line(mask, x0 - x, x0 + x, y0 - y)
        draw_hor_line(mask, x0 - y, x0 + y, y0 + x)
        draw_hor_line(mask, x0 - y, x0 + y, y0 - x)

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


def draw_disk(screen, unit_density, x0, y0, radius, unit_id):

    mask = get_disk_mask(radius)

    for x in range(len(mask)):
        for y in range(len(mask[0])):
            abs_x = x0 + x
            abs_y = y0 + y

            if abs_x >= 0 and abs_x < len(screen) \
                and abs_y >= 0 and abs_y < len(screen[0]) \
                and mask[x, y] != 0:
                screen[abs_x, abs_y] = unit_id
                unit_density[abs_x, abs_y] += 1


if __name__ == "__main__":
    
    SCREEN_WIDTH = 30
    SCREEN_HEIGHT = 30

    screen = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
    unit_density = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
    coordinates = []
    
    unit_id = 1
    radius = randint(1, 6)
    disks = randint(0, 6)
    other_disks = randint(0, 6)

    for i in range(disks):
        x = randint(-radius, SCREEN_WIDTH-radius)
        y = randint(-radius, SCREEN_HEIGHT-radius)
        draw_disk(screen, unit_density, x, y, radius, unit_id)
        coordinates.append((x+radius, y+radius))

    for i in range(other_disks):
        x = randint(-radius, SCREEN_WIDTH-radius)
        y = randint(-radius, SCREEN_HEIGHT-radius)
        other_radius = randint(1, 6)
        other_id = randint(2, 100)
        draw_disk(screen, unit_density, x, y, other_radius, other_id)

    print("screen:")
    print_screen(screen)
    print("unit_density:")
    print_screen(unit_density)

    centroids = get_centroid_centers(screen, unit_density, radius, unit_id)

    print("---------------")
    print("Expected (" + str(disks) + "):")
    coordinates.sort(key=lambda tup: (tup[0], tup[1]))
    for i in range(disks):
        print(str(coordinates[i]))

    print("\nReturned (" + str(len(centroids)) + "):")
    centroids.sort(key=lambda tup: (tup[0], tup[1])) 
    for i in range(len(centroids)):
        print(str(centroids[i]))