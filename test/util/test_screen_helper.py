import pickle 
import numpy as np
from oscar.util.screen_helper import *
from random import randint
from oscar.constants import *


# Used to create dummy observations, like in pysc2.
class Obs:
    def __init__(self, screen_unit_type, player_relative, unit_density):
        self.observation = {"screen": []}
        for i in range(17):
            self.observation[SCREEN].append(None)

        self.observation[SCREEN][SCREEN_UNIT_TYPE] = screen_unit_type
        self.observation[SCREEN][SCREEN_PLAYER_RELATIVE] = player_relative
        self.observation[SCREEN][SCREEN_UNIT_DENSITY] = unit_density


def print_screen(screen):
    """ Tests for the function "oscar.util.get_centroids """
    alias = {}
    next_alias = 1

    for j in range(len(screen[0])):
        line = ""
        for i in range(len(screen)):
            if screen[i][j] == 0:
                line += "0"
            else:
                if screen[i][j] not in alias:
                    alias[screen[i][j]] = next_alias
                    next_alias += 1

                line += str(alias[screen[i][j]])
        print(line)

    print("alias: " + str(alias))
    print("")


def get_disk_mask(radius):

    def draw_hor_line(mask, x0, x1, y):
        for x in range(x0, x1+1):
            if 0 <= x < len(mask) and 0 <= y < len(mask[0]):
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


def draw_disk(screen_unit_type, unit_id, player_relative, player_id, unit_density, x0, y0, radius):

    mask = get_disk_mask(radius)

    for x in range(len(mask)):
        for y in range(len(mask[0])):
            abs_x = x0 + x
            abs_y = y0 + y

            if 0 <= abs_x < len(screen_unit_type) and 0 <= abs_y < len(screen_unit_type[0]) and mask[x, y] != 0:
                screen_unit_type[abs_x, abs_y] = unit_id
                player_relative[abs_x, abs_y] = player_id
                unit_density[abs_x, abs_y] += 1


def test_match_all():
    """ Generates a random sample screen to test the match_all algorithm """
    SCREEN_WIDTH = 30
    SCREEN_HEIGHT = 30

    screen_unit_type = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
    player_relative = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
    unit_density = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
    coordinates = []
    
    unit_id = 1
    player_id = 1
    radius = randint(1, 6)
   
    disks = randint(0, 6)
    for i in range(disks):
        x = randint(-radius, SCREEN_WIDTH-radius)
        y = randint(-radius, SCREEN_HEIGHT-radius)
        draw_disk(screen_unit_type, unit_id, player_relative, player_id, unit_density, x, y, radius)
        coordinates.append((y+radius, x+radius))

    other_disks = randint(0, 6)
    for i in range(other_disks):
        x = randint(-radius, SCREEN_WIDTH-radius)
        y = randint(-radius, SCREEN_HEIGHT-radius)
        other_radius = randint(1, 6)
        other_id = randint(2, 100)
        draw_disk(screen_unit_type, unit_id, player_relative, 2, unit_density, x, y, other_radius)

    print("screen_unit_type:")
    print_screen(screen_unit_type)
    print("player_relative:")
    print_screen(player_relative)
    print("unit_density:")
    print_screen(unit_density)

    centroids = match_all(screen_unit_type, unit_id, player_relative, player_id, unit_density, radius)

    print("---------------")
    print("Expected (" + str(disks) + "):")
    coordinates.sort(key=lambda tup: (tup[0], tup[1]))
    for i in range(disks):
        print(str(coordinates[i]))

    print("\nReturned (" + str(len(centroids)) + "):")
    centroids.sort(key=lambda tup: (tup[0], tup[1])) 
    for i in range(len(centroids)):
        print(str(centroids[i]))


def test_get_center():
    obs = pickle.load(open("test/util/obs.p", "rb"))
    obs_id = 49

    print("screen_unit_type:")
    print_screen(obs[obs_id].observation[SCREEN][SCREEN_UNIT_TYPE])
    print("player_relative:")
    print_screen(obs[obs_id].observation[SCREEN][SCREEN_PLAYER_RELATIVE])
    print("unit_density:")
    print_screen(obs[obs_id].observation[SCREEN][SCREEN_UNIT_DENSITY])

    print(get_center(obs[obs_id], TERRAN_COMMAND_CENTER, 1))
    print(get_center(obs[obs_id], TERRAN_SCV, 1))
    print(get_center(obs[obs_id], MINERAL_FIELD, 3))


if __name__ == "__main__":
    test_match_all()
    #test_get_center()
