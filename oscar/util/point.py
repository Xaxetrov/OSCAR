import math


class Point:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def squared_distance(self, other):
        return (self.x-other.x)*(self.x-other.x) + (self.y-other.y)*(self.y-other.y)

    def distance(self, other):
        return math.sqrt(self.squared_distance(other))

    def get_flipped(self):
        return Point(self.y, self.x)

    def to_array(self):
        return [self.x, self.y]

    def equals(self, other):
        return self.x == other.x and self.y == other.y

    def difference(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def addition(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __str__(self):
        return "Point("+str(self.x)+","+str(self.y)+")"
