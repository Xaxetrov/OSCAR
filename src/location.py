import math

#############################
# Represents a location
# and provides helpful methods.
#############################
class Location():

    def __init__(self, _x = None, _y = None):
        self.x = _x
        self.y = _y

    def squarredDistance(self, other):
    	return (self.x-other.x)*(self.x-other.x) + (self.y-other.y)*(self.y-other.y)

    def distance(self, other):
    	return math.sqrt(self.squarredDistance(other));

    def toArray(self):
    	return [self.x, self.y]

    def equals(self, other):
        return (self.x == other.x and self.y == other.y)

    def difference(self, other):
    	return Location(self.x - other.x, self.y - other.y)

    def addition(self, other):
    	return Location(self.x + other.x, self.y + other.y)

    def __str__(self):
     return "Location("+str(self.x)+","+str(self.y)+")"