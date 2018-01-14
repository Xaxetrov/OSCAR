from oscar.util.point import Point


class Location():

    def __init__(self, minimap_loc=None, screen_loc=None, camera_loc=None):
        self.minimap = minimap_loc
        self.screen = screen_loc
        self.camera = camera_loc

    def compute_minimap_loc(self, obs, shared):
        scale_x = shared['camera'].width(obs) / shared['screen'].width(obs)
        scale_y = shared['camera'].height(obs) / shared['screen'].height(obs)

        self.minimap = shared['minimap'].bound(obs, Point(
            int(round(self.camera.x - 0.5*shared['camera'].width(obs) + self.screen.x * scale_x)),
            int(round(self.camera.y - 0.5*shared['camera'].height(obs) + self.screen.y * scale_y))))

    def squarred_distance(self, other):
        if not self.minimap or not other.minimap:
            raise Exception("Can't compute distance")
        return self.minimap.distance(other.minimap)

    def distance(self, other):
        return math.sqrt(self.squarred_distance(other))

    def equals(self, other):
        def equals_minimap():
            return (not self.minimap and not other.minimap) \
                or (self.minimap and other.minimap and self.minimap.equals(other.minimap))
        def equals_screen():
            return (not self.screen and not other.screen) \
                or (self.screen and other.screen and self.screen.equals(other.screen))
        def equals_camera():
            return (not self.camera and not other.camera) \
                or (self.camera and other.camera and self.camera.equals(other.camera))

        return (equals_minimap() and equals_screen() and equals_camera())

    def __str__(self):
     return "Location {\n" \
        + "  minimap: " + str(self.minimap) + "\n" \
        + "  screen: " + str(self.screen) + "\n" \
        + "  camera: " + str(self.camera) + "\n" \
        + "}"