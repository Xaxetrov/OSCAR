class Unit:

    def __init__(self, location, unit_id=None, player_id=None):
        self.location = location
        self.unit_id = unit_id
        self.player_id = player_id

    def equals(self, other):
        def is_same_location():
            return (not self.location and not other.location) \
                or (self.location and other.location and self.location.equals(other.location))
        def is_same_unit_id():
            return (self.unit_id is None and other.unit_id is None) \
                or (self.unit_id is not None and other.unit_id is not None and self.unit_id == other.unit_id)
        def is_same_player_id():
            return (self.player_id is None and other.player_id is None) \
                or (self.player_id is not None and other.player_id is not None and self.player_id == other.player_id)
        
        return is_same_location() and is_same_unit_id() and is_same_player_id()

