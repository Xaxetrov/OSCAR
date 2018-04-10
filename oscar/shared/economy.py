class Economy:
    """ Stores economic state """

    def __init__(self):
        self.supply_depots = []
        self.command_centers = []
        self.scv = 8

    def add_supply_depot(self, obs, shared, location):
        location.compute_minimap_loc(obs, shared)
        self.supply_depots.append(location)

    def add_command_center(self, obs, shared, location):
        location.compute_minimap_loc(obs, shared)
        self.command_centers.append(location)

    def add_scv(self):
        self.scv += 1
