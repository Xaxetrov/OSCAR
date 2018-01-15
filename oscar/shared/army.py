class Army:
	""" Stores army state """

	def __init__(self):
		self.barracks = []
		self.fresh_new_marines = []

	def add_barracks(self, obs, shared, location):
		location.compute_minimap_loc(obs, shared)
		self.barracks.append(location)

	def add_marine(self, barracks_loc):
		self.fresh_new_marines.append(barracks_loc)