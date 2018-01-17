class Army:
	""" Stores army state """

	def __init__(self):
		self.barracks = []
		self.marines = 0

	def add_barracks(self, obs, shared, location):
		location.compute_minimap_loc(obs, shared)
		self.barracks.append(location)

	def add_marine(self):
		self.marines += 1