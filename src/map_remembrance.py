import numpy as np
from pysc2.lib import features

class MapRemembrance(object):
    
    __visible = 2
    __decay_factor = 1.01
    
    def __init__(self, size, decaying_function=None):
        self.__map = np.zeros((size[0], size[1], 2))
        if decaying_function is not None:
            self.decaying_function = decaying_function
        else:
            self.decaying_function = lambda x: x/self.__decay_factor
        
    def maj_informations(self, minimap_informations, visible_parts):
        indices = np.where(visible_parts == self.__visible)
        inverse_indices = np.where(visible_parts != self.__visible)
        
        self.__map[:, :, 0][indices] = minimap_informations[indices]
        self.__map[:, :, 1][indices] = 1
        self.__map[:, :, 1][inverse_indices] = self.decaying_function(self.__map[:, :, 1][inverse_indices])
        
    def get_most_recent_informations(self, layer, value):
        indices = np.where(self.__map[:, :, layer] == value)
        proba = self.__map[:, :, -1][indices]
        
        # order the probabilities to get the most accurate information first
        order = proba.argsort()[::-1]
        proba = proba[order]
        
        # reshaping of the coordinates in an array instead of a tuple
        if len(indices) > 0:
            coordinates = np.zeros((len(indices[0]), len(indices)))
            for i, dimension in enumerate(indices):
                coordinates[:, i] = dimension
            coordinates = coordinates[order]
        else:
            coordinates = np.ndarray(0)
        
        return coordinates, proba
    
	#here to be pushed, but meant to be used in AI. not tested yet
	def __scout(self, obs):
        # check on the probability layer to access never seen zones
        black = self.memory.get_most_recent_informations(1, 0)[0]
        if len(black) == 0:
            return actions.FunctionCall(_NO_OP, [])
            
        player_relative = obs.observation["minimap"][_PLAYER_RELATIVE]
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        player = [int(player_x.mean()), int(player_y.mean())]
        
        # compute closest not seen point
        closest, min_dist = None, None
        for p in black:
            dist = np.linalg.norm(np.array(player) - np.array(p))
            if not min_dist or dist < min_dist:
                closest, min_dist = p, dist
    