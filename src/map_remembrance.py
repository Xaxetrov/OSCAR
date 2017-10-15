import numpy as np
from pysc2.lib import features

class MapRemembrance(object):
    
    __visible = 2
    __decay_factor = 1.01
    
    def __init__(self, size):
        self.__map = np.zeros((size[0], size[1], 2))
        
    def maj_informations(self, minimap_informations, visible_parts):
        indices = np.where(visible_parts == self.__visible)
        inverse_indices = np.where(visible_parts != self.__visible)
        
        self.__map[:, :, 0][indices] = minimap_informations[indices]
        self.__map[:, :, 1][indices] = 1
        self.__map[:, :, 1][inverse_indices] /= self.__decay_factor
        
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