import numpy as np


class MapRemembrance(object):
    """
    Memoire permettant une gestion intelligente des features layers :
    Permet de garder trace des éléments vus antérieurement, avec un facteur de certitude fourni en fonction
    de quand il a été vu
    """
    __visible = 2
    __decay_factor = 1.01

    def __init__(self, size, decaying_function=None):
        """

        :param size: taille x et y de la feature layer a conserver
        :param decaying_function: si différent de None, remplace la fonction de décrémentation de certitude par decaying_function
        """
        self.__map = np.zeros((size[0], size[1], 2))
        if decaying_function is not None:
            self.decaying_function = decaying_function
        else:
            self.decaying_function = lambda x: x / self.__decay_factor

    def maj_informations(self, minimap_informations, visible_parts):
        """
        Met à jour la mémoire sur les parties actuellements visibles. Décrémente la certitude sur les parties non visibles
        :param minimap_informations: 2D-nparray like, Une feature layer de la minimap
        :param visible_parts: 2D-nparray like, Quelles parties de la minimap sont effectivement visibles
        :return: None
        """

        # cherche les parties visibles
        indices = np.where(visible_parts == self.__visible)
        # toutes les autres cases
        inverse_indices = np.where(visible_parts != self.__visible)

        self.__map[:, :, 0][indices] = minimap_informations[indices]
        self.__map[:, :, 1][indices] = 1
        self.__map[:, :, 1][inverse_indices] = self.decaying_function(self.__map[:, :, 1][inverse_indices])

    def get_most_recent_informations(self, value, layer=0):
        """
        Récupère les points de la feature layer correspondant à value, et les retourne de façon ordonnés avec leur indice de certitude
        :param value: la valeur a rechercher dans la feature layer
        :param layer: 0 pour le moment, ne pas changer, présent pour anticiper une évolution vers un outil qui sauvegarderait de multiples features layers
        :return: un tuple (coordonnees, probabilités) tel que :
            - coordonnees : 2D-np array de forme (nombre de points trouvés, 2), les points retrouvés dans la feature layer
            - probabilités : 1D-array de taille (nombre de points trouvés,), les proba associées aux points, telles que probabilités[0] = max(probabilités)
        """

        # cherche les correspondances
        indices = np.where(self.__map[:, :, layer] == value)
        # recupère les indices de certitude associés
        proba = self.__map[:, :, -1][indices]

        # ordonne les proba pour avoir le plus sur en premier
        order = proba.argsort()[::-1]
        proba = proba[order]

        # transforme les tuples de coordonnees en tableau
        if len(indices) > 0:
            coordinates = np.zeros((len(indices[0]), len(indices)))
            for i, dimension in enumerate(indices):
                coordinates[:, i] = dimension
            coordinates = coordinates[order]
        else:
            coordinates = np.ndarray(0)

        return coordinates, proba