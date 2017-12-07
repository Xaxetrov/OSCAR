from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action import *
from oscar.constants import PLAYER_HOSTILE

from absl import logging

class Scout(CustomAgent):
    def __init__(self, message="I hate you"):
        self.supply_depot_built = False
        self.barracks_built = False
        self._message = message
        super().__init__()

    def step(self, obs):
        play = {}

        meta_action = [actions.FunctionCall(NO_OP, [])]
        play["actions"] = meta_action
        points, _ = self._shared_objects["ennemies"].get_most_recent_informations(PLAYER_HOSTILE)
        if len(points) > 0:
            logging.info("found ennemies at positions {}".format(points))
        return play

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
