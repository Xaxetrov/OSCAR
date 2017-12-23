from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action import *
from oscar.util.observer import *

class Watcher(CustomAgent):
    def __init__(self, message="I hate you"):
        self._message = message
        self._observer = Observer()
        self._coordinates_helper = Coordinates_helper()
        super().__init__()

    def step(self, obs):    
        play = {}
        play["actions"] = watch_enemy(obs, self._coordinates_helper, self._observer)
        return play

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
