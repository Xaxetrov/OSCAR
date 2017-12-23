from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action import *

class Explorator(CustomAgent):
    def __init__(self, message="I hate you"):
        self._message = message
        self.coordinates_helper = Coordinates_helper()
        self.cur_location = None
        super().__init__()

    def scout_sent(self):
        print("scout sent")

    def step(self, obs):
        if not self.cur_location:
            self.cur_location = self.coordinates_helper.get_loc_in_minimap(obs)

        play = {}

        meta_action = None
        try:
            meta_action = scout(obs, self.coordinates_helper, self.cur_location)
            if meta_action:
                play["success_callback"] = self.scout_sent
        except NoValidSCVError:
            print("scouting failed")
            meta_action = [actions.FunctionCall(NO_OP, [])]

        play["actions"] = meta_action
        return play

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
