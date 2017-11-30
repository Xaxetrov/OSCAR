from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action import *


class Economic(CustomAgent):
    def __init__(self, message="I hate you"):
        self.supply_depot_built = False
        self.barracks_built = False
        self._message = message
        super().__init__()

    def set_supply_depot_built(self):
        print("supply depot")
        self.supply_depot_built = True

    def set_barracks_built(self):
        print("barracks")
        self.barracks_built = True

    def step(self, obs):
        play = {}

        if not self.supply_depot_built:
            meta_action = build(obs, 2, BUILD_SUPPLY_DEPOT)
            play["actions"] = meta_action
            play["success_callback"] = self.set_supply_depot_built
            return play

        if not self.barracks_built:
            meta_action = build(obs, 3, BUILD_BARRACKS)
            play["actions"] = meta_action
            play["success_callback"] = self.set_barracks_built
            return play

        try:
            meta_action = train_unit(obs, TERRAN_BARRACKS_ID, TRAIN_MARINE_QUICK)
        except NoUnitError:
            meta_action = [actions.FunctionCall(NO_OP, [])]
        play["actions"] = meta_action
        return play

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
