from pysc2.agents import base_agent

from oscar.meta_action import *


class Economic(base_agent.BaseAgent):
    def __init__(self):
        self.actions_list = []
        self.supply_depot_built = False
        self.noop_mode = False
        super().__init__()

    def step(self, obs):
        if self.noop_mode:
            return actions.FunctionCall(NO_OP, [])

        if len(self.actions_list) != 0:
            return self.actions_list.pop(0)

        if obs.observation["player"][MINERALS] >= 100 and not self.supply_depot_built:
            self.actions_list = build(obs, 2, BUILD_SUPPLY_DEPOT)
            self.supply_depot_built = True
            return self.actions_list.pop(0)

        if obs.observation["player"][MINERALS] >= 400 and self.supply_depot_built:
            try:
                self.actions_list = build(obs, 3, BUILD_BARRACKS)
            except NoValidBuildingLocationError:
                print("activate no op mode")
                self.noop_mode = True
                return actions.FunctionCall(NO_OP, [])
            return self.actions_list.pop(0)

        return actions.FunctionCall(NO_OP, [])

