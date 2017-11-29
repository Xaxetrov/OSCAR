from oscar.meta_action import *


class Economic():
    def __init__(self):
        self.supply_depot_built = False
        self.barracks_built = False
        super().__init__()

    def set_supply_depot_not_built(self):
        self.supply_depot_built = False

    def set_barracks_not_built(self):
        self.barracks_built = False

    def step(self, obs):
        if not self.supply_depot_built:
            self.supply_depot_built = True
            meta_action = build(obs, 2, BUILD_SUPPLY_DEPOT)
            return (meta_action, self.set_supply_depot_not_built)

        if not self.barracks_built:
            self.barracks_built = True
            meta_action = build(obs, 3, BUILD_BARRACKS)
            return (meta_action, self.set_barracks_not_built)

        try:
            meta_action = train_unit(obs, TERRAN_BARRACKS_ID, TRAIN_MARINE_QUICK)
        except NoUnitError:
            meta_action = [actions.FunctionCall(NO_OP, [])]
        return (meta_action, )

