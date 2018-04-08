from oscar.agent.commander.context_save_commender import ContextSaveCommander
from oscar.agent.scripted.idle_scv import IdleSCVManagerBasic
from oscar.constants import *


class CollectIdleScvCommander(ContextSaveCommander):
    def __init__(self, subordinates: list):
        super().__init__(subordinates)
        if len(subordinates) != 2:
            raise ValueError("CollectIdleScvCommander can only manage two subordinate")
        self.idle_manager = None
        self.other_subordinate = None
        for a in subordinates:
            if type(a) == IdleSCVManagerBasic:
                self.idle_manager = a
            else:
                self.other_subordinate = a
        if self.idle_manager is None or self.other_subordinate is None:
            raise ValueError("CollectIdleScvCommander must have one and only one IdleSCVManagerBasic as child")

    def choose_subordinate(self, obs, locked_choice):
        if SELECT_IDLE_WORKER in obs.observation['available_actions']:
            return self.idle_manager
        else:
            return self.other_subordinate
