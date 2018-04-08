from oscar.agent.custom_agent import CustomAgent
from oscar.constants import *
from oscar.meta_action.select import select_idle_scv
from oscar.meta_action.harvest import harvest_mineral


class IdleSCVManagerBasic(CustomAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs, locked_choice=None):
        play = {}
        if SELECT_IDLE_WORKER in obs.observation['available_actions']:
            play['actions'] = select_idle_scv(obs) + harvest_mineral(obs)
        else:
            play['actions'] = [actions.FunctionCall(NO_OP, [])]

        return play


