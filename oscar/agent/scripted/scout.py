from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action.scout import *
from oscar.shared.env import Env


class Scout(CustomAgent):
    def __init__(self, message=""):
        self._message = message
        super().__init__()

    @staticmethod
    def scout_sent():
        print("scout sent")


    def step(self, obs, locked_choice=None):
        play = {}

        self._shared['env'].timestamp += 1
        self._shared['idle_tracker'].update(obs, self._shared)
        res = self._shared["idle_tracker"].search_idle_unit(obs, self._shared)

        if res['unit']:
            play['actions'] = \
                [actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, res['unit'].location.screen.get_flipped().to_array()])] \
                + scout(obs, self._shared)

        elif res['actions']:
            play['actions'] = res['actions']
            play['locked_choice'] = True

        else: # failed to find an idle unit
            play['actions'] = [actions.FunctionCall(NO_OP, [])]
            
        return play


    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)