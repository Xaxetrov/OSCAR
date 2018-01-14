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


    def step(self, obs):
        play = {}

        self._shared_objects['env'].timestamp += 1
        self._shared_objects["idle_tracker"].update(obs, self._shared_objects['env'].timestamp)
        res = self._shared_objects["idle_tracker"].search_idle_unit(obs)

        if res['unit']:
            play['actions'] = \
                [actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, res['unit'].location.screen.get_flipped().to_array()])] \
                + scout(obs)

        elif res['actions']:
            play['actions'] = res['actions']
            play['locked_choice'] = True

        else: # failed to find an idle unit
            play['actions'] = [actions.FunctionCall(NO_OP, [])]
            
        return play


    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)