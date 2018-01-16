import random
from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action.scout import *
from oscar.util.scout import *
from oscar.util.attack import *
from oscar.shared.env import Env


class Scout(CustomAgent):

    _MIN_EXPLORED_RATIO = 0.3

    def __init__(self, message=""):
        self._message = message
        self._worker_selected = False
        super().__init__()

    def step(self, obs, locked_choice=None):
        play = {}

        if not is_enemy_visible(obs) \
            or compute_explored_ratio(obs, self._shared) < Scout._MIN_EXPLORED_RATIO:
            if self._worker_selected and MOVE_MINIMAP in obs.observation["available_actions"]:
                print("move minimap")
                play['actions'] = scout(obs, self._shared)
                self._worker_selected = False
            else:
                play['locked_choice'] = True

                if obs.observation["player"][IDLE_WORKER_COUNT] > 0:
                    print("idle select")
                    play['actions'] = [actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION])]
                    self._worker_selected = True

                else: # if no idle scv, select one randomly
                    print("screen select")
                    on_screen_scv = self._shared['screen'].scan_units(obs, self._shared, [TERRAN_SCV], PLAYER_SELF)
                    
                    if len(on_screen_scv) == 0:
                        if len(self._shared['economy'].command_centers) > 0:
                            selected_command_center = random.choice(self._shared['economy'].command_centers)
                            play['actions'] = [actions.FunctionCall(MOVE_CAMERA,
                                [selected_command_center.camera.to_array()])]
                    else:
                        selected_scv = random.choice(on_screen_scv)
                        play['actions'] = [actions.FunctionCall(SELECT_POINT, 
                            [NEW_SELECTION, selected_scv.location.screen.get_flipped().to_array()])]
                        self._worker_selected = True

        if 'actions' not in play:
            self._worker_selected = False
            play['locked_choice'] = False
            play['actions'] = [actions.FunctionCall(NO_OP, [])]
            
        return play


    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)