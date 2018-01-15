import random
from oscar.agent.custom_agent import CustomAgent
from oscar.constants import *
from oscar.meta_action.build import *
from oscar.util.location import Location
from oscar.util.debug import *


class ArmySupplier(CustomAgent):
    def __init__(self, message="I hate you"):
        self._message = message
        self._worker_selected = False
        self._barracks_selected = False
        self._barracks_loc = None
        super().__init__()


    def step(self, obs, locked_choice=None):
        play = {}
        play['actions'] = []

        if obs.observation['player'][MINERALS] >= 150 \
            and len(self._shared['army'].barracks) == 0 \
            and len(self._shared['economy'].supply_depots) > 0:

            if not self._worker_selected:
                if obs.observation["player"][IDLE_WORKER_COUNT] > 0:
                    play['actions'] = [actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION])]
                    play['locked_choice'] = True
                    self._worker_selected = True
                    return play

                else:
                    """ if there is no idle scv, select one randomly """
                    on_screen_scv = self._shared['screen'].scan_units(obs, self._shared, [TERRAN_SCV], PLAYER_SELF)
                    selected_scv = random.choice(on_screen_scv)
                    play['actions'] = [actions.FunctionCall(SELECT_POINT, 
                        [NEW_SELECTION, selected_scv.location.screen.get_flipped().to_array()])]
                    play['locked_choice'] = True
                    self._worker_selected = True
                    return play

            elif BUILD_BARRACKS in obs.observation["available_actions"]:
                building_center = get_random_building_point(obs, self._shared, 15)
                if building_center:
                    play['actions'] = [actions.FunctionCall(BUILD_BARRACKS, [NOT_QUEUED, building_center.get_flipped().to_array()])]
                    self._worker_selected = False
                    self._shared['army'].add_barracks(obs, self._shared,
                        Location(camera_loc=self._shared['camera'].location(obs, self._shared), screen_loc=building_center))
                    return play
                else:
                    print("can't find barracks location")

        elif len(self._shared['army'].barracks) > 0 \
            and obs.observation['player'][MINERALS] >= 50:

            if not self._barracks_selected:
                self._barracks_loc = self._shared['army'].barracks[0]
                play['actions'] = [actions.FunctionCall(MOVE_CAMERA, [self._barracks_loc.camera.to_array()]),
                    actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, self._barracks_loc.screen.get_flipped().to_array()])]

                play['locked_choice'] = True
                self._barracks_selected = True
                return play
            else:
                if TRAIN_MARINE_QUICK in obs.observation["available_actions"]:
                    play['actions'] = [actions.FunctionCall(TRAIN_MARINE_QUICK, [NOT_QUEUED])]
                    self._shared['army'].add_marine(self._barracks_loc)
                self._barracks_selected = False

        if len(play['actions']) == 0:
            play['actions'] = [actions.FunctionCall(NO_OP, [])]
        
        return play


    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)