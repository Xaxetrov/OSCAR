import random
from oscar.agent.custom_agent import CustomAgent
from oscar.constants import *
from oscar.meta_action.build import *
from oscar.util.location import Location


class ArmySupplier(CustomAgent):

    # states
    _IDLE = 0
    _BUILDING_BARRACKS = 1
    _TRAINING_MARINE = 2


    def __init__(self, message="I hate you"):
        self._message = message
        self._state = ArmySupplier._IDLE
        self._worker_selected = False
        self._barracks_selected = False
        super().__init__()


    def step(self, obs, locked_choice=None):
        play = {}

        """ Selects new state """
        if self._state == ArmySupplier._IDLE:
            self._worker_selected = False
            self._barracks_selected = False

            if obs.observation['player'][MINERALS] > 500 \
                and len(self._shared['army'].barracks) < 3:
                self._state = ArmySupplier._BUILDING_BARRACKS

            elif obs.observation['player'][FOOD_CAP] - obs.observation['player'][FOOD_USED] > 0 \
                and len(self._shared['army'].barracks) > 0 \
                and obs.observation['player'][MINERALS] >= 50:
                self._state = ArmySupplier._TRAINING_MARINE

        """ Executes states """
        if self._state == ArmySupplier._BUILDING_BARRACKS:
            if not self._worker_selected:
                if obs.observation["player"][IDLE_WORKER_COUNT] > 0:
                    play['actions'] = [actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION])]
                    self._worker_selected = True

                else: # if no idle scv, select one randomly
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

            elif BUILD_BARRACKS in obs.observation["available_actions"]:
                building_center = get_random_building_point(obs, self._shared, 15)
                if building_center:
                    play['actions'] = [actions.FunctionCall(BUILD_BARRACKS,
                        [NOT_QUEUED, building_center.get_flipped().to_array()])]
                    self._shared['army'].add_barracks(obs, self._shared,
                        Location(camera_loc=self._shared['camera'].location(obs, self._shared), screen_loc=building_center))
                else:
                    print("Can't find barracks location")
                self._state = ArmySupplier._IDLE

        elif self._state == ArmySupplier._TRAINING_MARINE:
            if not self._barracks_selected:
                on_screen_barracks = self._shared['screen'].scan_units(obs, self._shared, [TERRAN_BARRACKS_ID], PLAYER_SELF)

                if len(on_screen_barracks) == 0:
                    selected_barracks = random.choice(self._shared['army'].barracks)
                    play['actions'] = [actions.FunctionCall(MOVE_CAMERA, [selected_barracks.camera.to_array()])]

                else:
                    selected_barracks = random.choice(on_screen_barracks)
                    play['actions'] = [actions.FunctionCall(SELECT_POINT, 
                        [NEW_SELECTION, selected_barracks.location.screen.get_flipped().to_array()])]
                    self._barracks_selected = True     
            else:
                if TRAIN_MARINE_QUICK in obs.observation["available_actions"]:
                    play['actions'] = [actions.FunctionCall(TRAIN_MARINE_QUICK, [NOT_QUEUED])]
                    self._shared['army'].add_marine()
                self._state = ArmySupplier._IDLE


        if 'actions' not in play:
            self._state = ArmySupplier._IDLE
            play['actions'] = [actions.FunctionCall(NO_OP, [])]

        if self._state != ArmySupplier._IDLE:
            play['locked_choice'] = True
        
        return play


    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)