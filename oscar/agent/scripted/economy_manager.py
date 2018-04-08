import random
from oscar.agent.custom_agent import CustomAgent
from oscar.constants import *
from oscar.meta_action.build import *
from oscar.util.location import Location


class EconomyManager(CustomAgent):

    # states
    _IDLE = 0
    _HARVESTING = 1
    _BUILDING_SUPPLY_DEPOT = 2
    _TRAINING_SCV = 3

    def __init__(self, message="I hate you"):
        self._message = message
        self._state = EconomyManager._IDLE
        self._worker_selected = False
        self._command_center_selected = False
        super().__init__()

    def step(self, obs, locked_choice=None):
        play = {}

        """ Selects new state """
        if self._state == EconomyManager._IDLE:
            self._worker_selected = False
            self._command_center_selected = False

            if obs.observation['player'][FOOD_CAP] - obs.observation['player'][FOOD_USED] < 5 \
                and obs.observation['player'][MINERALS] >= 100:
                self._state = EconomyManager._BUILDING_SUPPLY_DEPOT

            elif obs.observation['player'][FOOD_CAP] - obs.observation['player'][FOOD_USED] > 0 \
                and obs.observation['player'][MINERALS] >= 50 \
                and self._shared['economy'].scv < 20:
                self._state = EconomyManager._TRAINING_SCV

            elif obs.observation["player"][IDLE_WORKER_COUNT] > 0:
                self._state = EconomyManager._HARVESTING

        """ Executes states """
        if self._state == EconomyManager._HARVESTING:
            if not self._worker_selected:
                selected_command_center = random.choice(self._shared['economy'].command_centers)
                play['actions'] = [actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION]), 
                    actions.FunctionCall(MOVE_CAMERA, [selected_command_center.camera.to_array()])]
                self._worker_selected = True
            elif HARVEST_GATHER_SCREEN in obs.observation["available_actions"]:
                minerals = self._shared['screen'].scan_units(obs, self._shared, list(ALL_MINERAL_FIELD), PLAYER_NEUTRAL)
                if len(minerals) > 0:
                    selected_mineral = random.choice(minerals)
                    play['actions'] = [actions.FunctionCall(HARVEST_GATHER_SCREEN, 
                        [NOT_QUEUED, selected_mineral.location.screen.get_flipped().to_array()])]
                self._state = EconomyManager._IDLE
                
        elif self._state == EconomyManager._BUILDING_SUPPLY_DEPOT:
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

            elif BUILD_SUPPLY_DEPOT in obs.observation["available_actions"]:
                building_center = get_random_building_point(obs, self._shared, 11)
                if building_center:
                    play['actions'] = [actions.FunctionCall(BUILD_SUPPLY_DEPOT,
                        [NOT_QUEUED, building_center.get_flipped().to_array()])]
                    self._shared['economy'].add_supply_depot(obs, self._shared,
                        Location(camera_loc=self._shared['camera'].location(obs, self._shared), screen_loc=building_center))
                else:
                    print("Can't find supply depot location")
                self._state = EconomyManager._IDLE

        elif self._state == EconomyManager._TRAINING_SCV:
            if not self._command_center_selected:
                on_screen_command_centers = self._shared['screen'].scan_units(obs, self._shared, [TERRAN_COMMAND_CENTER], PLAYER_SELF)

                if len(on_screen_command_centers) == 0:
                    selected_command_center = random.choice(self._shared['economy'].command_centers)
                    play['actions'] = [actions.FunctionCall(MOVE_CAMERA, [selected_command_center.camera.to_array()])]

                else:
                    selected_command_center = random.choice(on_screen_command_centers)
                    play['actions'] = [actions.FunctionCall(SELECT_POINT, 
                        [NEW_SELECTION, selected_command_center.location.screen.get_flipped().to_array()])]
                    self._command_center_selected = True     
            else:
                if TRAIN_SCV_QUICK in obs.observation["available_actions"]:
                    play['actions'] = [actions.FunctionCall(TRAIN_SCV_QUICK, [NOT_QUEUED])]
                    self._shared['economy'].add_scv()
                self._state = EconomyManager._IDLE

        if 'actions' not in play:
            self._state = EconomyManager._IDLE
            play['actions'] = [actions.FunctionCall(NO_OP, [])]
        
        if self._state != EconomyManager._IDLE:
            play['locked_choice'] = True
        
        return play

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
