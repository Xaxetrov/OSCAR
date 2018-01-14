import random
from oscar.agent.custom_agent import CustomAgent
from oscar.constants import *
from oscar.meta_action.build import *
from oscar.util.location import Location


class EconomyManager(CustomAgent):
    def __init__(self, message="I hate you"):
        self._message = message
        self.is_worker_selected = False
        self.building_center = None
        self.last_obs = None
        super().__init__()

    def supply_depot_built(self):
        print("built supply depot")
        self._shared['economy'].add_supply_depot(self.last_obs, self._shared,
            Location(camera_loc=self._shared['camera'].location(self.last_obs, self._shared), screen_loc=self.building_center))
        self.is_worker_selected = False

    def step(self, obs, locked_choice=None):
        play = {}
        play['actions'] = []

        if self._shared['env'].timestamp == 1 \
            and len(self._shared['economy'].command_centers) == 0:
            for u in self._shared['screen'].scan_units(obs, self._shared, [TERRAN_COMMAND_CENTER], PLAYER_SELF):
                self._shared['economy'].add_command_center(obs, self._shared,
                    Location(screen_loc=u.location.screen, camera_loc=self._shared['camera'].location(obs, self._shared)))

        if not self.is_worker_selected \
            and obs.observation["player"][IDLE_WORKER_COUNT] > 0:
            play['actions'] += [actions.FunctionCall(SELECT_IDLE_WORKER, [NEW_SELECTION])]
            play['locked_choice'] = True
            self.is_worker_selected = True
            return play

        if len(self._shared['economy'].supply_depots) < 2 \
            and obs.observation['player'][MINERALS] >= 100:

            if not self.is_worker_selected:
                """ if there is no idle scv, select one randomly """
                on_screen_scv = self._shared['screen'].scan_units(obs, self._shared, [TERRAN_SCV], PLAYER_SELF)
                selected_scv = random.choice(on_screen_scv)
                play['actions'] += [actions.FunctionCall(SELECT_POINT, 
                    [NEW_SELECTION, selected_scv.location.screen.get_flipped().to_array()])]

            self.building_center = get_random_building_point(obs, self._shared, 2 * TILES_SIZE_IN_CELL)
            if self.building_center:
                play['actions'] += [actions.FunctionCall(BUILD_SUPPLY_DEPOT, [NOT_QUEUED, self.building_center.get_flipped().to_array()])]
                play['success_callback'] = self.supply_depot_built
                self.last_obs = obs

        elif self.is_worker_selected:
            minerals = self._shared['screen'].scan_units(obs, self._shared, list(ALL_MINERAL_FIELD), PLAYER_NEUTRAL)
            if len(minerals) > 0:
                selected_mineral = random.choice(minerals)
                play['actions'] += [actions.FunctionCall(HARVEST_GATHER_SCREEN, 
                    [NOT_QUEUED, selected_mineral.location.screen.get_flipped().to_array()])]

        if len(play['actions']) == 0:
            if len(self._shared['economy'].command_centers) > 0:
                play['actions'] = [actions.FunctionCall(MOVE_CAMERA, 
                        [self._shared['economy'].command_centers[0].minimap.to_array()])]
            else:
                play['actions'] = [actions.FunctionCall(NO_OP, [])]

        return play


    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)