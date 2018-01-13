from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action import *
from oscar.util.coordinates_helper import Coordinates_helper
import time

"""
Positions units at strategic locations and moves camera
in order to collect information on the enemy.
"""
class Spy(CustomAgent):
    def __init__(self, message="I hate you"):
        self._message = message
        self.coordinates_helper = Coordinates_helper()
        self.cur_location = None
        self.camera_moved = False
        super().__init__()

    def spy_sent(self):
        print("spy sent")

    def step(self, obs):

        if not self.cur_location:
            self.cur_location = self.coordinates_helper.get_loc_in_minimap(obs)

        play = {}

        if self.camera_moved:
            self._shared_objects["units_tracker"].scan_screen(obs, self.cur_location)
            play["actions"] = [actions.FunctionCall(NO_OP, [])]
            self.camera_moved = False
        else:
            target = get_spy_target(
                obs, 
                self.cur_location, 
                self._shared_objects["units_tracker"], 
                self.coordinates_helper
            )

            if target:
                if self.is_location_visible(obs, target):
                    play["actions"] = move_camera(target, self.coordinates_helper)
                    play["locked_choice"] = True
                    self.cur_location = target
                    self.camera_moved = True
                else:
                    try:
                        play["actions"] = self.select_unit(obs)
                        minimap_view_center_offset = Location(
                            0.5 * self.coordinates_helper.field_of_view_minimap['x'],
                            0.5 * self.coordinates_helper.field_of_view_minimap['y']
                        )
                        play["actions"].append(
                            actions.FunctionCall(MOVE_MINIMAP, 
                                [NOT_QUEUED, target.addition(minimap_view_center_offset).to_array()]))
                        play["success_callback"] = self.spy_sent
                    except NoValidSCVError:
                        play["actions"] = [actions.FunctionCall(NO_OP, [])]
            else:
                play["actions"] = [actions.FunctionCall(NO_OP, [])]

        return play


    def select_unit(self, obs):
        selection = None
        """try:
            selection = select_idle_scv_screen_priority(obs)
        except NoValidSCVError:"""
        try:
            selection = select_scv_on_screen(obs)
        except NoValidSCVError:
            raise
        return selection


    def is_location_visible(self, obs, location):
        _MIN_VISIBLE_PROPORTION = 0.5

        visibles, non_visibles = 0, 0
        for x in range(location.x, location.x + self.coordinates_helper.field_of_view_minimap['x'] - 1):
            for y in range(location.y, location.y + self.coordinates_helper.field_of_view_minimap['y'] - 1):
                if obs.observation['minimap'][MINI_VISIBILITY][y, x] == VISIBLE_CELL:
                    visibles += 1
                else:
                    non_visibles += 1

        return (visibles/(visibles+non_visibles) >= _MIN_VISIBLE_PROPORTION)



    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
