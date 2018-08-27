import numpy as np
import sys
import time

from oscar.constants import *
from oscar.util.location import Location
from oscar.util.coordinates_helper import Coordinates_helper
from oscar.util.exploration_helper import *
from oscar.meta_action import *

# States of the finite states automaton
class States:
    START = 0
    EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET = 1
    EXPLORATION__MOVE_UNITS_TO_TARGET = 2
    EXPLORATION__CENTER_CAMERA_ON_UNITS = 3
    ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY = 4
    ATTACK__CENTER_CAMERA_ON_UNITS = 5

class Find_and_defeat_zerglings():

    _NEW_TARGET_NB_TRY = 5

    def __init__(self):
        self.coordinates_helper = Coordinates_helper()
        self.state = States.START
        self.target = None

        self._loc_in_minimap = None
        self._ennemies_x = None
        self._ennemies_y = None
        self._units_x = None
        self._units_y = None
        self._units_mean_loc = None
        self._is_units_mean_loc_updated = False
        self._is_ennemies_visible = None
        self._target_reached = None

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass

    def step(self, obs):
        
        self._ennemies_x = None
        self._ennemies_y = None
        self._units_x = None
        self._units_y = None
        self._is_units_mean_loc_updated = False
        self._is_ennemies_visible = None
        self._target_reached = None

        # if some entities are selected
        if MOVE_SCREEN in obs.observation["available_actions"]:
            self.state = self._get_next_state(obs, self.state)
            print("state: " + str(self.state))
            time.sleep(0.2)
            return self._get_action(obs, self.state)
            
        elif _SELECT_ARMY in obs.observation["available_actions"]:
            return actions.FunctionCall(SELECT_ARMY, [SELECT_ALL])

        else:
            return actions.FunctionCall(_NO_OP, [])
     
    def _get_next_state(self, obs, curState):
        if curState == States.START:
            if self._is_ennemy_visible(obs):
                return States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY
            else:
                return States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET

        elif curState == States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET:
            return States.EXPLORATION__MOVE_UNITS_TO_TARGET

        elif curState == States.EXPLORATION__MOVE_UNITS_TO_TARGET:
            return States.EXPLORATION__CENTER_CAMERA_ON_UNITS

        elif curState == States.EXPLORATION__CENTER_CAMERA_ON_UNITS:
            if self._is_ennemy_visible(obs):
                return States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY
            else:
                if self._is_target_reached(obs) or not self._is_units_moving(obs):
                    return States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET
                else:
                    return States.EXPLORATION__CENTER_CAMERA_ON_UNITS

        elif curState == States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY:
            return States.ATTACK__CENTER_CAMERA_ON_UNITS

        elif curState == States.ATTACK__CENTER_CAMERA_ON_UNITS:
            if self._is_ennemy_visible(obs):
                return States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY
            else:
                return States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET

    def _get_action(self, obs, state):
        if state == States.EXPLORATION__MOVE_CAMERA_TO_NEW_TARGET:
            self.target = get_new_target(obs, self._get_loc_in_minimap(obs), self.coordinates_helper, self._NEW_TARGET_NB_TRY)
            return move_camera(self.target, self.coordinates_helper)[0]

        elif state == States.EXPLORATION__MOVE_UNITS_TO_TARGET:
            return actions.FunctionCall(ATTACK_SCREEN, [NOT_QUEUED, self.coordinates_helper.get_screen_center().to_array()])

        elif state == States.EXPLORATION__CENTER_CAMERA_ON_UNITS:
            self._loc_in_minimap = self._get_minimap_loc_centered_on_units(obs)
            return move_camera(self._loc_in_minimap, self.coordinates_helper)[0]

        elif state == States.ATTACK__MOVE_UNITS_TO_CLOSEST_ENNEMY:
            return actions.FunctionCall(ATTACK_SCREEN, [NOT_QUEUED, self._get_closest_ennemy(obs).to_array()])

        elif state == States.ATTACK__CENTER_CAMERA_ON_UNITS:
            self._loc_in_minimap = self._get_minimap_loc_centered_on_units(obs)
            return move_camera(self._loc_in_minimap, self.coordinates_helper)[0]

    def _get_loc_in_minimap(self, obs):
        if not self._loc_in_minimap:
            self._loc_in_minimap = self.coordinates_helper.get_loc_in_minimap(obs)
        return self._loc_in_minimap

    def _get_ennemies_locations(self, obs):
        if self._ennemies_x is None or self._ennemies_y is None:
            player_relative = obs.observation[SCREEN][SCREEN_PLAYER_RELATIVE]
            self._ennemies_y, self._ennemies_x = (player_relative == PLAYER_HOSTILE).nonzero()
        return self._ennemies_x, self._ennemies_y

    def _get_units_locations(self, obs):
        if self._units_x is None or not self._units_y is None:
            player_relative = obs.observation[SCREEN][SCREEN_PLAYER_RELATIVE]
            self._units_y, self._units_x = (player_relative == PLAYER_SELF).nonzero()
        return self._units_x, self._units_y

    def _get_units_mean_location(self, obs):
        if not self._is_units_mean_loc_updated:
            units_x, units_y = self._get_units_locations(obs) 
            if units_x.size > 0:
                newLoc = Location(int(units_x.mean()), int(units_y.mean()))
                if not self._units_mean_loc or not newLoc.equals(self._units_mean_loc):
                    self._units_mean_loc = newLoc
                    self._is_units_mean_loc_updated = True
        return self._units_mean_loc;

    def _is_units_moving(self, obs):
        self._get_units_mean_location(obs)
        return self._is_units_mean_loc_updated

    def _is_ennemy_visible(self, obs):
        if self._is_ennemies_visible is None:
            ennemies_x, ennemies_y = self._get_ennemies_locations(obs)
            self._is_ennemies_visible = ennemies_x.any()
        return self._is_ennemies_visible

    def _is_target_reached(self, obs):
        _TARGET_REACHED_SQUARED_DISTANCE = 25
        units_mean_loc = self._get_units_mean_location(obs)
        return (not self.target or \
                    units_mean_loc.squarred_distance(self.target) < _TARGET_REACHED_SQUARED_DISTANCE)

    def _get_closest_ennemy(self, obs):
        closest, min_dist = None, None
        ennemies_x, ennemies_y = self._get_ennemies_locations(obs)
        units_mean_loc = self._get_units_mean_location(obs)

        for p in zip(ennemies_x, ennemies_y):
            dist = np.linalg.norm(np.array(units_mean_loc.to_array()) - np.array(p))
            if not min_dist or dist < min_dist:
                closest, min_dist = p, dist
        return Location(closest[0], closest[1])

    def _get_minimap_loc_centered_on_units(self, obs):
        units_mean_loc = self._get_units_mean_location(obs)
        move = units_mean_loc.difference(self.coordinates_helper.get_screen_center())
        global_loc = self.coordinates_helper.minimap_to_global(self._get_loc_in_minimap(obs))
        centered_global_loc = global_loc.addition(move)
        centered_minimap_loc = self.coordinates_helper.global_to_minimap(centered_global_loc)
        return self.coordinates_helper.bound(centered_minimap_loc)