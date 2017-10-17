import numpy as np
import sys
import time
from random import randint

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from map_remembrance import MapRemembrance as Mem

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_MINI_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MINI_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_MINI_CAMERA = features.MINIMAP_FEATURES.camera.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [0]
_SCREEN = [0]

class FindAndDefeatZerglings():

    def __init__(self):
        self.memory = None
        # empirically determined values (bug ?) 
        # the observations of the camera are unconsistent with the action of the camera movement
        # more, the x and y axis are inverted in the observation
        self.cam_mov = {'min': {'x': 24, 'y':17}, 'max':{'x': 48, 'y': 47}}
        self.camera_observation_offset = 12

    def setup(self, obs_spec, action_spec):
        pass

    def reset(self):
        pass

    def step(self, obs):
        
        if self.memory is None:
            self.memory = Mem(obs.observation["minimap"][_MINI_PLAYER_RELATIVE].shape)
                
        self.memory.maj_informations(obs.observation["minimap"][_MINI_PLAYER_RELATIVE],
                                     obs.observation["minimap"][_MINI_VISIBILITY])
            
        # if some entities are selected
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            
            # Find our units and targets
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            ennemies_y, ennemies_x = (player_relative == _PLAYER_HOSTILE).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

            # if no observations, move camera and marines
            if not ennemies_y.any() or not player_y.any():
                seen_ennemies = self.memory.get_most_recent_informations(0, _PLAYER_HOSTILE)[0]
                if len(seen_ennemies) > 0:
                    # definition of the camera informations
                    cam_pos_x, cam_pos_y = obs.observation["minimap"][_MINI_CAMERA].nonzero()
                    cam_pos = np.asarray([cam_pos_y.min(), cam_pos_x.min()])
                    cam_size = np.asarray([cam_pos_y.max(), cam_pos_x.max()]) - cam_pos
                    # correction of the offset induced by the unconsistency so it matches other observations
                    cam_pos += self.camera_observation_offset
                    
                    # determination of the new camera position
                    new_cam_pos = seen_ennemies[0] - cam_size/2
                    new_cam_pos = np.asarray([min(max(new_cam_pos[0], self.cam_mov['max']['y']), self.cam_mov['min']['y']),
                                              min(max(new_cam_pos[1], self.cam_mov['max']['x']), self.cam_mov['min']['x'])])
                    dist = np.linalg.norm(cam_pos - new_cam_pos)
                    print(dist)
                    # move camera if it is not done, otherwise move marines
                    if dist > 1:
                        return actions.FunctionCall(_MOVE_CAMERA, [new_cam_pos])
                    else:
                        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [42, 42]])
                    
                return actions.FunctionCall(_NO_OP, [])

            player = [int(player_x.mean()), int(player_y.mean())]
            
            # compute closest ennemy
            closest, min_dist = None, None
            for p in zip(ennemies_x, ennemies_y):
                dist = np.linalg.norm(np.array(player) - np.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist

            time.sleep(0.4)
            return actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, closest])
            
        else:
            return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])

        