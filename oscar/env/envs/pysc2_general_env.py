import gym
from gym import spaces

from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Race, Difficulty, Agent, Bot, Dimensions
from pysc2.env import environment

from oscar.env.envs.pysc2_env import Pysc2Env
from oscar.agent.commander.general import General
import numpy as np
from oscar.constants import *
import warnings

DEFAULT_CONFIGURATION = "config/learning.json"


class Pysc2GeneralEnv(Pysc2Env):
    """
        Gym version of the pysc2 env using the OSCAR general/commander/agent structure
    """

    # action_space = spaces.Discrete(1)
    # observation_space = None  # set in init

    def __init__(self,
                 path_to_configuration=DEFAULT_CONFIGURATION,
                 enable_visualisation=True,
                 game_step_per_update=8):
        self.pysc2_env = SC2Env(  # map_name='CollectMineralsAndGas',
            map_name='Simple64',
            players=[Agent(Race.terran),
                     Bot(Race.random, Difficulty.very_easy)],
            agent_interface_format=[AgentInterfaceFormat(feature_dimensions=Dimensions(screen=(SCREEN_RESOLUTION,
                                                                                               SCREEN_RESOLUTION),
                                                                                       minimap=(MINIMAP_RESOLUTION,
                                                                                                MINIMAP_RESOLUTION)),
                                                         camera_width_world_units=TILES_VISIBLE_ON_SCREEN_WIDTH)],
            # git version give camera position in observation if asked
            visualize=enable_visualisation,
            step_mul=game_step_per_update,
            game_steps_per_episode=None  # use map default
        )
        self.general = General(path_to_configuration)
        action_spec = self.pysc2_env.action_spec()
        observation_spec = self.pysc2_env.observation_spec()
        self.general.setup(observation_spec, action_spec)
        # self.observation_space = self.general.training_memory.observation_space
        super().__init__()

    def step(self, action):
        """
        This beautiful environment as a good sens of hierarchy.
        He has a general and his general is better than you, and so the action you give
        is pointless and general will choose what is a good idea (when you only have one
        choice its easier for you, no ?).
        :param action: an int, if you want but None is fine too (dict, tuple and object
         are accepted but not recommended)
        :return: a tuple with the observation, reward, done and an empty dict
        """
        action = self.general.step(self.last_obs)
        obs, reward, done, debug_dict = super().step([action])
        # explore observation to decide if agent can still play
        # first condition is to check if a command center exist
        # second is to check if the agent has unit or minerals to create one
        if obs.observation['player'][FOOD_CAP] < 15 \
                or (obs.observation['player'][FOOD_USED] == 0 and obs.observation['player'][MINERALS] < 50):
            warnings.warn("Environment decided that game is lost")
            done = True
        return obs, reward, done, debug_dict

    def reset(self):
        self.general.reset()
        return super().reset()

    def render(self, mode='human', close=False):
        super().render(mode, close)

    def close(self):
        super().close()

    def seed(self, seed=None):
        super().seed(seed)

    def get_action_mask(self):
        return None  # no action required here
