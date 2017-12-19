import gym
from gym import spaces

from pysc2.env.sc2_env import SC2Env
from pysc2.env import environment

from oscar.env.envs.pysc2_env import Pysc2Env
from oscar.agent.commander.general import General

DEFAULT_CONFIGURATION = "config/learning.json"


class Pysc2GeneralEnv(Pysc2Env):
    """
        Gym version of the pysc2 env using the OSCAR general/commander/agent structure
    """

    # action_space = spaces.Discrete(1)
    # observation_space = None  # set in init

    def __init__(self):
        self.pysc2_env = SC2Env(  # map_name='CollectMineralsAndGas',
            map_name='Simple64',
            agent_race='T',
            screen_size_px=(64, 64),
            minimap_size_px=(64, 64),
            visualize=True,
            step_mul=16,
            game_steps_per_episode=None  # use map default
        )
        self.general = General(DEFAULT_CONFIGURATION)
        action_spec = self.pysc2_env.action_spec()
        observation_spec = self.pysc2_env.observation_spec()
        self.general.setup(observation_spec, action_spec)
        # self.observation_space = self.general.training_memory.observation_space
        super().__init__()

    def _step(self, action):
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
        return super()._step([action])

    def _reset(self):
        self.general.reset()
        return super()._reset()

    def _render(self, mode='human', close=False):
        super()._render(mode, close)

    def _close(self):
        super()._close()

    def _seed(self, seed=None):
        super()._seed(seed)

    def get_action_mask(self):
        return None  # no action required here
