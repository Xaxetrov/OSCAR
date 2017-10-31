import gym
from absl import flags

from pysc2.env.sc2_env import SC2Env
from pysc2.env import environment

# set empty argument to prevent pysc2 from crashing...
FLAGS = flags.FLAGS
FLAGS([""])


class Pysc2Env(gym.Env):
    """
        abstract class for building pysc2 env for Gym
    """
    metadata = {'render.modes': ['human']}

    # must be defined by sub class !
    pysc2_env = None

    last_obs = None

    def __init__(self):
        pass

    def _step(self, action):
        """
        move the environment forward of one step
        :param action: a pysc2 action
        :return: tuple of pysc2 full observation structure, the reward for the step, if is the last
            step or not and a dict to debug information (empty)
        """
        # Pysc2 can take a list of action ( https://github.com/deepmind/pysc2/blob/7a04e74effc88d3e2fe0e4562c99a18d06a099b2/pysc2/env/sc2_env.py#L247 )
        self.last_obs = self.pysc2_env.step(action)[0]
        done = self.last_obs.step_type == environment.StepType.LAST
        return self.last_obs, self.last_obs.reward, done, {}

    def _reset(self):
        self.last_obs = self.pysc2_env.reset()[0]
        return self.last_obs

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        self.pysc2_env.close()

    def _seed(self, seed=None):
        # any way to set the seed of pysc2 ?
        pass

    def get_action_mask(self):
        pass