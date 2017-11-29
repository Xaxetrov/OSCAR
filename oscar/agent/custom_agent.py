from constants import *

class CustomAgent(object):
    """A base agent to write custom scripted agents."""

    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self._default_play = {'actions': [actions.FunctionCall(NO_OP, [])],
                                'locked_choice': False,
                                'success_callback': self.default_callback,
                                'failure_callback': self.default_callback}

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def reset(self):
        self.episodes += 1

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward
        play = self._default_play
        return play

    def default_callback(self):
        pass
