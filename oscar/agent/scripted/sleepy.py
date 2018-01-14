from oscar.constants import *


class NoOpAgent():
    def __init__(self):
        pass

    def step(self, _, locked_choice=None):
        return ([actions.FunctionCall(NO_OP, [])], )

