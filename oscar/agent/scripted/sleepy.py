from oscar.constants import *


class NoOpAgent():
    def __init__(self):
        pass

    def step(self, _):
        return ([actions.FunctionCall(NO_OP, [])], )

