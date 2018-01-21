from oscar.agent.commander.base_commander import BaseCommander
from oscar.agent.learning_structure import LearningStructure


class LearningCommander(BaseCommander, LearningStructure):
    """
    A special commander that is able to learn
    Double inheritance class : BaseCommander and LearningStructure
    """

    def __init__(self, subordinate, train_mode=False, shared_memory=None):
        # init base commander mother class
        BaseCommander.__init__(self, subordinates=subordinate)

        # action space is the size of the subordinate list
        self.action_space = len(self._subordinates)
        # init learning structure mother class
        LearningStructure.__init__(self, train_mode, shared_memory)

        # in this case do step apply to the choose subordinate method
        self.choose_subordinate = self.do_step

    def add_subordinate(self, agent):
        raise RuntimeError("Learning commander cannot add subordinate dynamically")

    def remove_subordinate(self, agent):
        raise RuntimeError("Learning commander cannot remove subordinate dynamically")
