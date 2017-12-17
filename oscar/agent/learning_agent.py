from oscar.agent.learning_structure import LearningStructure
from oscar.agent.custom_agent import CustomAgent


class LearningAgent(LearningStructure, CustomAgent):
    """
    An abstract class to build learning agent
    """

    def __init__(self, train_mode=False, shared_memory=None):
        LearningStructure.__init__(self, train_mode, shared_memory)



