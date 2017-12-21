from oscar.agent.learning_structure import LearningStructure
from oscar.agent.custom_agent import CustomAgent


class LearningAgent(LearningStructure, CustomAgent):
    """
    An abstract class to build learning agent
    """

    def __init__(self, train_mode=False, shared_memory=None):
        LearningStructure.__init__(self, train_mode, shared_memory)
        self.failed_meta_action_counter = 0

    def _learning_step(self, obs):
        result = super()._learning_step(obs)
        result["failure_callback"] = self.failure_callback
        return result

    def failure_callback(self):
        self.failed_meta_action_counter += 1

    def reset(self):
        super().reset()
        print("Failed meta action :", self.failed_meta_action_counter)
        self.failed_meta_action_counter = 0

