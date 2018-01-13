from oscar.agent.learning_structure import LearningStructure
from oscar.agent.custom_agent import CustomAgent


class LearningAgent(LearningStructure, CustomAgent):
    """
    An abstract class to build learning agent

    Sub class must implement the following LearningStructure methods:
      - _step
      - _format_observation
      - _transform_action
    (see LearningStructure code for more information on them)
    """

    def __init__(self, train_mode=False, shared_memory=None):
        """
        Constructor of the abstract class LearningAgent
        Sub class must keep the same calling format to work with the hierarchy factory
        self.observation_space and self.action_space must be set before calling this constructor
        :param train_mode: if the agent must train or play (default: False -> play)
        :param shared_memory: the memory used during training to communicate with the environment
                    (useless when playing, indispensable when training)
        """
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

