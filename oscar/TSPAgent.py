from oscar.DQNAgent import DQNAgent, InputStructure, OutputStructure


class TSPAgent(DQNAgent):
    def __init__(self):
        input = InputStructure(screen_size=64, screen_number=2, non_spatial_features=0)
        output = OutputStructure(spatial_action_size=64, non_spatial_action_size=1)
        super().__init__(input, output)
