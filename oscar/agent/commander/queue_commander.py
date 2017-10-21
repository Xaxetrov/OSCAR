from oscar.agent.commander.commander import Commander
from oscar.agent.nn.tsp_agent import TSPAgent


class QueueCommander(Commander):

    def __init__(self):
        subordinates = []
        agent_one = TSPAgent()
        subordinates.append(agent_one)
        super().__init__(subordinates)
        self.__next_agent = 0

    def step(self, obs):
        super().step(obs)
        playing_subordinate = self._subordinates[self.__next_agent]
        self.__next_agent = (self.__next_agent + 1) % len(self._subordinates)
        return playing_subordinate.step(obs)
