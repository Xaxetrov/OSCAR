from oscar.agent.commander.queue_commander import QueueCommander
from oscar.constants import MINI_VISIBILITY, MINI_PLAYER_RELATIVE


class NotBlindQueueCommander(QueueCommander):

    def step(self, obs):
        self._shared_objects["ennemies"].maj_informations(obs.observation["minimap"][MINI_PLAYER_RELATIVE], obs.observation["minimap"][MINI_VISIBILITY])
        return super().step(obs)
