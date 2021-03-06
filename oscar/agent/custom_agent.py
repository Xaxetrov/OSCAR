from pysc2.agents.base_agent import BaseAgent


class CustomAgent(BaseAgent):
    """A base agent to write custom scripted agents."""

    def __init__(self):
        super().__init__()
        self._shared = {}

    def add_shared(self, name, shared):
        self._shared[name] = shared