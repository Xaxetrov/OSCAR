from pysc2.agents import base_agent     # The base class for agent
from pysc2.lib import actions           # All actions available
from pysc2.lib import features          # Features types

from src.tools import debugger


class Agent(base_agent.BaseAgent, debugger.Debugger):
    """A very basic agent that just move unit to beacon"""

    def step(self, obs):
        """This is what our agent will do at each game step.
        In this case, try to reach a beacon."""

        # Call parent's method
        super(Agent, self).step(obs)

        # So we have:
        #   1 marine (id: 0)
        #   1 beacon

        # Print some values and sleep a bit
        # self.debug(obs, 10)

        # First, we have to select our marine
        if actions.FUNCTIONS.Move_screen.id not in obs.observation["available_actions"]:
            # We can't move, because we have not selected any unit!
            # Let's select one (the only one we have)
            return actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])

        # Once our marine is selected,
        # we have to locate the beacon and move the unit to it
        else:
            # So here the beacon is the only neutral thing on the map
            # They are represented by the number 3 in the observations from the screen
            things = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]

            # Filter the neutral cases where there is a piece of neutral thing
            neutral_y, neutral_x = (things == 3).nonzero()  # array of [array of y, array of x]

            # Determinate the beacon's center
            beacon = [int(neutral_x.mean()), int(neutral_y.mean())]

            # Move to the beacon
            # (the first param means we do not queue the action)
            return actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], beacon])
