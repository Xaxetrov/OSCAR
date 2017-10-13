from pysc2.agents import base_agent     # The base class for agent
from pysc2.lib import actions           # All actions available


class Agent(base_agent.BaseAgent):
    """A very basic agent (it does... nothing)"""

    def step(self, obs):
        """This is what our agent will do at each game step.
        In this case, it will simply do... Nothing :)"""

        # Call parent's method
        super(Agent, self).step(obs)

        # Somehow return a value
        # (this lets the game now it can proceed to the next step)
        #
        # Here, we return the value returned by the action of doing nothing
        # from the SC2 API.
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

