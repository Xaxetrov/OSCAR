# Reinforcement learning

Instead of scripting an agent behavior, this one could be learnt automatically while playing the game.
Before being added to the main structure, an agent can be trained separately in a specific environment, for instance, using a general and the agent alone and with a mini-game instead of the entire Starcraft II. Each RL agent could define its own observation space (using raw data from screen or pre-computed features), action space and reward function.

Hence, we have created an OpenAI's gym environment compatible with our architecture. This allows to use some learning algorithms from the Web without having to adapt them for the Starcraft game.

### How to create an Gym environment ?

To use the custom made environment, the following package should be imported first:

    import oscar.env`
    
It can then be used as any other Gym environment:

    gym.make("general-learning-v0")

The environment is created using the standard configuration file in [oscar/env/envs/pysc2_general_env.py](https://github.com/Xaxetrov/OSCAR/blob/master/oscar/env/envs/pysc2_general_env.py). This is obviously not a good way to do it, but it works (see [TODO](https://github.com/Xaxetrov/OSCAR#todo)).

### What is the observation and action space in this environment ?

The observation / action space of the "general-learning-v0" environment could be set for each agent independently. This could be the PySc2's observations and actions or custom features and actions defined by the user.

[oscar/agent/nn/meta_action_perceptron_agent.py](https://github.com/Xaxetrov/OSCAR/blob/master/oscar/agent/nn/meta_action_perceptron_agent.py) is a simple example of a learning agent using some pre-computed features and some custom actions.