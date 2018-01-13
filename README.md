# OSCAR

OSCAR is a school project we had during our 5th year at INSA Lyon.

The goal of this project is to discover Machine Learning by exploring the construction of an AI for the StarCraft II game using the pysc2 API published by DeepMind.

## Architecture

The AI is build around a classical commender / agent architecture. Both of them are agents but the commender cannot choose an action by himself and only choose to which agent he delegate the task (this delegate agent can also be a commender) On top of that we added what we called a general, his goal is to convert our calling convention into the ones of the pysc2 api.

Some of the agent are trained others are only scripted.

## Training

An agent can be train in a specific environment before to be added to the main structure. For example you can train an agent using only a general and it on a mini-game, but in that case the agent will not have learn how to interact with the others agents.

That's why we also created a OpenAI's gym compatible environment of our architecture, which allow you to run the entire architecture but one agent and in addition to use start training algorithm that you can find on the web. The reward as well as the input and output space are defined in the agent. So the architecture can made of agents choosing action from screen directly or agent using only precomputed indicators.

### How to create an Gym environment ?

To use the custom made environement you need to import the following package:

    import oscar.env

This will add to the gym standards environment the environment build for OSCAR.
Now you can create the environment using the standard `gym.make()` command.

    gym.make("general-learning-v0")
    
And use the environment as any other Gym environment.

The environment is created using the standard configuration file set into the [oscar/env/envs/pysc2_general_env.py](https://github.com/Xaxetrov/OSCAR/blob/master/oscar/env/envs/pysc2_general_env.py). This is obviously not a good way to do it, but work (see TODO).

### What are the action and observation space on this environment ?

The action / observation space of the "general-learning-v0" environement only depand on the agent trained (and so of the configuration file used).

Thus an agent can define his own observation and action space from the pysc2's observation structure and pysc2's actions or meta actions respectively.

[oscar/agent/nn/meta_action_perceptron_agent.py](https://github.com/Xaxetrov/OSCAR/blob/master/oscar/agent/nn/meta_action_perceptron_agent.py) is a basic example of how to do it.


## TODO

* Manage context switch for the transition from an agent to the other
* Fix neural network save in DQL algorithm [learning_tools/baseline/dqn/custom_learn.py](https://github.com/Xaxetrov/OSCAR/blob/master/learning_tools/baseline/dqn/custom_learn.py)
* Allow to generate the "general-learning-v0" with as parameter the config file to use (and the map ?) ((this will not work when using the  `gym.make` command but nothing forgive us to use our own constructor))
* Improve Readme:
    * configuration file
    * meta action
    * schema of the commanding structure
    * scripted agent ?
    * specific utils ?


