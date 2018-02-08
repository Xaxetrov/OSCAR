# OSCAR

OSCAR (*Omniscient Starcraft Awesome Retaliator*) is an autonomous team project conducted during a semester at INSA Lyon, France. It aims at experimenting artificial intelligence for Starcraft II, using the [PySc2](https://github.com/deepmind/pysc2) API from DeepMind.    
The project consists in:
* some small scripted agents to deal with subtasks of the game (scouting, managing economy, ...)
* a hierarchical framework to unify some small specialized agents into a bot capable to play the full game
* some tools and agents to experiment reinforcement learning

## Installation (Unix)
The project requires Python 3.5.

`git clone https://github.com/Xaxetrov/OSCAR.git`   
`pip3 install pysc2`    
`pip3 install gym`    
`pip3 install baselines`   

## Run (Unix)
`cd OSCAR`    
Add current directory to Python path: `export PYTHONPATH=$(pwd)`

#### Scripted bot
The General agent should be launched with PySc2. For example,    
`python -m pysc2.bin.agent --agent_race T --bot_race T --difficulty 1 --map Flat64 --agent oscar.agent.commander.general.General`

#### Deep Q-Network
training: `python learning_tools/baseline/dqn/custom_learn.py`    
test: `python learning_tools/baseline/dqn/custom_test.py`    

#### Asynchronous Actor-Critic Agents (A3C)
training: `python learning_tools/A3C_learner/main.py`    
test: `python learning_tools/A3C_test/a3c_tester.py`

## Architecture
There are three different basic bricks in our architecture: a general, 0 to many commanders and at least one agent.
 - The general is at the top of the hierarchy. It provides an interface between the used library pysc2 and our application. It has a unique child, either an agent or a commander
 - A commander is a specialized agent that can have several children, all of them being an agent. Its role is to choose which of its children it will call. As a commander is also an agent, it can have a commander as a child.
 - An agent is a leaf of the hierarchy tree. When called, it will return an action to be executed. It can not have any child. As the interfaces between an agent and its commander are specified, the agent can either be trained or scripted.
 
The process to choose which action to execute is three phases:
 - First, the general asks its child to play. If it is a commander, it will recursively ask one of its child to play, until an agent is chosen.
 - Then, the agent will return the action to be executed to its commandant, and then to the general.
 - Finally, the general can call a callback, depending of the result (success or error) of the action.
 
### Return format
An agent will return a dictionary with four items to its commander:
 - "actions": The list of actions to be executed in a row by the general. If an error occurs before that the list is empty, the general will empty the list and call the error's callback (if provided). Else, it will call the success' callback (if provided). This item is the only that is mandatory.
 - "success_callback": The callback that will be called if all actions were done without any error. If omitted, no callback will be called.
 - "failure_callback": The callback that will be called if an error occurs when the actions were executed. If omitted, no callback will be called.
 - "locked_choice": A boolean to ensure that the same agent will be called once the current list of actions is empty. Default to False if omitted.
 
### Configuration file
The architecture was designed to be flexible. In order to improve that, a configuration file was created describing what are the agents and commanders involved, and how are they organized. Here is a basic example of such a file.
    
    {
        "structure": {
            0: [1, 2],
            1: [3, 4, 5]
        },
        "agents": [
            {
                "id": 0,
                "class_name": "oscar.agent.scripted.strategy_manager.StrategyManager",
                "arguments": {}
            },
            ...
        ]
    }
    
The key "structure" described which are the children of each commander. Here, agents whose id is 1 and 2 are the children of commander whose id is 0, and agents whose id are 3, 4 and 5 are children of commander whose id is 1.

The key "agents" provide a description of each agent mentioned previously, here from 0 to 5. This description include the path to the class, as well as optional arguments.
## Training

An agent can be train in a specific environment before to be added to the main structure. For example you can train an agent using only a general and it on a mini-game, but in that case the agent will not have learn how to interact with the others agents.

That's why we also created a OpenAI's gym compatible environment of our architecture, which allow you to run the entire architecture but one agent and in addition to use start training algorithm that you can find on the web. The reward as well as the input and output space are defined in the agent. So the architecture can made of agents choosing action from screen directly or agent using only precomputed indicators.

### How to create an Gym environment ?

To use the custom made environment you need to import the following package:

    import oscar.env

This will add to the gym standards environment the environment build for OSCAR.
Now you can create the environment using the standard `gym.make()` command.

    gym.make("general-learning-v0")
    
And use the environment as any other Gym environment.

The environment is created using the standard configuration file set into the [oscar/env/envs/pysc2_general_env.py](https://github.com/Xaxetrov/OSCAR/blob/master/oscar/env/envs/pysc2_general_env.py). This is obviously not a good way to do it, but work (see TODO).

### What are the action and observation space on this environment ?

The action / observation space of the "general-learning-v0" environment only depend on the agent trained (and so of the configuration file used).

Thus an agent can define his own observation and action space from the pysc2's observation structure and pysc2's actions or meta actions respectively.

[oscar/agent/nn/meta_action_perceptron_agent.py](https://github.com/Xaxetrov/OSCAR/blob/master/oscar/agent/nn/meta_action_perceptron_agent.py) is a basic example of how to do it.


## TODO

* Manage context switch for the transition from an agent to the other
* Fix neural network save in DQL algorithm [learning_tools/baseline/dqn/custom_learn.py](https://github.com/Xaxetrov/OSCAR/blob/master/learning_tools/baseline/dqn/custom_learn.py)
* Allow to generate the "general-learning-v0" with as parameter the config file to use (and the map ?) ((this will not work when using the  `gym.make` command but nothing forbid us to use our own constructor))
* Improve Readme:
    * Complete the run section
    * schema of the commanding structure
    * scripted agent ?
    * specific utils ?

## Authors

Bruno Godefroy

Edern Haumont

Jérome Liermann

Ruben Pericas-Moya

François Robion

Nicolas Six
