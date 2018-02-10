# OSCAR

OSCAR (*Omniscient Starcraft Awesome Retaliator*) is an autonomous team project conducted during a semester at INSA Lyon, France. It aims at experimenting artificial intelligence for Starcraft II, using the [PySc2](https://github.com/deepmind/pysc2) API from DeepMind.    
The project consists in:
* a hierarchical framework to unify some small specialized agent ([read more](https://github.com/Xaxetrov/OSCAR/blob/master/docs/hierarchical_framework.md))
* a scripted multi-agent bot playing the full game ([read more](https://github.com/Xaxetrov/OSCAR/blob/master/docs/scripted_bot.md))
* some tools and agents to experiment reinforcement learning ([read more](https://github.com/Xaxetrov/OSCAR/blob/master/docs/reinforcement_learning.md))

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


## TODO

* Fix neural network save in DQL algorithm [learning_tools/baseline/dqn/custom_learn.py](https://github.com/Xaxetrov/OSCAR/blob/master/learning_tools/baseline/dqn/custom_learn.py)
* Allow to generate the "general-learning-v0" environment with the config file as a parameter (and the map ?) ((this will not work when using the  `gym.make` command but nothing forbids us to use our own constructor))


## Authors

Bruno Godefroy

Edern Haumont

Jérome Liermann

Ruben Pericas-Moya

François Robion

Nicolas Six
