# source of this code is from Jaromir Janisch, 2017
# and come from his blog: https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/

import numpy as np

import time

# local class
from learning_tools.A3C_learner.agent import Environment
from learning_tools.A3C_learner.brain import Optimizer, Brain
from learning_tools.A3C_learner.neuralmodel import save_neural_network
# custom env
from oscar.env import envs

# constants
from learning_tools.A3C_learner.constants import *

"""
Start by determining the shape of states and actions in a given environment.
This allows for fast switching of environments simply by changing a single constant (at least for simple environments).
"""
# set test env using meaning less value which will be over writen by future env deff
env_test = Environment(global_brain=None, num_actions=0, render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(shape=NUM_STATE)

"""Then create instances of Brain, Environment and Optimizer."""

env_test.env.close()
del env_test

brain = Brain(num_actions=NUM_ACTIONS,
              num_state=NUM_STATE,
              none_state=NONE_STATE)  # brain is global in A3C
brain.model.summary()

env_list = [Environment(global_brain=brain, num_actions=NUM_ACTIONS) for i in range(THREADS)]
opts = [Optimizer(brain=brain) for i in range(OPTIMIZERS)]

"""
Finally, it just starts the threads, wait given number of seconds, stops them and displays a trained agent to the user.
"""
for o in opts:
    o.start()

for e in env_list:
    e.start()

try:
    time.sleep(RUN_TIME)
except RuntimeError:
    print("trainning cancelled")
except KeyboardInterrupt:
    print("trainning cancelled")

try:
    print("Training time spend")

    for e in env_list:
        e.stop()
    for e in env_list:
        e.join()

    for o in opts:
        o.stop()
    for o in opts:
        o.join()
except RuntimeError:
    print("Thread stoping cancelled, saving NN and close")
except KeyboardInterrupt:
    print("Thread stoping cancelled, saving NN and close")

print("Training finished")
save_neural_network(brain.model)
print("model saved")
for e in env_list:
    e.env.close()
    del e

del env_list

# env_test.run()
