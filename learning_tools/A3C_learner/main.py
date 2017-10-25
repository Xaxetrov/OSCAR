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
from learning_tools.A3C_learner.config import *

# set test env using meaning less value which will be over writen by future env deff
env_test = Environment(global_brain=None, num_action=0, render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(shape=NUM_STATE)

brain = Brain(num_actions=NUM_ACTIONS,
              num_state=NUM_STATE,
              none_state=NONE_STATE)  # brain is global in A3C
brain.model.summary()

envs = [Environment(global_brain=brain, num_action=NUM_ACTIONS) for i in range(THREADS)]
opts = [Optimizer(brain=brain) for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
save_neural_network(brain.model)
for e in envs:
    e.env.close()

env_test.run()
