import numpy as np
import gym
import os
import pickle
import argparse
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from oscar.env.envs.general_learning_env import GeneralLearningEnv

CONFIG_FILE = 'config/learning_complex.json'
WEIGHT_FILE = 'ML_homework/results/2018-04-22_16/duel_dqn_learning_complex_weights.h5f'

# Get the environment and extract the number of actions.
env = GeneralLearningEnv(CONFIG_FILE, True, log_file_path=None, publish_stats=False)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
boltzmann_policy = BoltzmannQPolicy(tau=1.0, clip=(0.0, 500.0))
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, policy=boltzmann_policy,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights(WEIGHT_FILE)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=1, visualize=False)

env.close()
del env
