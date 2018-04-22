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


parser = argparse.ArgumentParser()
parser.add_argument('-fs', '--fit-step',
                    default=100000,
                    type=int,
                    dest="fit_step")
parser.add_argument('-o', '--out-dir',
                    default="ML_homework/results",
                    type=str,
                    dest="out_dir")
parser.add_argument('-lm', '--load-memory',
                    default="empty",  # or random or agent
                    type=str,
                    dest="load_memory")
parser.add_argument('-tau-m', '--tau-min',
                    default=1.0,
                    type=float,
                    dest="tau_min")
parser.add_argument('-tau-M', '--tau-max',
                    default=40.0,
                    type=float,
                    dest="tau_max")
parser.add_argument('-ds', '--decreasing-steps',
                    default=10000,
                    type=float,
                    dest="decreasing_steps")
parser.add_argument('-c', '--config-file',
                    default='config/learning_complex.json',
                    type=str,
                    dest="config_file")
args = parser.parse_args()

steps_warming_up = 10
if args.load_memory == "random":
    steps_warming_up = 50000
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

CONFIG_FILE = args.config_file
LOG_FILE = args.out_dir + '/duel_dqn_{}.csv'.format(CONFIG_FILE[7:-5])
MEMORY_FILE = 'ML_homework/memory_{}.pickle'.format(CONFIG_FILE[7:-4])


# Get the environment and extract the number of actions.
env = GeneralLearningEnv(CONFIG_FILE, False, log_file_path=LOG_FILE, publish_stats=False)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
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

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
boltzmann_policy = BoltzmannQPolicy(tau=20.0, clip=(0.0, 500.0))
policy = LinearAnnealedPolicy(boltzmann_policy,
                              'tau',
                              args.tau_min,
                              args.tau_max,
                              args.tau_min,
                              args.decreasing_steps)
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=steps_warming_up,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# load memory if possible
if args.load_memory == "agent":
    if not os.path.isfile(MEMORY_FILE):
        env.close()
        del env
        exit(0)
    with open(MEMORY_FILE, 'rb') as handle:
        memory_list = pickle.load(handle)
        for obs, a, r, done, training in memory_list:
            memory.append(obs, a, r, done, training)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=args.fit_step, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights(args.out_dir + '/duel_dqn_{}_weights.h5f'.format(CONFIG_FILE[7:-5]),
                 overwrite=True)

env.close()
del env

df = pd.DataFrame()
df = df.assign(c=args.config_file[7:-5])
df = df.assign(memory=args.load_memory)
df = df.assign(decreasing_steps=args.decreasing_steps)

df.to_csv(args.out_dir + '/args.csv', sep=',', mode='w')

# env = GeneralLearningEnv(CONFIG_FILE, True, publish_stats=False)
#
# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=False)
