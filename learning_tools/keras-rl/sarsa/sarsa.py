import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy

from oscar.env.envs.general_learning_env import GeneralLearningEnv

CONFIG_FILE = 'config/learning.json'
LOG_FILE = 'learning_tools/learning_nn/keras-rl/duel_dqn_{}_weights.csv'.format(CONFIG_FILE[7:-4])


# Get the environment and extract the number of actions.
env = GeneralLearningEnv(CONFIG_FILE, False, log_file_path=LOG_FILE)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
sarsa.fit(env, nb_steps=500000, visualize=False, verbose=2)

# After training is done, we save the final weights.
sarsa.save_weights('learning_tools/learning_nn/keras-rl/sarsa_{}_weights.h5f'.format(CONFIG_FILE[7:-4]),
                   overwrite=True)

env.close()
del env

# Finally, evaluate our algorithm for 5 episodes.
# sarsa.test(env, nb_episodes=5, visualize=True)
