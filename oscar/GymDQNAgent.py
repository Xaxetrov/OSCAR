import sys
import gflags as flags
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.models import save_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from oscar.nnModels.DenseNeuralModel import get_neural_network
import oscar_env

# flags are not used but prevent pysc2 from writing error log when looking for a flag
FLAGS = flags.FLAGS
FLAGS(sys.argv)
ENV_NAME = 'pysc2-mineralshard-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = get_neural_network()
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=10,
               target_model_update=1e-2,
               policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env,
        nb_steps=1000,
        visualize=False,
        verbose=1,
        action_repetition=10,
        nb_max_start_steps=50 # Do random action during the first 50 steps
        )

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
save_model(dqn.model, "DenseMineralShard.knn")

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
