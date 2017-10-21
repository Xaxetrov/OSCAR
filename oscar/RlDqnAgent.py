# from __future__ import division
import argparse
from absl import flags
import sys

import numpy as np
import gym

from keras.optimizers import Adam
from keras.models import save_model

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from oscar.nnModels.DenseNeuralModel import get_neural_network
# import oscar_env
from oscar_env.envs import pysc2_mineralshard_env

# flags are not used but prevent pysc2 from writing error log when looking for a flag
FLAGS = flags.FLAGS
FLAGS(sys.argv)

WINDOW_LENGTH = 1
ENV_NAME = 'pysc2-mineralshard-v0'


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default=ENV_NAME)
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

mode = 'train'
# mode = 'test'

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
# env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).

model = get_neural_network()
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=20000, # original example value was 1 000 000 but 80 000 take more than 5Go of RAM
                          window_length=WINDOW_LENGTH
                          )

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=1.,
                              value_min=.01,
                              value_test=.05,
                              nb_steps=100000
                              )

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               policy=policy,
               memory=memory,
               nb_steps_warmup=5000,
               gamma=.99,
               target_model_update=1000,
               train_interval=4,
               delta_clip=1.
               )
dqn.compile(Adam(lr=.00025),
            metrics=['mae']
            )

if mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    # callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    # callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env,
            # callbacks=callbacks,
            nb_steps=175000,
            log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)
    save_model(dqn.model, "Neuralnetwork/DenseMineralShard.knn")

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif mode == 'test':
    # weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    # if args.weights:
    #     weights_filename = args.weights
    # dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
    env.close()
