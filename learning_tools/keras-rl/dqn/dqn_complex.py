import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from oscar.env.envs.general_learning_env import GeneralLearningEnv
from ML_homework.policy_iteration.complex_policy_iteration import policy_iteration_iterator, state_from_obs

CONFIG_FILE = 'config/learning_complex.json'
LOG_FILE = 'learning_tools/learning_nn/keras-rl/duel_dqn_{}.csv'.format(CONFIG_FILE[7:-5])


# Get the environment and extract the number of actions.
env = GeneralLearningEnv(CONFIG_FILE, False, log_file_path=None, publish_stats=False)
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
policy = LinearAnnealedPolicy(boltzmann_policy, 'tau', 1.0, 20.0, 1.0, 10000)
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# warm up
pi = None
for p in policy_iteration_iterator(10, 0.5, file_path="/tmp/state_table.csv", save_path="/tmp/OSCAR/"):
    pi = p

for i in range(20):
    obs = env.reset()
    while True:
        s = state_from_obs(obs)
        a = pi[s.id()]
        old_obs = obs
        obs, r, done, debug_dict = env.step(a)
        memory.append(old_obs, a, r, done, False)
        if done:
            break

env.close()
env = GeneralLearningEnv(CONFIG_FILE, False, log_file_path=LOG_FILE, publish_stats=False)

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=500000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('learning_tools/learning_nn/keras-rl/duel_dqn_{}_weights.h5f'.format(CONFIG_FILE[7:-5]),
                 overwrite=True)

env.close()
del env
# env = GeneralLearningEnv(CONFIG_FILE, True, publish_stats=False)
#
# # Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=False)
