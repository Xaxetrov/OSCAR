import numpy as np
import pickle

from oscar.env.envs.general_learning_env import GeneralLearningEnv
from ML_homework.policy_iteration.complex_policy_iteration import policy_iteration_iterator, state_from_obs

CONFIG_FILE = 'config/learning_complex.json'
MEMORY_FILE = 'ML_homework/memory_{}.pickle'.format(CONFIG_FILE[7:-4])

env = GeneralLearningEnv(CONFIG_FILE, False, log_file_path=None, publish_stats=False)
np.random.seed(123)
env.seed(123)

# warm up
pi = None
for p in policy_iteration_iterator(10, 0.5, file_path="/tmp/state_table.csv", save_path="/tmp/OSCAR/"):
    pi = p

memory = []

while len(memory) < 50000:
    print(len(memory))
    obs = env.reset()
    while True:
        s = state_from_obs(obs)
        a = pi[s.id()]
        old_obs = obs
        obs, r, done, debug_dict = env.step(a)
        memory.append((old_obs.copy(), a, r, done, False))
        if done:
            break

with open(MEMORY_FILE, mode='wb') as handle:
    pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

env.close()
del env

