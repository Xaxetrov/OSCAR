import numpy as np
import pickle

from oscar.env.envs.general_learning_env import GeneralLearningEnv
from ML_homework.policy_iteration.basic_policy_iteration import policy_iteration

CONFIG_FILE = 'config/learning.json'
MEMORY_FILE = 'ML_homework/memory_{}.pickle'.format(CONFIG_FILE[7:-4])

env = GeneralLearningEnv(CONFIG_FILE, False, log_file_path=None, publish_stats=False)
np.random.seed(123)
env.seed(123)

# warm up
pi = policy_iteration(0.5)

memory = []
obs_to_s = np.array([-1, 2, 8, 4, 16], dtype=np.int)

while len(memory) < 50000:
    print(len(memory))
    obs = env.reset()
    while True:
        s = int(np.sum(obs * obs_to_s) + 1)
        a = pi[s]
        old_obs = obs
        obs, r, done, debug_dict = env.step(a)
        memory.append((old_obs.copy(), a, r, done, False))
        if done:
            break

with open(MEMORY_FILE, mode='wb') as handle:
    pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

env.close()
del env
