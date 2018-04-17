import numpy as np
import gym
import time
import pandas as pd
import os

from oscar.env.envs.general_learning_env import GeneralLearningEnv
from ML_homework.value_iteration.generate_transition import generate_transition_basic_env


RESULT_FILE = "ML_homework/value_iteration/basic.csv"
NUMBER_OF_TEST = 1


def value_iteration(gamma, max_iter, delta):
    r, p = generate_transition_basic_env()

    u = np.random.normal(0.0, 0.5, size=r.shape[0])

    for i in range(max_iter):
        previous_u = u.copy()
        q = np.einsum('ijk,ijk -> ij', p, r + gamma * u)
        u = np.max(q, axis=1)

        if np.max(np.abs(u - previous_u)) < delta:
            break

    policy = np.argmax(q, axis=1)
    return u, policy, i + 1


def value_iteration_iterator(gamma, max_iter):
    r, p = generate_transition_basic_env()

    u = np.random.normal(0.0, 0.5, size=r.shape[0])

    for i in range(max_iter):
        previous_u = u.copy()
        q = np.einsum('ijk,ijk -> ij', p, r + gamma * u)
        u = np.max(q, axis=1)

        policy = np.argmax(q, axis=1)
        yield policy, np.max(np.abs(u - previous_u))


if __name__ == '__main__':
    env = GeneralLearningEnv("config/learning.json", True)

    done = False
    obs = env.reset()
    first = True

    obs_to_s = np.array([-1, 2, 8, 4, 16], dtype=np.int)

    for p, error in value_iteration_iterator(0.1, 10):
        for i in range(NUMBER_OF_TEST):
            while not done:
                s = int(np.sum(obs * obs_to_s) + 1)
                a = p[s]
                obs, _, done, debug_dict = env.step(a)
            obs = env.reset()
            done = False
            df = debug_dict['stats']
            df = df.assign(value_change=[error])
            if os.path.isfile(RESULT_FILE) or first:
                df.to_csv(RESULT_FILE, sep=',', mode='w', header=True)
                first = False
            else:
                df.to_csv(RESULT_FILE, sep=',', mode='a', header=False)

    env.close()
    del env

