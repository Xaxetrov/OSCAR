import numpy as np
import time
import pandas as pd
import os

from oscar.env.envs.general_learning_env import GeneralLearningEnv
from ML_homework.value_iteration.generate_transition import generate_transition_basic_env

RESULT_FILE = "ML_homework/policy_iteration/basic.csv"
NUMBER_OF_TEST = 1


def policy_iteration(gamma=0.99, max_iterations=10**6, delta=10**-3):
    r, p = generate_transition_basic_env()

    # initialize with a random policy and initial value function
    policy = np.random.randint(0, r.shape[1], size=r.shape[0])
    u = np.zeros(r.shape[0])

    # iterate and improve policies
    for i in range(max_iterations):
        previous_policy = policy.copy()

        for j in range(max_iterations):
            previous_u = u.copy()
            u = np.zeros(shape=(r.shape[0]))
            for s in range(r.shape[0]):
                a = policy[s]
                u[s] = np.sum(p[s, a, :] * (r[s, a, :] + gamma * previous_u[:]))
            if np.max(np.abs(u - previous_u)) < delta:
                break

        q = np.einsum('ijk,ijk -> ij', p, r + gamma * u)
        policy = np.argmax(q, axis=1)
        print(policy)

        if np.array_equal(policy, previous_policy):
            break
    # return optimal policy
    return policy


def policy_iteration_iterator(iteration, gamma=0.99, max_iterations=10**6, delta=10**-3):
    r, p = generate_transition_basic_env()

    # initialize with a random policy and initial value function
    policy = np.random.randint(0, r.shape[1], size=r.shape[0])
    u = np.zeros(r.shape[0])

    # iterate and improve policies
    for i in range(iteration):
        for j in range(max_iterations):
            previous_u = u.copy()
            u = np.zeros(shape=(r.shape[0]))
            for s in range(r.shape[0]):
                a = policy[s]
                u[s] = np.sum(p[s, a, :] * (r[s, a, :] + gamma * previous_u[:]))
            if np.max(np.abs(u - previous_u)) < delta:
                break

        q = np.einsum('ijk,ijk -> ij', p, r + gamma * u)
        policy = np.argmax(q, axis=1)

        yield policy


if __name__ == '__main__':
    env = GeneralLearningEnv("config/learning.json", True)

    done = False
    obs = env.reset()
    first = True

    obs_to_s = np.array([-1, 2, 8, 4, 16], dtype=np.int)

    for i, p in enumerate(policy_iteration_iterator(10, 0.1)):
        print(p)
        for j in range(NUMBER_OF_TEST):
            while not done:
                s = int(np.sum(obs * obs_to_s) + 1)
                a = p[s]
                obs, _, done, debug_dict = env.step(a)
            obs = env.reset()
            done = False
            df = debug_dict['stats']
            df = df.assign(policy_iteration=[i])
            if os.path.isfile(RESULT_FILE) or first:
                df.to_csv(RESULT_FILE, sep=',', mode='w', header=True)
                first = False
            else:
                df.to_csv(RESULT_FILE, sep=',', mode='a', header=False)

    env.close()
    del env

