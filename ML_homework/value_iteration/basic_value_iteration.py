import numpy as np
import gym
import time

from oscar.env.envs.general_learning_env import GeneralLearningEnv

from ML_homework.value_iteration.generate_transition import generate_transition_basic_env


def value_iteration(gamma, max_iter, delta):
    r, p = generate_transition_basic_env()

    u = np.random.normal(0.0, 1.0, size=r.shape[0])

    for i in range(max_iter):
        previous_u = u.copy()
        q = np.einsum('ijk,ijk -> ij', p, r + gamma * u)
        u = np.max(q, axis=1)

        if np.max(np.abs(u - previous_u)) < delta:
            break

    policy = np.argmax(q, axis=1)
    return u, policy, i + 1


if __name__ == '__main__':
    u, p, ite = value_iteration(0.1, 1000, 0.0001)
    print(ite)
    print(p.shape)
    print(p)
    print(u)
    print(np.argmax(u), np.max(u))
    env = GeneralLearningEnv("config/learning.json")

    done = False
    obs = env.reset()

    obs_to_s = np.array([-1, 2, 8, 4, 16], dtype=np.int)

    i = 0
    while not done:
        s = int(np.sum(obs * obs_to_s) + 1)
        a = p[s]
        obs, _, done, _ = env.step(a)
        i += 1

    print(i)

    env.reset()
    env.close()
    del env

