import numpy as np
import gym
import os

from oscar.env.envs.general_learning_env import GeneralLearningEnv
from ML_homework.value_iteration.generate_transition import *


def value_iteration(gamma, max_iter, delta, file_path: str='state_table.csv'):
    # results = generate_transition_complex_env()

    number_of_state = ComplexState.get_number_of_state()
    number_of_action = 6
    u = np.random.normal(0.0, 1.0, size=number_of_state)

    with open(file_path, 'r') as state_file:
        for i in range(max_iter):
            print(i)
            previous_u = u.copy()
            q = np.zeros(shape=(number_of_state, number_of_action))
            state_file.seek(0)  # go back to file start
            for line in state_file:
                # read state / transition information
                splited = line.split(',')
                state_id, action_id, next_state_id = int(splited[0]), int(splited[1]), int(splited[2])
                r, p = float(splited[3]), float(splited[4])
                # update q
                q[state_id, action_id] += p * (r + gamma * u[next_state_id])
            u = np.max(q, axis=1)
            if np.max(np.abs(u - previous_u)) < delta:
                break

    policy = np.argmax(q, axis=1)
    return u, policy, i + 1


def state_from_obs(obs):
    state = ComplexState()
    state.minerals = int(min(obs[0] // 10, MINERALS_LIMIT))
    state.food = int(min((obs[1] - 15) // 8, FOOD_LIMIT))
    state.army_count = int(min(obs[2] // 10, ARMY_COUNT_LIMIT))
    state.scv_count = int(max(12, min(obs[3], SCV_COUNT_LIMIT + 12)))
    state.barracks = int(min(obs[4], BARRACK_LIMIT))
    state.time_step = int(min(obs[5] // 100, TIME_STEP_LIMIT))
    print(state.barracks)
    return state


if __name__ == '__main__':
    STATE_FILE = "ML_homework/state_table.csv"
    UTILITY_FILE = "ML_homework/utility.npy"
    POLICY_FILE = "ML_homework/policy.npy"
    if not os.path.isfile(STATE_FILE):
        generate_transition_complex_env()
        print("state generation finished")
    else:
        print("state generation already done")

    if not os.path.isfile(UTILITY_FILE) or not os.path.isfile(POLICY_FILE):
        utility, pi, iteration = value_iteration(0.1, 1000, 0.0001)
        np.save(UTILITY_FILE, utility)
        np.save(POLICY_FILE, pi)
    else:
        utility = np.load(UTILITY_FILE, 'r')
        pi = np.load(POLICY_FILE, 'r')

    env = GeneralLearningEnv("config/learning_complex.json")

    done = False
    obs = env.reset()

    i = 0
    while not done:
        s = state_from_obs(obs)
        a = pi[s.id()]
        obs, _, done, _ = env.step(a)
        i += 1

    env.reset()
    env.close()
    del env




