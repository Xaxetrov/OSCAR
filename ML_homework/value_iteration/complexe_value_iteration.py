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


def value_iteration_iterator(gamma, max_iter, file_path: str='state_table.csv'):
    q_file_path = "ML_homework/value_iteration/tmp_files/"
    if not os.path.exists(q_file_path):
        os.mkdir(q_file_path)
    number_of_state = ComplexState.get_number_of_state()
    number_of_action = 6
    u = np.random.normal(0.0, 1.0, size=number_of_state)

    with open(file_path, 'r') as state_file:
        for i in range(max_iter):
            previous_u = u.copy()
            q_file = os.path.join(q_file_path, "q_" + str(i) + ".npy")
            if os.path.exists(q_file):
                q = np.load(q_file, 'r')
            else:
                q = np.zeros(shape=(number_of_state, number_of_action))
                state_file.seek(0)  # go back to file start
                for line in state_file:
                    # read state / transition information
                    splited = line.split(',')
                    state_id, action_id, next_state_id = int(splited[0]), int(splited[1]), int(splited[2])
                    r, p = float(splited[3]), float(splited[4])
                    # update q
                    q[state_id, action_id] += p * (r + gamma * u[next_state_id])
                np.save(q_file, q)
            u = np.max(q, axis=1)
            policy = np.argmax(q, axis=1)
            yield policy, np.max(np.abs(u - previous_u))


def state_from_obs(obs):
    state = ComplexState()
    state.minerals = int(min(obs[0] // 10, MINERALS_LIMIT))
    state.food = int(min((obs[1] - 15) // 8, FOOD_LIMIT))
    state.army_count = int(min(obs[2] // 10, ARMY_COUNT_LIMIT))
    state.scv_count = int(max(12, min(obs[3], SCV_COUNT_LIMIT + 12)))
    state.barracks = int(min(obs[4], BARRACK_LIMIT))
    state.time_step = int(min(obs[5] // 100, TIME_STEP_LIMIT))
    return state


if __name__ == '__main__':
    STATE_FILE = "ML_homework/state_table.csv"
    # UTILITY_FILE = "ML_homework/utility.npy"
    # POLICY_FILE = "ML_homework/policy.npy"
    NUMBER_OF_TEST = 1
    RESULT_FILE = "ML_homework/value_iteration/complex.csv"
    if not os.path.isfile(STATE_FILE):
        generate_transition_complex_env(STATE_FILE)
        print("state generation finished")
    else:
        print("state generation already done")

    # if not os.path.isfile(UTILITY_FILE) or not os.path.isfile(POLICY_FILE):
    #     utility, pi, iteration = value_iteration(0.1, 1000, 0.000001)
    #     np.save(UTILITY_FILE, utility)
    #     np.save(POLICY_FILE, pi)
    # else:
    #     utility = np.load(UTILITY_FILE, 'r')
    #     pi = np.load(POLICY_FILE, 'r')

    env = GeneralLearningEnv("config/learning_complex.json", True)

    done = False
    obs = env.reset()
    for pi, error in value_iteration_iterator(0.1, 10):
        for i in range(NUMBER_OF_TEST):
            while not done:
                s = state_from_obs(obs)
                a = pi[s.id()]
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




