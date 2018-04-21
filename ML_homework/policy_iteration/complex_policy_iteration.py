import numpy as np
import time
import pandas as pd
import os

from oscar.env.envs.general_learning_env import GeneralLearningEnv
from ML_homework.value_iteration.generate_transition import *

NUMBER_OF_TEST = 1


def policy_iteration_iterator(iterations, gamma=0.99, max_iterations=10 ** 6,  delta=10 ** -3,
                              file_path: str= 'state_table.csv', save_path: str='/tmp'):
    number_of_state = ComplexState.get_number_of_state()
    number_of_actions = 6

    # initialize with a random policy and initial value function
    policy = np.random.randint(0, number_of_actions, size=number_of_state)
    u = np.zeros(number_of_state)

    # iterate and improve policies
    with open(file_path, 'r') as state_file:
        for i in range(iterations):
            policy_file = os.path.join(save_path, "policy_" + str(i) + ".npy")
            if os.path.isfile(policy_file):
                policy = np.load(policy_file)
            else:
                for j in range(max_iterations):
                    previous_u = u.copy()
                    u = np.zeros(shape=number_of_state)
                    state_file.seek(0)  # go back to file start
                    for line in state_file:
                        # read state / transition information
                        splited = line.split(',')
                        state_id, action_id, next_state_id = int(splited[0]), int(splited[1]), int(splited[2])
                        r, p = float(splited[3]), float(splited[4])
                        if action_id != policy[state_id]:
                            continue
                        u[state_id] += p * (r + gamma * previous_u[next_state_id])
                    if np.max(np.abs(u - previous_u)) < delta:
                        break

                q = np.zeros(shape=(number_of_state, number_of_actions))
                state_file.seek(0)  # go back to file start
                for line in state_file:
                    # read state / transition information
                    splited = line.split(',')
                    state_id, action_id, next_state_id = int(splited[0]), int(splited[1]), int(splited[2])
                    r, p = float(splited[3]), float(splited[4])
                    q[state_id, action_id] += p * (r + gamma * u[next_state_id])

                policy = np.argmax(q, axis=1)
                np.save(policy_file, policy)

            yield policy


def state_from_obs(obs):
    state = ComplexState()
    state.minerals = int(min(int((obs[0] * 40)), MINERALS_LIMIT))
    state.food = int(min((int(obs[1] * 100) - 15) // 8, FOOD_LIMIT))
    state.army_count = int(min(int(obs[2] * 100) // 10, ARMY_COUNT_LIMIT))
    state.scv_count = int(max(12, min(int(obs[3] * 24), SCV_COUNT_LIMIT + 12)))
    state.barracks = int(min(int(obs[4] * 4), BARRACK_LIMIT))
    state.time_step = int(min(int(obs[5] * 10.0), TIME_STEP_LIMIT))
    return state


if __name__ == '__main__':
    # STATE_FILE = "ML_homework/state_table.csv"
    STATE_FILE = "/tmp/state_table.csv"
    P_SAVE_PATH = "/tmp/OSCAR/"
    # UTILITY_FILE = "ML_homework/utility.npy"
    # POLICY_FILE = "ML_homework/policy.npy"
    NUMBER_OF_TEST = 30
    RESULT_FILE = "ML_homework/policy_iteration/complex.csv"
    if not os.path.isfile(STATE_FILE):
        generate_transition_complex_env(STATE_FILE)
        print("state generation finished")
    else:
        print("state generation already done")

    env = GeneralLearningEnv("config/learning_complex.json", False)

    obs = env.reset()

    for i, p in enumerate(policy_iteration_iterator(10, 0.1, file_path=STATE_FILE, save_path=P_SAVE_PATH)):
        for j in range(NUMBER_OF_TEST):
            while True:
                s = state_from_obs(obs)
                a = p[s.id()]
                obs, _, done, debug_dict = env.step(a)
                if done:
                    break
            obs = env.reset()
            df = debug_dict['stats']
            df = df.assign(policy_iteration=[i])
            if not os.path.isfile(RESULT_FILE):
                df.to_csv(RESULT_FILE, sep=',', mode='w', header=True)
            else:
                df.to_csv(RESULT_FILE, sep=',', mode='a', header=False)

    env.close()
    del env