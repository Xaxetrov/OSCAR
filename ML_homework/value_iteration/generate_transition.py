import numpy as np
# import copy
# from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from math import exp
from multiprocessing import Pool

from oscar.env.envs.general_learning_env import *


def generate_transition_basic_env():
    """
    Try to aproximate the reward in function of the state for the basic environment
    :return: reward and probability arrays
    """
    rewards = np.zeros((32, 5, 32))
    probability = np.zeros((32, 5, 32))

    state_mask = {'max_food': 1, 'army_10': 2, 'supply': 4, 'barrack': 8, 'enemy_fund': 16}

    # iterate possible states
    for state in range(32):
        # iterate actions
        for action_id, action in enumerate(['no_op', 'supply', 'barrack', 'marines', 'attack']):
            if action == 'no_op':
                # rewards stay to 0
                probability[state, action_id, state] = 1.0
            elif action == 'supply':
                # rewards stay to 0
                new_state = state | state_mask['supply']
                probability[state, action_id, new_state] = 1.0
            elif action == 'barrack':
                # rewards stay to 0
                if state & state_mask['supply'] > 0:
                    new_state = state | state_mask['barrack']
                    probability[state, action_id, new_state] = 1.0
                else:
                    probability[state, action_id, state] = 1.0
            elif action == 'marines':
                if state & state_mask['barrack'] > 0 and state & state_mask['max_food'] == 0:
                    for max_food in [True, False]:
                        if max_food:
                            new_state = state | state_mask['max_food']
                            p = 0.05
                        else:
                            new_state = state
                            p = 0.95
                        for army10 in [True, False]:
                            if army10:
                                new_state = new_state | state_mask['army_10']
                                p = p * 0.1
                            else:
                                p = p * 0.9
                            probability[state, action_id, new_state] = p
                            rewards[state, action_id, new_state] = CREATED_MARINES_REWARD
                else:
                    probability[state, action_id, state] = 1.0
            elif action == 'attack':
                if state & state_mask['barrack'] > 0:
                    new_state = state & (~ state_mask['max_food'])
                    new_state = new_state & (~ state_mask['army_10'])
                    if state & state_mask['max_food'] > 0:
                        r = KILLED_UNITS_REWARD + KILLED_BUILDINGS_REWARD + 0.1 * WIN_REWARD
                    elif state & state_mask['army_10'] > 0:
                        r = KILLED_UNITS_REWARD + 0.5 * KILLED_BUILDINGS_REWARD
                    else:
                        r = 0.0
                    if state & state_mask['enemy_fund'] > 0:
                        probability[state, action_id, new_state] = 1.0
                        rewards[state, action_id, new_state] = r
                    else:
                        probability[state, action_id, new_state] = 0.95
                        rewards[state, action_id, new_state] = 0.0
                        new_state = new_state | state_mask['enemy_fund']
                        probability[state, action_id, new_state] = 0.05
                        rewards[state, action_id, new_state] = r
                else:
                    probability[state, action_id, state] = 1.0
    return rewards, probability


def generate_transition_complex_env(file_path: str='state_table.csv'):
    """
    Try to approximate the reward in function of the state for the basic environment
    :return: reward and probability arrays
    """
    # rewards = []
    # probability = []

    # iterate possible states
    print(ComplexState.get_number_of_state())
    with open(file_path, 'w') as out_file:
        pool = Pool()
        for result_list in pool.imap_unordered(study_state, range(ComplexState.get_number_of_state()), chunksize=10000):
            for state_id, action_id, next_state_id, r, p in result_list:
                out_file.write("{:d},{:d},{:d},{:f},{:f}\n".format(state_id,
                                                                   action_id,
                                                                   next_state_id,
                                                                   r,
                                                                   p))
    return


def study_state(state_id: int):
    state = ComplexState()
    state.from_id(state_id)
    results = []
    # iterate actions
    for action_id, action in enumerate(['no_op', 'supply', 'barrack', 'marines', 'attack', 'scv']):
        if action == 'no_op':
            # rewards stay to 0
            new_states, p, r = no_op(state)
        elif action == 'supply':
            # rewards stay to 0
            new_state = state.get_base_next_state()
            new_state.add_supply()
            new_states = [new_state]
            p = [1.0]
            r = [0.0]
        elif action == 'barrack':
            # rewards stay to 0
            new_state = state.get_base_next_state()
            new_state.add_barrack()
            new_states = [new_state]
            p = [1.0]
            r = [0.0]
        elif action == 'marines':
            if state.barracks > 0 and state.can_create_unit():
                new_state = state.get_base_next_state()
                created_marines_num = new_state.add_marines(state.barracks)
                new_state_army_increased = new_state.copy()
                new_state_army_increased.increase_army_count()
                new_states = [new_state, new_state_army_increased]
                p = [0.1 * created_marines_num,
                     1.0 - 0.1 * created_marines_num]
                r = [CREATED_MARINES_REWARD * created_marines_num,
                     CREATED_MARINES_REWARD * created_marines_num]
            else:
                new_states, p, r = no_op(state)
        elif action == 'attack':
            if state.barracks > 0:
                new_state = state.get_base_next_state()
                new_state.attack()
                new_states = [new_state]
                p = [1.0]
                enemy_unit = 100.0 * (1.0 - exp(-state.time_step / 2.0))
                owned_units = state.army_count * 10
                if 1.2 * enemy_unit > owned_units:
                    r = [owned_units ** 2 / (enemy_unit * 2) * KILLED_UNITS_REWARD]
                elif 2.0 * enemy_unit < owned_units:
                    r = [enemy_unit / 2 * KILLED_UNITS_REWARD + KILLED_BUILDINGS_REWARD,
                         enemy_unit / 2 * KILLED_UNITS_REWARD + KILLED_BUILDINGS_REWARD + WIN_REWARD]
                    p = [0.9,
                         0.1]
                    new_states = [new_state, new_state]
                else:
                    r = [enemy_unit / 2 * KILLED_UNITS_REWARD + 0.5 * KILLED_BUILDINGS_REWARD]
            else:
                new_states, p, r = no_op(state)
        elif action == 'scv':
            new_state = state.get_base_next_state()
            new_state.add_scv()
            new_states = [new_state]
            p = [1.0]
            r = [0.0]
        else:
            raise ValueError("Unknown action", action)
        for next_state, trans_p, trans_r in zip(new_states, p, r):
            if next_state.time_step < TIME_STEP_LIMIT:
                results.append(get_action_tuple(state_id, action_id, next_state.id(), trans_r, trans_p * 0.995))
                next_state.to_next_time_step()
                results.append(get_action_tuple(state_id, action_id, next_state.id(), trans_r, trans_p * 0.005))
            else:
                results.append(get_action_tuple(state_id, action_id, next_state.id(), trans_r, trans_p))
    return results


def no_op(state):
    new_states = [state.get_base_next_state()]
    p = [1.0]
    r = [0.0]
    return new_states, p, r


def get_action_tuple(state_id, action_id, next_state_id, reward, probability):
    return (np.uint32(state_id),
            np.uint8(action_id),
            np.uint32(next_state_id),
            np.float32(reward),
            np.float32(probability))


MAX_MINERALS = 21
MINERALS_LIMIT = MAX_MINERALS - 1

MAX_ARMY_COUNT = 19
ARMY_COUNT_LIMIT = MAX_ARMY_COUNT - 1

MAX_SCV_COUNT = 13
SCV_COUNT_LIMIT = MAX_SCV_COUNT - 1

MAX_BARRACK = 5
BARRACK_LIMIT = MAX_BARRACK - 1

MAX_FOOD = 23
FOOD_LIMIT = MAX_FOOD - 1

MAX_TIME_STEP = 5
TIME_STEP_LIMIT = MAX_TIME_STEP - 1


class ComplexState:
    def __init__(self):
        self.minerals = 0
        self.army_count = 0
        self.scv_count = 0
        self.barracks = 0
        self.food = 0
        self.time_step = 0

    def can_create_unit(self):
        return (self.army_count + 1) * 10 + self.scv_count < self.get_food_available() and self.minerals > 5

    def minerals_collected_by_step(self):
        if self.scv_count <= 16:
            return self.scv_count * 2
        elif self.scv_count <= 24:
            return 16 * 2 + self.scv_count // 2
        else:
            return 16 * 2 + 8 // 2

    def get_base_next_state(self):
        ret = self.copy()
        ret.minerals += self.minerals_collected_by_step()
        if ret.minerals > MINERALS_LIMIT:
            ret.minerals = MINERALS_LIMIT
        return ret

    def get_food_available(self):
        return 15 + 8 * self.food

    def id(self):
        state_num = self.minerals
        state_num += MAX_MINERALS * self.army_count
        state_num += MAX_MINERALS * MAX_ARMY_COUNT * (self.scv_count - 12)
        state_num += MAX_MINERALS * MAX_ARMY_COUNT * MAX_SCV_COUNT * self.food
        state_num += MAX_MINERALS * MAX_ARMY_COUNT * MAX_SCV_COUNT * MAX_FOOD * self.time_step
        state_num += MAX_MINERALS * MAX_ARMY_COUNT * MAX_SCV_COUNT * MAX_FOOD * MAX_TIME_STEP * self.barracks
        return int(state_num)

    @staticmethod
    def get_number_of_state():
        return MAX_MINERALS * MAX_ARMY_COUNT * MAX_SCV_COUNT * MAX_FOOD * MAX_TIME_STEP * MAX_BARRACK

    def from_id(self, state_id):
        tmp = state_id
        self.minerals = tmp % MAX_MINERALS
        tmp = tmp // MAX_MINERALS
        self.army_count = tmp % MAX_ARMY_COUNT
        tmp = tmp // MAX_ARMY_COUNT
        self.scv_count = 12 + tmp % MAX_SCV_COUNT
        tmp = tmp // MAX_SCV_COUNT
        self.food = tmp % MAX_FOOD
        tmp = tmp // MAX_FOOD
        self.time_step = tmp % MAX_TIME_STEP
        tmp = tmp // MAX_TIME_STEP
        self.barracks = tmp
        if tmp > MAX_BARRACK:
            raise ValueError("Conversion from id to state failed")

    def add_supply(self):
        if self.food < FOOD_LIMIT and self.minerals >= 10:
            self.food += 1
            self.minerals -= 10

    def add_barrack(self):
        if self.food > 0 and self.minerals >= 15 and self.barracks < BARRACK_LIMIT:
            self.barracks += 1
            self.minerals -= 15

    def add_marines(self, count: int):
        can_create = min(self.get_food_available() - ((self.army_count + 1) * 10 + self.scv_count),
                         self.minerals // 5)
        can_create = max(0, can_create)
        to_create = min(count, can_create)
        self.minerals -= 5 * to_create
        return to_create

    def add_scv(self):
        if self.can_create_unit() and self.scv_count < SCV_COUNT_LIMIT + 12:
            self.scv_count += 1
            self.minerals -= 5

    def attack(self):
        self.army_count = 0

    def to_next_time_step(self):
        if self.time_step < TIME_STEP_LIMIT:
            self.time_step += 1

    def increase_army_count(self):
        if self.army_count < ARMY_COUNT_LIMIT:
            self.army_count += 1

    def copy(self):
        ret = ComplexState()
        ret.army_count = self.army_count
        ret.time_step = self.time_step
        ret.barracks = self.barracks
        ret.food = self.food
        ret.scv_count = self.scv_count
        ret.minerals = self.minerals
        return ret

