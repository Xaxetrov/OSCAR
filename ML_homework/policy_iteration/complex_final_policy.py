from ML_homework.policy_iteration.complex_policy_iteration import *
from ML_homework.value_iteration.complexe_value_iteration import *

STATE_FILE = "/tmp/state_table.csv"
P_SAVE_PATH = "/tmp/OSCAR/"
Q_SAVE_PATH = "/tmp/OSCAR/"


def find_final_policy():
    old_p = None
    for i, p in enumerate(policy_iteration_iterator(1000, file_path=STATE_FILE, save_path=P_SAVE_PATH)):
        print(i)
        if (p == old_p).all():
            return i, p
        old_p = p.copy()


def find_final_value():
    old_p = None
    for i, (p, error) in enumerate(
            value_iteration_iterator(0.1, 1000, file_path=STATE_FILE, dump_file_path=Q_SAVE_PATH)):
        print(i)
        if (p == old_p).all():
            return i, p
        old_p = p.copy()


def find_final_value_from_policy(target_policy):
    for i, (p, error) in enumerate(
            value_iteration_iterator(0.1, 1000, file_path=STATE_FILE, dump_file_path=Q_SAVE_PATH)):
        diff = np.sum(target_policy != p)
        print(i, diff)
        if diff == 0:
            return i, p


if __name__ == '__main__':
    it, pi_pi = find_final_policy()
    print("final policy at iteration", it)
    it, pi_vi = find_final_value()
    print("final value at iteration", it)
    print((pi_pi != pi_vi).sum())
    it, _ = find_final_value_from_policy(pi_pi)

