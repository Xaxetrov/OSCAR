import unittest
from oscar.env.envs.pysc2_mineralshard_env2 import Pysc2MineralshardEnv2
from pysc2.lib import actions


class TestPysc2MineralshardEnv2(unittest.TestCase):
    def test_get_action_id_from_action(self):
        number_of_action = Pysc2MineralshardEnv2.action_space.n
        for i in range(number_of_action):
            if i < 256:
                sc2_action, arg = Pysc2MineralshardEnv2.get_select_action(i)
                sc2_action = actions.FUNCTIONS.select_point.id
                arg = arg[:2]
            else:
                sc2_action, arg = Pysc2MineralshardEnv2.get_move_action(i - 256)
            action_id = Pysc2MineralshardEnv2.get_action_id_from_action(sc2_action, arg)
            print(i, ": ", sc2_action, arg)
            self.assertEqual(i, action_id, " i: " + str(i) + " action_id: " + str(action_id))


if __name__ == '__main__':
    unittest.main()
