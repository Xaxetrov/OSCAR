import unittest
from oscar.env.envs.pysc2_mineralshard_env2 import Pysc2MineralshardEnv2
from pysc2.lib import actions


class TestPysc2MineralshardEnv2(unittest.TestCase):
    def test_get_action_id_from_action(self):
        number_of_action = Pysc2MineralshardEnv2.action_space.n
        for i in range(number_of_action):
            if i >= 256:
                sc2_action, arg = Pysc2MineralshardEnv2.get_select_action(i - 256)
                sc2_action = actions.FUNCTIONS.select_point.id
                arg = arg[:2]
            else:
                sc2_action, arg = Pysc2MineralshardEnv2.get_move_action(i)
            action_id = Pysc2MineralshardEnv2.get_action_id_from_action(sc2_action, arg)
            print(i, ": ", sc2_action, arg)
            self.assertEqual(i, action_id, " i: " + str(i) + " action_id: " + str(action_id))

    def test_steps_obs_persistance(self):
        env = Pysc2MineralshardEnv2()
        obs0 = env.reset()
        obs1, _, _, _ = env.step(0)
        obs2, _, _, _ = env.step(0)

        self.assertIsNot(obs0, obs1)
        self.assertIsNot(obs1, obs2)
        self.assertIsNot(obs0, obs2)


if __name__ == '__main__':
    unittest.main()
