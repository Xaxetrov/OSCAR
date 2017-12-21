import gym
import threading
import numpy as np
from gym import spaces
import time

from oscar.env.envs.pysc2_general_env import Pysc2GeneralEnv
from oscar.constants import *

from baselines import logger

GAME_MAX_STEP = 1700


class GeneralLearningEnv(gym.Env):
    def __init__(self):
        self.env_thread = Pysc2EnvRunner()
        self.shared_memory = self.env_thread.shared_memory
        # get semaphores from shared memory
        self.semaphore_obs_ready = self.shared_memory.semaphore_obs_ready
        self.semaphore_action_set = self.shared_memory.semaphore_action_set
        # set action/observation space from shared memory
        self.action_space = self.shared_memory.action_space
        self.observation_space = self.shared_memory.observation_space
        # start thread
        self.env_thread.start()

    def _step(self, action):
        info_dict = {}
        # set action into shared memory
        self.shared_memory.shared_action = action
        # let the thread run again
        self.semaphore_action_set.release()
        # increase the number of steps performed by the learning agent
        self.env_thread.learning_agent_step += 1
        # wait a call to the target agent
        self.semaphore_obs_ready.acquire(blocking=True, timeout=None)
        # get the obs of the current state as seen by the agent
        obs = self.shared_memory.shared_obs
        # get the reward of the current run and reset it to 0 for next step
        reward = self.env_thread.reward  # / self.env_thread.step_count
        self.env_thread.reward = 0
        # get done state from thread
        done = self.env_thread.was_done
        if done:
            self.env_thread.was_done = False  # reset was_done to false for next loop
            info_dict["win_state"] = self.env_thread.get_win_state()
        # return current obs
        return obs, reward, done, info_dict

    def _reset(self):
        # wait for the env to stop as waiting action
        self.semaphore_obs_ready.acquire(blocking=True, timeout=None)
        return self.shared_memory.shared_obs

    def _close(self):
        self.env_thread.stop = True
        self.semaphore_action_set.release()  # to be sure that the thread go to stop condition

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed=None):
        pass

    def get_action_mask(self):
        # every action is supposed to be playable
        return np.ones(shape=self.action_space.n)


class Pysc2EnvRunner(threading.Thread):

    def __init__(self):
        self.done = False
        self.was_done = False
        self.reward = 0
        self.total_reward = 0
        self.last_army_count = 0
        self.step_count = 0
        self.last_obs = None
        # variable for stats
        self.episodes_reward = []
        self.start_time = 0
        self.learning_agent_step = 0
        # setup env
        self.env = Pysc2GeneralEnv()
        # get shared memory from general
        self.shared_memory = self.env.general.training_memory
        self.semaphore_obs_ready = self.shared_memory.semaphore_obs_ready
        self.semaphore_action_set = self.shared_memory.semaphore_action_set
        self.stop = False
        super().__init__()

    def run(self):
        self.start_time = time.time()
        while True:
            self.done = False
            self.reward = 0
            self.total_reward = 0
            self.last_army_count = 0
            self.step_count = 0
            self.learning_agent_step = 0
            self.episodes_reward.append(0)
            self.last_obs = self.env.reset()
            while not self.done:
                self.step_count += 1
                self.last_obs, _, self.done, _ = self.env.step(None)
                self.update_reward()
                if self.stop:
                    self.env.close()
                    del self.env
                    return
            self.was_done = True

            self.print_episode_stats()
            # run end, release semaphore to let main env finish its step
            self.semaphore_obs_ready.release()

    def update_reward(self):
        new_total_reward = 0
        new_total_reward += self.last_obs.observation['score_cumulative'][KILLED_UNITS]
        new_total_reward += self.last_obs.observation['score_cumulative'][KILLED_BUILDINGS]
        step_reward = 0
        step_reward += new_total_reward - self.total_reward
        self.total_reward = new_total_reward
        army_count = self.last_obs.observation['player'][ARMY_COUNT]
        step_reward += max(0, army_count - self.last_army_count)
        self.last_army_count = army_count
        self.reward += step_reward
        self.episodes_reward[-1] += step_reward

    def get_win_state(self):
        """
        Look at the current state of the game (assuming that it's the episode end)
        to determine if the agent win / loss / null
        :return: int: 0 loss, 1 null and 2 win
        """
        if self.step_count >= GAME_MAX_STEP:
            return 1
        elif self.last_obs.observation['player'][ARMY_COUNT] > 10:
            return 2
        else:
            return 0

    def print_episode_stats(self):
        logger.record_tabular("time", (time.time() - self.start_time) / 60)
        logger.record_tabular("steps", self.env.general.steps)
        logger.record_tabular("episodes", len(self.episodes_reward))
        logger.record_tabular("episode steps", self.step_count)
        logger.record_tabular("learning agent steps", self.learning_agent_step)
        logger.record_tabular("episode reward", self.episodes_reward[-1])
        logger.record_tabular("mean episode reward", np.mean(self.episodes_reward[-101:]))
        logger.record_tabular("median episode reward", np.median(self.episodes_reward[-101:]))
        logger.record_tabular("win state", self.get_win_state())
        logger.dump_tabular()
