import gym
import threading
from gym import spaces

from oscar.env.envs.pysc2_general_env import Pysc2GeneralEnv
from oscar.constants import *


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
        # set action into shared memory
        self.shared_memory.shared_action = action
        # let the thread run again
        self.semaphore_action_set.release()
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
        # return current obs
        return obs, reward, done, {}

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


class Pysc2EnvRunner(threading.Thread):

    def __init__(self):
        self.done = False
        self.was_done = False
        self.reward = 0
        self.total_reward = 0
        self.step_reward = 0
        self.last_army_count = 0
        self.step_count = 0
        self.last_obs = None
        # setup env
        self.env = Pysc2GeneralEnv()
        # get shared memory from general
        self.shared_memory = self.env.general.training_memory
        self.semaphore_obs_ready = self.shared_memory.semaphore_obs_ready
        self.semaphore_action_set = self.shared_memory.semaphore_action_set
        self.stop = False
        super().__init__()

    def run(self):
        while True:
            self.done = False
            self.reward = 0
            self.step_count = 0
            self.last_obs = self.env.reset()
            while not self.done:
                self.step_count += 1
                self.last_obs, _, self.done, _ = self.env.step(None)
                self.update_reward()
                if self.stop:
                    return
            self.was_done = True
            self.semaphore_obs_ready.release()
            # run end, release semaphore to let main env finish its step

    def update_reward(self):
        new_total_reward = 0
        new_total_reward += self.last_obs.observation['score_cumulative'][5]
        new_total_reward += self.last_obs.observation['score_cumulative'][6]
        self.step_reward = new_total_reward - self.total_reward
        self.total_reward = new_total_reward
        army_count = self.last_obs.observation['player'][ARMY_COUNT]
        self.step_reward += max(0, army_count - self.last_army_count)
        self.last_army_count = army_count
        self.reward += self.step_reward
