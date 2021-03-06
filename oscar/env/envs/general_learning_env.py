import gym
import threading
import numpy as np
import pandas as pd
from gym import spaces
import time
import os

from oscar.env.envs.pysc2_general_env import Pysc2GeneralEnv
from oscar.constants import *

# from baselines import logger

GAME_MAX_STEP = 1700 * 16  # defined as 1700 agent step when playing one action every 16 game step

# reward cost:
KILLED_UNITS_REWARD = 0.1
KILLED_BUILDINGS_REWARD = 0.2
CREATED_MARINES_REWARD = 0.1
NO_ARMY_REWARD = 0.0
WIN_REWARD = 1.0
LOSS_REWARD = 0.0
STATE_MATE_REWARD = 0.2

DEFAULT_CONFIGURATION = "config/learning.json"


class GeneralLearningEnv(gym.Env):
    def __init__(self, configuration_file=DEFAULT_CONFIGURATION, enable_visualisation=True, game_steps_per_update=8,
                 log_file_path=None, publish_stats=True):
        self.env_thread = Pysc2EnvRunner(configuration_file=configuration_file,
                                         enable_visualisation=enable_visualisation,
                                         game_steps_per_update=game_steps_per_update)
        self.shared_memory = self.env_thread.shared_memory
        self.log_file = log_file_path
        self.publish_stats = publish_stats
        # get semaphores from shared memory
        self.semaphore_obs_ready = self.shared_memory.semaphore_obs_ready
        self.semaphore_action_set = self.shared_memory.semaphore_action_set
        # set action/observation space from shared memory
        self.action_space = self.shared_memory.action_space
        self.observation_space = self.shared_memory.observation_space
        # start thread
        self.env_thread.start()

    def step(self, action):
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
            if self.publish_stats:
                info_dict["stats"] = self.env_thread.stats
            # log result to disk if asked
            if self.log_file is not None and self.env_thread.stats is not None:
                if os.path.isfile(self.log_file):
                    self.env_thread.stats.to_csv(self.log_file, sep=',', mode='a', header=False)
                else:
                    self.env_thread.stats.to_csv(self.log_file, sep=',', mode='w', header=True)
        # return current obs
        return obs.copy(), reward, done, info_dict

    def reset(self):
        # wait for the env to stop as waiting action
        self.semaphore_obs_ready.acquire(blocking=True, timeout=None)
        return self.shared_memory.shared_obs

    def close(self):
        self.env_thread.stop = True
        self.semaphore_action_set.release()  # to be sure that the thread go to stop condition

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        pass

    def get_action_mask(self):
        # every action is supposed to be playable
        # return np.ones(shape=self.action_space.n)
        # get the actions available according to the learning agent
        return self.shared_memory.available_action_mask


class Pysc2EnvRunner(threading.Thread):

    def __init__(self, configuration_file, enable_visualisation, game_steps_per_update):
        self.done = False
        self.was_done = False
        self.reward = 0
        self.last_army_count = 0
        self.last_killed_units = 0
        self.last_killed_building = 0
        self.step_count = 0
        self.last_obs = None
        self.stats = None
        self.game_steps_per_update = game_steps_per_update
        # variable for stats
        self.episodes_reward = []
        self.start_time = 0
        self.learning_agent_step = 0
        # setup env
        self.env = Pysc2GeneralEnv(configuration_file, enable_visualisation, game_steps_per_update)
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

            self.compute_episode_stats()
            # run end, release semaphore to let main env finish its step
            self.semaphore_obs_ready.release()

    def update_reward(self):
        step_reward = 0

        killed_units = self.last_obs.observation['score_cumulative'][KILLED_UNITS]
        killed_buildings = self.last_obs.observation['score_cumulative'][KILLED_BUILDINGS]
        if killed_units > self.last_killed_units:
            step_reward += KILLED_UNITS_REWARD
        if killed_buildings > self.last_killed_building:
            step_reward += KILLED_BUILDINGS_REWARD
        self.last_killed_units = killed_units
        self.last_killed_building = killed_buildings

        army_count = self.last_obs.observation[PLAYER][ARMY_COUNT]
        step_reward += max(0, army_count - self.last_army_count) * CREATED_MARINES_REWARD
        self.last_army_count = army_count
        if army_count == 0:
            step_reward += NO_ARMY_REWARD
        # add end of game reward
        if self.done:
            if self.get_win_state() == 0:
                step_reward = LOSS_REWARD
            if self.get_win_state() == 1:
                step_reward = STATE_MATE_REWARD
            if self.get_win_state() == 2:
                step_reward = WIN_REWARD
        self.reward += step_reward
        # count total reward for this game
        self.episodes_reward[-1] += step_reward

    def get_win_state(self):
        """
        Look at the current state of the game (assuming that it's the episode end)
        to determine if the agent win / loss / null
        :return: int: 0 loss, 1 null and 2 win
        """
        if self.step_count >= GAME_MAX_STEP / self.game_steps_per_update:
            return 1
        elif self.last_obs.observation[PLAYER][ARMY_COUNT] > 10:
            return 2
        else:
            return 0

    def compute_episode_stats(self):
        # logger.record_tabular("time", (time.time() - self.start_time) / 60)
        # logger.record_tabular("steps", self.env.general.steps)
        # logger.record_tabular("episodes", len(self.episodes_reward))
        # logger.record_tabular("episode steps", self.step_count)
        # logger.record_tabular("learning agent steps", self.learning_agent_step)
        # logger.record_tabular("episode reward", self.episodes_reward[-1])
        # logger.record_tabular("mean episode reward", np.mean(self.episodes_reward[-101:]))
        # logger.record_tabular("median episode reward", np.median(self.episodes_reward[-101:]))
        # logger.record_tabular("win state", self.get_win_state())
        # logger.dump_tabular()
        self.stats = pd.DataFrame(data=[[(time.time() - self.start_time) / 60,
                                         self.env.general.steps,
                                         len(self.episodes_reward),
                                         self.step_count,
                                         self.learning_agent_step,
                                         self.episodes_reward[-1],
                                         np.mean(self.episodes_reward[-101:]),
                                         np.median(self.episodes_reward[-101:]),
                                         self.get_win_state()]],
                                  columns=["time",
                                           "steps",
                                           "episodes",
                                           "episode_steps",
                                           "learning_agent_steps",
                                           "episode_reward",
                                           "mean_episode_reward",
                                           "median_episode_reward",
                                           "win_state"]
                                  )



