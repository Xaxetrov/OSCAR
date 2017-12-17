import gym
import os
import errno
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

import oscar.env

ENV_NAME = "pysc2-simple64-meta-per-v0"
SAVE_PATH = "learning_tools/learning_nn/" + ENV_NAME + "/dqn"

NUMBER_OF_TESTING_GAME = 10
NUMBER_OF_CPU = 4


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session(NUMBER_OF_CPU) as sess:
        # Create the environment
        env = gym.make(ENV_NAME)
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        # try to restore old weight
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_PATH + '/')

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        t = 0
        for g in range(NUMBER_OF_TESTING_GAME):
            done = False
            while not done:
                # Take action and update exploration to the newest value
                action = act(obs[None], update_eps=0.0)[0]
                print(action)
                new_obs, rew, done, _ = env.step(action)
                # Store transition in the replay buffer.
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    obs = env.reset()
                    episode_rewards.append(0)
                # if done and len(episode_rewards) % 10 == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                    logger.dump_tabular()
                t += 1

        print("test finished")
        env.close()
        del env

