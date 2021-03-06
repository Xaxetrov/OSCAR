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

# ENV_NAME = "pysc2-simple64-meta-per-v0"
ENV_NAME = "general-learning-v0"
SAVE_PATH = "learning_tools/learning_nn/" + ENV_NAME + "/dqn"

NUMBER_OF_TRAINING_GAME = 5
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
        try:
            saver.restore(sess, SAVE_PATH + '/')
        except:
            print("old save not found, use new NN")

        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=100000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        t = 0
        old_t = 0
        try:
            for g in range(NUMBER_OF_TRAINING_GAME):
                done = False
                while not done:
                    # Take action and update exploration to the newest value
                    action = act(obs[None], update_eps=exploration.value(t))[0]
                    new_obs, rew, done, _ = env.step(action)
                    # Store transition in the replay buffer.
                    replay_buffer.add(obs, action, rew, new_obs, float(done))
                    obs = new_obs

                    episode_rewards[-1] += rew
                    if done:
                        obs = env.reset()
                        episode_rewards.append(0)

                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if t > 1000:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # Update target network periodically.
                    if t % 1000 == 0:
                        update_target()

                    if done:  # and len(episode_rewards) % 10 == 0:
                        # logger.record_tabular("steps", t)
                        # logger.record_tabular("episodes", len(episode_rewards) - 1)
                        # logger.record_tabular("episode steps", t - old_t)
                        # logger.record_tabular("episode reward", episode_rewards[-2])
                        # logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                        logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                        logger.dump_tabular()
                        # old_t = t
                    t += 1
        except KeyboardInterrupt:
            print("Training aborted")
        else:
            print("Training finished")
        env.close()
        del env

        # save neural network
        # saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        try:
            os.makedirs(SAVE_PATH)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(SAVE_PATH):
                pass
            else:
                raise
        saver.save(sess, SAVE_PATH + '/')


