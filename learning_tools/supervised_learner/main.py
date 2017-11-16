
from oscar.env import envs
# from oscar.agent.scripted.minigame.deepmindAgents import CollectMineralShards
from oscar.agent.scripted.minigame.bruno_mineralshard import CollectMineralShards
from learning_tools.A3C_learner.neuralmodel import get_neural_network, save_neural_network
from learning_tools.A3C_learner.constants import ENV
import gym
import numpy as np
import time

# ENV = 'pysc2-mineralshard-v1' set into the constants file (of A3C learner)
TRAINING_STEPS = 240 * 100
TEST_RUN = 5

RUN_NN_ACTION = True

env = gym.make(ENV)
agent = CollectMineralShards()

output_shape = env.action_space.n
input_shape = env.observation_space.shape

model = get_neural_network(input_shape=(None,) + input_shape,
                           output_shape=[output_shape, 1])
model.summary()

action_batch = []
obs_batch = []

episode_reward = 0
agent.reset()
obs = env.reset()

print("learning for", TRAINING_STEPS, "steps")
try:
    for i in range(TRAINING_STEPS):
        # time.sleep(0.1)
        action = agent.step(env.last_obs)

        action_id = env.get_action_id_from_action(sc2_action=action.function,
                                                  sc2_args=action.arguments)

        learning_action = np.zeros(shape=output_shape)
        learning_action[action_id] = 1.0
        if RUN_NN_ACTION:
            # train NN
            model.fit(x=np.reshape(obs, (1, len(obs),) + obs[0].shape),
                      y=[np.reshape(learning_action, (1,) + learning_action.shape), np.zeros(1)],
                      verbose=0,
                      epochs=3
                      )
            # get his action
            p = model.predict(x=np.reshape(obs, (1, len(obs),) + obs[0].shape))[0][0]
            # get mask for unavailable action
            action_mask = env.get_action_mask()
            # apply mask to action probability
            p *= action_mask
            # normalize (set sum back to 1.0)
            p_sum = np.sum(p)
            p /= p_sum
            # print("max", np.max(action[0]), "min", np.min(action[0]))
            # played_action_id = np.argmax(p)
            played_action_id = np.random.choice(len(p), p=p)
        else:
            # generate batch for training (on episode end only)
            action_batch.append(learning_action)
            obs_batch.append(obs)
            played_action_id = action_id

        # print("action: train", action_id, "selected", played_action_id)
        # if played_action_id < 256:
        #     action = env.get_select_action(played_action_id)
        # else:
        #     action = env.get_move_action(played_action_id - 256)
        # print("played action:", action)
        # if action_id < 256:
        #     action = env.get_select_action(action_id)
        # else:
        #     action = env.get_move_action(action_id - 256)
        # print("trained action:", action)

        obs, reward, done, _ = env.step(played_action_id)

        episode_reward += reward

        if done:
            print("step:", i, "- episode reward:", episode_reward)
            episode_reward = 0
            if not RUN_NN_ACTION:
                # learn from current batch
                model.fit(x=np.array(obs_batch),
                          y=[np.array(action_batch), np.zeros(len(action_batch))],
                          batch_size=len(action_batch),
                          verbose=0,
                          epochs=10
                          )
                action_batch = []
                obs_batch = []
            # reset env and agent
            agent.reset()
            obs = env.reset()
except KeyboardInterrupt:
    print("Keyboard interrupt, save NN and exit")
else:
    print("Test for", TEST_RUN, "minigames")
    obs = env.reset()
    minigame_reward = 0
    for i in range(TEST_RUN * 240 + 1):
        p = model.predict(x=np.array([obs]),
                          batch_size=1)[0][0]
        # get mask for unavailable action
        action_mask = env.get_action_mask()
        # apply mask to action probability
        p *= action_mask
        # normalize (set sum back to 1.0)
        p_sum = np.sum(p)
        p /= p_sum
        # action_id = np.argmax(p)
        action_id = np.random.choice(len(p), p=p)
        # if action_id < 256:
        #     action = env.get_select_action(action_id)
        # else:
        #     action = env.get_move_action(action_id - 256)
        # print("played action:", action)
        obs, reward, done, _ = env.step(action_id)
        minigame_reward += reward
        if done:
            obs = env.reset()
            print("reward", minigame_reward)
            minigame_reward = 0


save_neural_network(model=model)

agent.reset()
env.reset()
env.close()

del env
