
from oscar.env import envs
# from oscar.agent.scripted.minigame.bruno_mineralshard import CollectMineralShards
from oscar.agent.scripted.minigame.bruno_mineralshard import CollectMineralShards
from learning_tools.A3C_learner.neuralmodel import get_neural_network, save_neural_network
from learning_tools.A3C_learner.constants import ENV
import gym
import numpy as np
import time

# ENV = 'pysc2-mineralshard-v1' set into the config file (of A3C learner)
TRAINING_STEPS = 240*1
TEST_RUN = 5

RUN_NN_ACTION = True

env = gym.make(ENV)
agent = CollectMineralShards()

output_shape = env.action_space.n
input_shape = env.observation_space.shape

model = get_neural_network(input_shape=(None,) + input_shape,
                           output_shape=[output_shape, 1])

action_batch = []
obs_batch = []

episode_reward = 0
agent.reset()
obs = env.reset()

print("learning for", TRAINING_STEPS, "steps")
try:
    for i in range(TRAINING_STEPS):
        time.sleep(0.5)
        action = agent.step(env.last_obs)

        action_id = env.get_action_id_from_action(sc2_action=action.function,
                                                  sc2_args=action.arguments)

        learning_action = np.zeros(shape=output_shape)
        learning_action[action_id] = 1.0
        print("training:", action_id)
        if RUN_NN_ACTION:
            # train NN
            model.fit(x=np.reshape(obs, (1, len(obs),) + obs[0].shape),
                      y=[np.reshape(learning_action, (1,) + learning_action.shape), np.zeros(1)],
                      verbose=0,
                      epochs=1
                      )
            # get his action
            action = model.predict(x=np.reshape(obs, (1, len(obs),) + obs[0].shape))
            # print("max", np.max(action[0]), "min", np.min(action[0]))
            action_id = np.argmax(action[0][0])
            # action_id = np.random.choice(output_shape, p=action[0][0])
        else:
            # generate batch for training (on episode end only)
            action_batch.append(learning_action)
            obs_batch.append(obs)

        print("selected:", action_id)

        obs, reward, done, _ = env.step(action_id)

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
                          epochs=5
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
        action = model.predict(x=np.array([obs]),
                               batch_size=1)
        action_id = np.argmax(action[0][0])
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
