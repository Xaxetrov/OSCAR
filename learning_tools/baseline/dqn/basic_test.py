import gym
from baselines import deepq

from oscar import env

ENV_NAME = "pysc2-simple64-meta-per-v0"
SAVE_PATH = "learning_tools/learning_nn/" + ENV_NAME + ".pkl"


def main():
    env = gym.make(ENV_NAME)
    act = deepq.load(SAVE_PATH)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
