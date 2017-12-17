import os
import tempfile
import gym
from baselines import deepq
import baselines.common.tf_util as U
import zipfile
import dill
from baselines import logger

from oscar import env

ENV_NAME = "pysc2-simple64-meta-per-v0"
# ENV_NAME = "CartPole-v0"
SAVE_PATH = "learning_tools/learning_nn/" + ENV_NAME + ".pkl"


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make(ENV_NAME)
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=10,
        buffer_size=50000,
        exploration_fraction=0.4,
        exploration_final_eps=0.02,
        print_freq=10
        # callback=callback
    )
    print("Saving model to", SAVE_PATH)
    # act.save(SAVE_PATH)
    save(act, SAVE_PATH)


def save(act, path):
    """Save model to a pickle located at `path`"""
    with tempfile.TemporaryDirectory() as td:
        U.save_state(os.path.join(td, "model"))
        arc_name = os.path.join(td, "packed.zip")
        with zipfile.ZipFile(arc_name, 'w') as zipf:
            for root, dirs, files in os.walk(td):
                for fname in files:
                    file_path = os.path.join(root, fname)
                    if file_path != arc_name:
                        zipf.write(file_path, os.path.relpath(file_path, td))
        with open(arc_name, "rb") as f:
            model_data = f.read()
    with open(path, "wb") as f:
        params = act._act_params
        dill.dump((model_data, params), f)


if __name__ == '__main__':
    main()
