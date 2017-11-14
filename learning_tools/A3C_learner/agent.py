import gym
import time
import random
import threading
import numpy as np

from learning_tools.A3C_learner.constants import *

brain = None  # brain is global in A3C
NUM_ACTIONS = 0
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.

    def get_epsilon(self):
        if frames >= self.eps_steps:
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s, env):
        """
        In Agent class, the act() method returns an action to take.
        To support exploration, it implements ε-greedy policy with linearly decreasing rate.
        The action is selected according to the policy distribution returned by the neural network.
        :param s: state
        :param env: environment
        :return: an action
        """
        eps = self.get_epsilon()
        global frames
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # get mask for unavailable action
            action_mask = env.get_action_mask()
            # apply mask to action probability
            p *= action_mask
            # normalize (set sum back to 1.0)
            p_sum = np.sum(p)
            p /= p_sum

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        """
        The train() method receives samples, processes them and pushes them into the training queue.
        First, it turns actions into one hot encoded array needed later.
        Then it stores the current transition in an internal memory, which is used to compute the n-step return.
        :param s: state
        :param a: action
        :param r: reward
        :param s_: next state
        :return:
        """
        def get_sample(memory, n):
            """
            An internal get_sample() function is responsible to compute n-step discounted reward
            and return a proper tuple:
            :param memory: step memory
            :param n: nth step
            :return: tuple
            """
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        """
        The last thing we have to deal with is the case when our agent encounters a terminal state.
        As there will be no more states after, we have to deplete the whole buffer.
        In loop, we shorten the buffer in each iteration and compute the n-step return,
        where n is equal to the current length of the buffer.
        """
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0
        """
        Last n samples are stored in this buffer and when there are enough of them,
        n-step discounted reward R is computed.
        Proper variables are retrieved and a tuple (s_0, a_0, R, s_n) is inserted into the brain’s training queue.
        """
        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect


class Environment(threading.Thread):
    """
    The Environment class is an instance of OpenAI Gym environment and contains an instance of Agent.
    It is also a thread that continuously runs one episode after another.
    """
    stop_signal = False

    def __init__(self, global_brain, num_actions, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)
        global brain, NUM_ACTIONS
        brain = global_brain
        NUM_ACTIONS = num_actions

        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def run_episode(self):
        """
        The THREAD_DELAY parameter controls a delay between steps.
        It enables to have more parallel environments than there are CPUs.
        High number of agents is crucial for convergence of the algorithm
        as the gradient approximation quality depends on high diversity of samples.
        :return:
        """
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render: self.env.render()

            a = self.agent.act(s, self.env)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True
