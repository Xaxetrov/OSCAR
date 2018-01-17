import threading


class SharedObjects:
    """
    Container for shared object between the game thread and the learning thread.
    Share observations and actions during training
    """

    def __init__(self):
        # Semaphore unlocked when the observation are set
        self.semaphore_obs_ready = threading.Semaphore(value=0)
        # Semaphore unlocked when the action is set
        self.semaphore_action_set = threading.Semaphore(value=0)
        # shared memory containing the action that must be done by the env
        self.shared_action = None
        # shared memory containing the observation get by the env
        self.shared_obs = None
        # number of possible action (set by the env)
        self.action_space = None
        # shape of the observations given by the env
        self.observation_space = None
        # vector masking defining the action that are available or not
        self.available_action_mask = None
