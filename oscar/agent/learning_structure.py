

class LearningStructure:

    # must be set by sub classes
    observation_space = None
    action_space = None

    def __init__(self, train_mode=False, shared_memory=None):
        """
        Initialize custom agent and set step function according to train mode
        :param train_mode: indicate if the agent must be constructed in training mode or not
        """
        self.train_mode = train_mode
        self.shared_memory = shared_memory
        # check shared memory is consistent with train mode
        if train_mode is True and shared_memory is None:
            raise BadSharedMemoryError("When train mode is set to True a valid shared memory must be provided")
        # check if action/obs space are set by the sub classes
        if self.action_space is None or self.observation_space is None:
            raise RuntimeError("action_space and observation_space must be set before calling LearningStructure init !")

        # set memory with agent action/obs spaces
        shared_memory.action_space = self.action_space
        shared_memory.observation_space = self.observation_space

        # set which step method must be called
        if self.train_mode:
            self.do_step = self._learning_step
        else:
            self.do_step = self._step
        super().__init__()

    def step(self, obs):
        """
        Choose between calling normal step or training step.
        :param obs: observation used to choose the action to do
        :return: a dict of the agent's choice for this step (action list, callbacks)
        """
        return self.do_step(obs)

    def _step(self, obs):
        """
        Step function to be used when not in training mode, but when training but must be
        available.
        :param obs: observation used to choose the action to do
        :return: a dict of the agent's choice for this step (action list, callbacks)
        """
        # TODO: error message?
        raise NotImplementedError()

    def _learning_step(self, obs):
        """
        Step function to be used when in training mode
        :param obs: observation used to choose the action to do
        :return: a dict of the agent's choice for this step (action list, callbacks)
        """
        # transform the pysc2 obs into agent specific observation
        formatted_obs = self._format_observation(obs)
        # give the obs to the shared memory
        self.shared_memory.shared_obs = formatted_obs
        # set the semaphore to allow the learning thread to choose an action
        self.shared_memory.semaphore_obs_ready.release()
        # wait for the action to be selected
        self.shared_memory.semaphore_action_set.acquire(blocking=True, timeout=None)
        # transform the action into an hierarchical compatible action
        action = self.shared_memory.shared_action
        return self._transform_action(action_id=action)

    def _format_observation(self, full_obs):
        """
        Format the observation from the pysc2 format into an agent specific format
        :param full_obs: the observation structure as given by pysc2
        :return: an numpy array of the obs as wanted by the current agent
        """
        raise NotImplementedError()

    def _transform_action(self, action_id):
        """
        Transform the action decided by the agent (tipicaly a number in [0;n]) to a
        real agent action (dict of action list, callbacks...)
        :param action_id: number of the action chosen
        :return: agent play action in the hierarchy format (dict of action list, callbacks
         and lock)
        """
        raise NotImplementedError()


class BadSharedMemoryError(RuntimeError):
    """the given shared memory is not correctly setup"""

