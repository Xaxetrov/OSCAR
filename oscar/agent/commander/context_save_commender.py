from abc import ABC, abstractmethod

from oscar.agent.commander.base_commander import BaseCommander
from oscar.util.point import Point
from oscar.meta_action import *


class ContextSaveCommander(BaseCommander):
    def __init__(self, subordinates: list):
        super().__init__(subordinates=subordinates)
        self._subordinate_context = {}
        self._is_changing_context = False
        self.add_shared('env', Env())
        self.add_shared('camera', Camera())

    def step(self, obs, locked_choice=None):
        self._shared['env'].timestamp += 1

        if locked_choice is None:
            locked_choice = self._locked_choice

        # if we were changing context do not ask for choosing a subordinate
        if self._is_changing_context:
            self._is_changing_context = False
            playing_subordinate = self._playing_subordinate
        else:
            playing_subordinate = self.choose_subordinate(obs, locked_choice)

        # if we are changing active subordinate, save and restore context (require one action)
        if playing_subordinate is not self._playing_subordinate and self._playing_subordinate is not None:
            self.save_context(obs)
            play = self.restore_context(playing_subordinate, obs)
            self._playing_subordinate = playing_subordinate
        else:
            self._playing_subordinate = playing_subordinate
            play = self._playing_subordinate.step(obs, locked_choice)
            if "locked_choice" in play:
                self._locked_choice = play["locked_choice"]
            else:
                self._locked_choice = False
        return play

    def restore_context(self, subordinate, obs):
        if subordinate not in self._subordinate_context:
            print("context unavailable")
            context = AgentContext()
            location = self._shared['camera'].location(obs=obs, shared=self._shared)
            context.camera = location
        else:
            context = self._subordinate_context[subordinate]
        play = {}
        play['actions'] = [actions.FunctionCall(MOVE_CAMERA, [context.camera.to_array()])]
        play['locked_choice'] = True
        self._is_changing_context = True
        return play

    def save_context(self, obs):
        context = AgentContext()
        location = self._shared['camera'].location(obs=obs, shared=self._shared)
        context.camera = location
        self._subordinate_context[self._playing_subordinate] = context

    @abstractmethod
    def choose_subordinate(self, obs, locked_choice):
        """
        Choose a subordinate among the list of subordinates, and make it play.
        :return: A subordinate among the list of subordinates.
        """


class AgentContext:
    def __init__(self):
        self.camera = Point()

