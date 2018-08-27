from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action import *
from oscar.shared.camera import Camera
from oscar.shared.env import Env
import time


class Spy(CustomAgent):
    """
    Positions units at strategic locations and moves camera
    in order to collect information on the enemy.
    """

    # states
    _INITIAL_STATE = 0
    _SCREEN_SCAN = 1
    _SEND_UNIT = 2

    def __init__(self, message=''):
        self._message = message
        super().__init__()

        # state machine
        self._state = Spy._INITIAL_STATE
        self._target = None

    def spy_sent(self):
        print('spy sent')

    def step(self, obs, locked_choice=None):

        play = {}
        self._shared['env'].timestamp += 1
        self._shared['idle_tracker'].update(obs, self._shared)

        if self._state == Spy._INITIAL_STATE:
            self._target = get_spy_target(obs, self._shared)

            if not self._target:
                play['actions'] = [actions.FunctionCall(NO_OP, [])]

            else:
                if self._is_location_visible(obs, self._target):
                    play['actions'] = [actions.FunctionCall(MOVE_CAMERA, [self._target.to_array()])]
                    play['locked_choice'] = True
                    self._state = Spy._SCREEN_SCAN
                else:
                    play['actions'] = [actions.FunctionCall(NO_OP, [])]
                    play['locked_choice'] = True
                    self._state = Spy._SEND_UNIT

        elif self._state == Spy._SCREEN_SCAN:
            self._shared['enemy_tracker'].scan_screen(obs, self._shared)
            play['actions'] = [actions.FunctionCall(NO_OP, [])]
            self._state = Spy._INITIAL_STATE

        elif self._state == Spy._SEND_UNIT:
            res = self._shared['idle_tracker'].search_idle_unit(obs, self._shared, target=self._target)

            if res['unit']:
                play['actions'] = \
                    [actions.FunctionCall(SELECT_POINT,
                                          [NEW_SELECTION, res['unit'].location.screen.get_flipped().to_array()])] \
                    + [actions.FunctionCall(MOVE_MINIMAP, [NOT_QUEUED, self._target.to_array()])]
                play['success_callback'] = self.spy_sent
                self._state = Spy._INITIAL_STATE

            elif res['actions']:
                play['actions'] = res['actions']
                play['locked_choice'] = True

            else:  # failed to find an idle unit
                play['actions'] = [actions.FunctionCall(NO_OP, [])]
                self._state = Spy._INITIAL_STATE

        return play

    def _is_location_visible(self, obs, location):
        _MIN_VISIBLE_RATIO = 0.5

        visible, non_visible = 0, 0
        for p in self._shared['camera'].iterate(obs, location):
            if obs.observation[MINIMAP][MINI_VISIBILITY][p.y, p.x] == VISIBLE_CELL:
                visible += 1
            else:
                non_visible += 1

        return visible / (visible + non_visible) >= _MIN_VISIBLE_RATIO

    def print_tree(self, depth):
        return 'I am a {} and my depth is {}. I have a message to tell you : {}'.format(type(self).__name__, depth
                                                                                        , self._message)
