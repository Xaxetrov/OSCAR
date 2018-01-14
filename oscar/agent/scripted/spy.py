from oscar.agent.custom_agent import CustomAgent
from oscar.meta_action import *
from oscar.util.camera import Camera
from oscar.shared.env import Env
import time


class Spy(CustomAgent):
    '''
    Positions units at strategic points and moves camera
    in order to collect information on the enemy.
    '''

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
        self._shared['idle_tracker'].update(obs, self._shared['env'].timestamp)

        point = Camera.location(obs)

        if self._state == Spy._INITIAL_STATE:
            self._target = get_spy_target(
                obs, 
                self._shared['enemy_tracker'],
                self._shared['env'].timestamp
                )

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
            self._shared['enemy_tracker'].scan_screen(obs, point, self._shared['env'].timestamp)
            play['actions'] = [actions.FunctionCall(NO_OP, [])]
            self._state = Spy._INITIAL_STATE

        elif self._state == Spy._SEND_UNIT:
            res = self._shared['idle_tracker'].search_idle_unit(obs, target=self._target)

            if res['unit']:
                play['actions'] = \
                    [actions.FunctionCall(SELECT_POINT, [NEW_SELECTION, res['unit'].location.screen.get_flipped().to_array()])] \
                    + [actions.FunctionCall(MOVE_MINIMAP, [NOT_QUEUED, self._target.to_array()])]
                play['success_callback'] = self.spy_sent
                self._state = Spy._INITIAL_STATE

            elif res['actions']:
                play['actions'] = res['actions']
                play['locked_choice'] = True

            else: # failed to find an idle unit
                play['actions'] = [actions.FunctionCall(NO_OP, [])]
                self._state = Spy._INITIAL_STATE

        return play


    def _is_location_visible(self, obs, location):
        _MIN_VISIBLE_RATIO = 0.5

        visibles, non_visibles = 0, 0
        for p in Camera.iterate(obs, location):
            if obs.observation['minimap'][MINI_VISIBILITY][p.y, p.x] == VISIBLE_CELL:
                visibles += 1
            else:
                non_visibles += 1

        return (visibles/(visibles+non_visibles) >= _MIN_VISIBLE_RATIO)


    def print_tree(self, depth):
        return 'I am a {} and my depth is {}. I have a message to tell you : {}'.format(type(self).__name__, depth
                                                                                        , self._message)
