from oscar.agent.custom_agent import CustomAgent
from oscar.util.micro_management import *


class MicroManager(CustomAgent):
    # states
    _IDLE = 0
    _MOVING_CAMERA = 1
    _MICRO_CONTROL = 2

    def __init__(self, message="I hate you"):
        self._message = message
        self._state = MicroManager._IDLE
        super().__init__()

    def step(self, obs, locked_choice=None):
        play = {}

        """ Selects new state """
        if self._state == MicroManager._IDLE:
            self._state = MicroManager._MOVING_CAMERA

        elif self._state == MicroManager._MOVING_CAMERA:
            self._state = MicroManager._MICRO_CONTROL

        elif self._state == MicroManager._MICRO_CONTROL:
            self._state = MicroManager._MOVING_CAMERA

        """ Executes states """
        if self._state == MicroManager._MOVING_CAMERA:
            micro_loc = get_micro_management_location(obs, self._shared)
            if len(micro_loc) > 0:
                selected_loc = random.choice(micro_loc)
                play['actions'] = [actions.FunctionCall(MOVE_CAMERA, [selected_loc.to_array()])]

        elif self._state == MicroManager._MICRO_CONTROL:
            print("micro control")
            play['actions'] = []

            friendly = self._shared['screen'].scan_units(obs, self._shared, [TERRAN_SCV, TERRAN_MARINE], PLAYER_SELF)
            enemy_influence_map = get_enemy_influence_map(obs, self._shared)

            for f in friendly:
                if f.location.screen.x >= 0 and f.location.screen.x < self._shared['screen'].width(obs) \
                        and f.location.screen.y >= 0 and f.location.screen.y < self._shared['screen'].height(obs) \
                        and enemy_influence_map[f.location.screen.x, f.location.screen.y] > 0:
                    play['actions'] += [actions.FunctionCall(SELECT_POINT,
                                                             [NEW_SELECTION,
                                                              f.location.screen.get_flipped().to_array()])]

                    safe_loc = get_safe_screen_location(obs, self._shared, f.location.screen, enemy_influence_map)
                    play['actions'] += [actions.FunctionCall(
                        MOVE_SCREEN, [NOT_QUEUED, safe_loc.get_flipped().to_array()])]

                    if f.unit_id == TERRAN_MARINE:
                        target = get_closest_enemy(obs, self._shared, safe_loc)
                        if target:
                            play['actions'] += [
                                actions.FunctionCall(ATTACK_SCREEN, [QUEUED, target.get_flipped().to_array()])]

        if 'actions' not in play or len(play['actions']) == 0:
            self._state = MicroManager._IDLE
            play['actions'] = [actions.FunctionCall(NO_OP, [])]

        elif self._state == MicroManager._MOVING_CAMERA:
            play['locked_choice'] = True

        return play

    def print_tree(self, depth):
        return "I am a {} and my depth is {}. I have a message to tell you : {}".format(type(self).__name__, depth
                                                                                        , self._message)
