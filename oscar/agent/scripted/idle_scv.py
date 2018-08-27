from oscar.agent.custom_agent import CustomAgent
from oscar.constants import *
from oscar.util.selection import *
from oscar.shared.env import Env
from oscar.shared.camera import Camera
from oscar.meta_action.select import select_idle_scv
from oscar.meta_action.harvest import harvest_mineral
from oscar.meta_action.attack import attack_minimap


class IdleSCVManagerBasic(CustomAgent):
    def __init__(self):
        super().__init__()
        self.command_center_pos = None
        self.add_shared('env', Env())
        self.add_shared('camera', Camera())

    def step(self, obs, locked_choice=None):
        self._shared['env'].timestamp += 1
        if self._shared['env'].timestamp == 1:
            self.command_center_pos = self._shared['camera'].location(obs=obs, shared=self._shared)
        play = {}
        # if a scv is already selected it was most probably selected during last call
        # but in case we just queue the harvest order
        if obs.observation['single_select'][0][0] == TERRAN_SCV:
            try:
                play['actions'] = harvest_mineral(obs, queued=True)
            except NoUnitError:
                # if we get here, chance are that no more command center exist, send scv to attack !
                play['actions'] = attack_minimap(obs, queued=True)
            # select a new idle scv to prevent looping on the same
            try:
                play['actions'] += select_idle_scv(obs)
            except NoValidSCVError:
                pass
        elif SELECT_IDLE_WORKER in obs.observation['available_actions']:
            play['actions'] = select_idle_scv(obs)
            try:
                # if we still have a command center somewhere
                if obs.observation[PLAYER][FOOD_CAP] % 8 != 0:
                    play['actions'] += harvest_mineral(obs)
                else:
                    play['actions'] += attack_minimap(obs, queued=True)
            except NoUnitError:
                play['actions'] += [actions.FunctionCall(MOVE_CAMERA, [self.command_center_pos.to_array()])]
                play['locked_choice'] = True
        else:
            play['actions'] = [actions.FunctionCall(NO_OP, [])]

        return play

    def reset(self):
        super().reset()
        self.command_center_pos = None
        self.add_shared('env', Env())
        self.add_shared('camera', Camera())
